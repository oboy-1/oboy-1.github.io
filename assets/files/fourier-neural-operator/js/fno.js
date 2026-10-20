// FNO forward pass in plain JS, float64. Port of fno_reference.py (itself verified against
// PyTorch). Checked numerically by tools/verify_forward.mjs before the UI uses it.
import { gelu, pearson, spatialStd } from "./nn.js";
import { fft2, synthesize, centeredEnergy, keptMask, truncate } from "./fourier.js";

const C = 32;

function conv1x1(x, t, bias, res) {
  // x: array of Cin Float64Array(res*res); t.shape = [Cout, Cin, 1]
  const [cout, cin] = t.shape;
  const n = res * res;
  const out = [];
  for (let o = 0; o < cout; o++) {
    const y = new Float64Array(n);
    const b = bias ? bias.data[o] : 0;
    for (let i = 0; i < n; i++) y[i] = b;
    for (let c = 0; c < cin; c++) {
      const w = t.data[o * cin + c];
      if (w === 0) continue;
      const src = x[c];
      for (let i = 0; i < n; i++) y[i] += w * src[i];
    }
    out.push(y);
  }
  return out;
}

const gelus = (xs) => xs.map((f) => f.map(gelu));

function spectralConv(x, W, bias, res) {
  // W.shape = [Cin, Cout, 16, 9], complex interleaved; bias.shape = [Cout, 1, 1]
  const [cin, cout, M0, M1] = W.shape;
  const N = res * res;
  const kept = new Float64Array(cin * M0 * M1 * 2);
  for (let b = 0; b < cin; b++) {
    const F = fft2(x[b], res);
    for (let ix = 0; ix < M0; ix++) {
      const r = (((ix - M0 / 2) % res) + res) % res;
      for (let iy = 0; iy < M1; iy++) {
        const k = ((b * M0 + ix) * M1 + iy) * 2;
        kept[k] = F.re[r * res + iy] / N;
        kept[k + 1] = F.im[r * res + iy] / N;
      }
    }
  }
  const out = [];
  for (let e = 0; e < cout; e++) {
    const coef = new Float64Array(M0 * M1 * 2);
    for (let b = 0; b < cin; b++) {
      for (let ix = 0; ix < M0; ix++) {
        for (let iy = 0; iy < M1; iy++) {
          const xk = ((b * M0 + ix) * M1 + iy) * 2;
          const wk = (((b * cout + e) * M0 + ix) * M1 + iy) * 2;
          const xr = kept[xk], xi = kept[xk + 1], wr = W.data[wk], wi = W.data[wk + 1];
          const ck = (ix * M1 + iy) * 2;
          coef[ck] += xr * wr - xi * wi;
          coef[ck + 1] += xr * wi + xi * wr;
        }
      }
    }
    const y = synthesize(res, (kx, ky) => {
      const ck = ((kx + M0 / 2) * M1 + ky) * 2;
      return [coef[ck], coef[ck + 1]];
    });
    const bb = bias.data[e];
    for (let i = 0; i < N; i++) y[i] += bb;
    out.push(y);
  }
  return out;
}

/**
 * Run the model on one input map. Returns every intermediate the views use, in the same shapes
 * as the precomputed bundle, plus prediction in physical units.
 */
export function forward(model, input, res, { yMean, yStd }) {
  const w = model;
  const N = res * res;
  const gx = new Float64Array(N), gy = new Float64Array(N), a = new Float64Array(N);
  for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
    gx[i * res + j] = i / res; gy[i * res + j] = j / res; a[i * res + j] = input[i * res + j];
  }
  let x = conv1x1(gelus(conv1x1([a, gx, gy], w["lift0.w"], w["lift0.b"], res)), w["lift1.w"], w["lift1.b"], res);
  const lifted = x;
  const layers = [];
  for (let l = 0; l < 4; l++) {
    const spectral = spectralConv(x, w[`L${l}.spec.w`], w[`L${l}.spec.b`], res);
    const local = conv1x1(x, w[`L${l}.skip.w`], null, res);
    const pre = spectral.map((s, c) => { const y = new Float64Array(N); const lc = local[c]; for (let i = 0; i < N; i++) y[i] = s[i] + lc[i]; return y; });
    const act = l < 3 ? gelus(pre) : pre;
    const mlp = conv1x1(gelus(conv1x1(act, w[`L${l}.mlp0.w`], w[`L${l}.mlp0.b`], res)), w[`L${l}.mlp1.w`], w[`L${l}.mlp1.b`], res);
    const gate = w[`L${l}.gate.w`].data;
    let out = mlp.map((m, c) => { const y = new Float64Array(N); const g = gate[c], src = x[c]; for (let i = 0; i < N; i++) y[i] = m[i] + g * src[i]; return y; });
    if (l < 3) out = gelus(out);
    layers.push({ spectral, local, out });
    x = out;
  }
  const q = conv1x1(gelus(conv1x1(x, w["proj0.w"], w["proj0.b"], res)), w["proj1.w"], w["proj1.b"], res)[0];
  const prediction = q.map((v) => v * yStd + yMean);
  return { res, lifted, layers, prediction };
}

const maskCache = new Map();
function maskFor(res) {
  if (!maskCache.has(res)) maskCache.set(res, keptMask(res));
  return maskCache.get(res);
}

/** Per-layer statistics, same definitions as export_web.py. */
export function layerStats(run) {
  const { res } = run;
  const N = res * res, h = res / 2;
  const K = maskFor(res);
  const dc = h * res + h;
  let prev = run.lifted;
  return run.layers.map((L) => {
    const sum = L.spectral.map((s, c) => s.map((v, i) => v + L.local[c][i]));
    const Es = centeredEnergy(L.spectral, res), El = centeredEnergy(L.local, res), Et = centeredEnergy(sum, res);
    const Ein = centeredEnergy(prev, res), Etr = centeredEnergy(prev.map((f) => truncate(f, res)), res);
    let sO = 0, lO = 0, tO = 0, sT = 0, lT = 0, inAll = 0, inNoDc = 0, trAll = 0, trNoDc = 0;
    for (let i = 0; i < N; i++) {
      sT += Es[i]; lT += El[i]; inAll += Ein[i]; trAll += Etr[i];
      if (i !== dc) { inNoDc += Ein[i]; trNoDc += Etr[i]; }
      if (!K[i]) { sO += Es[i]; lO += El[i]; tO += Et[i]; }
    }
    let best = 0, bestV = -1, ss = 0, ls = 0;
    for (let c = 0; c < C; c++) {
      const a = spatialStd(L.spectral[c]), b = spatialStd(L.local[c]);
      ss += a * a; ls += b * b;
      if (a + b > bestV) { bestV = a + b; best = c; }
    }
    prev = L.out;
    return {
      outside_cutoff_share_spectral: sO / tO,
      outside_cutoff_share_local: lO / tO,
      spectral_own_energy_outside_cutoff: sO / sT,
      local_own_energy_outside_cutoff: lO / lT,
      truncation_kept_incl_mean: trAll / inAll,
      truncation_kept_excl_mean: trNoDc / inNoDc,
      std_spectral: Math.sqrt(ss / C),
      std_local: Math.sqrt(ls / C),
      default_channel: best,
    };
  });
}

/** spectra[layer][kind]: log10 channel-summed energy, (0,0) centered. kind 0 input, 1 truncated, 2 spectral path. */
export function spectra(run) {
  const { res } = run;
  const K = maskFor(res);
  let prev = run.lifted;
  return run.layers.map((L) => {
    const Ein = centeredEnergy(prev, res);
    const Etr = centeredEnergy(prev.map((f) => truncate(f, res)), res);
    const Esp = centeredEnergy(L.spectral, res);
    prev = L.out;
    const log = (E, masked) => Float32Array.from(E, (v, i) => (masked && !K[i] ? -30 : Math.log10(v + 1e-30)));
    return [log(Ein, false), log(Etr, true), log(Esp, false)];
  });
}

/** Per-channel correlation with the input map (and target, when there is one), keyed like the manifest. */
export function channelCorrelations(run, input, target) {
  const stages = { lifted: run.lifted };
  run.layers.forEach((L, i) => { stages[`layer${i + 1}`] = L.out; });
  const r_in = {}, r_u = {};
  for (const [k, fields] of Object.entries(stages)) {
    r_in[k] = fields.map((f) => pearson(f, input));
    if (target) r_u[k] = fields.map((f) => pearson(f, target));
  }
  return { r_in, r_u: target ? r_u : null };
}
