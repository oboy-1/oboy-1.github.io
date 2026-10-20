// Slide "spectrum", second panel: add the layer's real waves back one at a time, strongest first.
import { el, paintField, maxAbs, fmt, pct, lut } from "../render.js";
import { fft2, ifft2, splitAtCutoff, keptMask } from "../fourier.js";

const DECADES = 6;
const maskCache = new Map();
const mask = (res) => { if (!maskCache.has(res)) maskCache.set(res, keptMask(res)); return maskCache.get(res); };

function rgb(v) {
  const s = getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  if (s.startsWith("#")) { const h = s.length === 4 ? [...s.slice(1)].map((c) => c + c).join("") : s.slice(1); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
  const m = s.match(/\d+(\.\d+)?/g);
  return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
}

export function buildUpPanel(app, { l }) {
  const s = app.sample;
  const res = s.live ? s.live.res : 64;
  const h = res / 2, N = res * res;
  const inFields = s.live ? (l === 1 ? s.live.lifted : s.live.layers[l - 2].out) : s.fields[l === 1 ? "lifted" : `layer${l - 1}.out`];

  // Same channel as the truncation step: the one with the most detail above the cutoff.
  const km = mask(res);
  let best = 0, bestHi = -1;
  inFields.forEach((f, c) => {
    const sp = splitAtCutoff(f, res, km);
    let e = 0; for (const v of sp.high) e += v * v;
    if (e > bestHi) { bestHi = e; best = c; }
  });
  const x = inFields[best];
  const vm = maxAbs(x);
  const F = fft2(x, res);

  // A real image's waves come in mirror pairs (k and -k are one wave), so count pairs.
  const energy = new Float64Array(N);
  for (let i = 0; i < N; i++) energy[i] = F.re[i] ** 2 + F.im[i] ** 2;
  const partner = (i) => { const a = Math.floor(i / res), b = i % res; return ((res - a) % res) * res + ((res - b) % res); };
  const waves = [];
  let totalAC = 0;
  for (let i = 0; i < N; i++) {
    const p = partner(i);
    if (p < i) continue;
    const e = p === i ? energy[i] : energy[i] + energy[p];
    waves.push({ i, p, e });
    if (i !== 0) totalAC += e;
  }
  waves.sort((a, b) => b.e - a.e);
  const NW = waves.length;
  const cum = new Float64Array(NW + 1);
  for (let j = 0; j < NW; j++) cum[j + 1] = cum[j] + (waves[j].i === 0 ? 0 : waves[j].e);

  // Heat colours for the grid, same log scale as the heatmap above.
  const logE = (i) => Math.log10(energy[i] / (N * N) + 1e-30);
  let hi = -Infinity;
  for (let i = 0; i < N; i++) hi = Math.max(hi, logE(i));
  const lo = hi - DECADES;
  const blues = lut("Blues");
  const ground = rgb("--panel");
  const cellColor = (i) => {
    const t = Math.max(0, Math.min(1, (logE(i) - lo) / (hi - lo)));
    const j = Math.round(t * 255) * 3;
    return [blues[j], blues[j + 1], blues[j + 2]];
  };
  const base = new Array(N);
  for (let i = 0; i < N; i++) base[i] = cellColor(i);
  const cen = (i) => (((Math.floor(i / res) + h) % res) * res + ((i % res) + h) % res);

  // Canvases
  const grid = el("canvas", { width: res, height: res, "aria-label": "Waves added so far, on the spectrum grid" });
  const over = el("canvas", { width: res * 8, height: res * 8, "aria-hidden": "true" });
  const recCv = el("canvas"), realCv = el("canvas"), newCv = el("canvas");
  paintField(realCv, x, res, { vmax: vm });

  let cur = new Float64Array(N);
  const V = { az: 0.7, elev: 0.75 };
  const W3 = 620, H3 = 320;
  const cv3 = el("canvas", { class: "bu-3d", width: W3 * 2, height: H3 * 2, role: "img", "aria-label": "Rotatable 3-D surface of the rebuilt channel. Drag to rotate." });
  function draw3d() {
    const ctx = cv3.getContext("2d");
    ctx.setTransform(2, 0, 0, 2, 0, 0);
    ctx.clearRect(0, 0, W3, H3);
    const L = lut("RdBu_r");
    const st = Math.max(1, Math.round(res / 40)), M = Math.floor((res - 1) / st) + 1;
    const ca = Math.cos(V.az), sa = Math.sin(V.az), ce = Math.cos(V.elev), se = Math.sin(V.elev);
    const SC = 130, cx = W3 / 2, cy = H3 * 0.58, ZS = 0.9 / vm;
    const P = (i, j, v) => {
      const x = (i / (M - 1)) * 2 - 1, y = (j / (M - 1)) * 2 - 1, z = v * ZS;
      const X = x * ca - y * sa, Y = x * sa + y * ca;
      return { sx: cx + SC * X, sy: cy - SC * (z * ce + Y * se * 0.9), d: Y * ce - z * se, v };
    };
    const grab = (f) => { const a = new Array(M * M); for (let j = 0; j < M; j++) for (let i = 0; i < M; i++) a[j * M + i] = P(i, j, f[(i * st) * res + j * st]); return a; };
    const pts = grab(cur), ref = grab(x);
    const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    ctx.strokeStyle = css("--rule-strong"); ctx.lineWidth = 1; ctx.beginPath();
    [[0, 0], [M - 1, 0], [M - 1, M - 1], [0, M - 1]].map(([i, j]) => P(i, j, -vm)).forEach((c, k) => (k ? ctx.lineTo(c.sx, c.sy) : ctx.moveTo(c.sx, c.sy)));
    ctx.closePath(); ctx.stroke();
    const quads = [];
    for (let j = 0; j < M - 1; j++) for (let i = 0; i < M - 1; i++) {
      const a = pts[j * M + i], b = pts[j * M + i + 1], c = pts[(j + 1) * M + i + 1], d = pts[(j + 1) * M + i];
      quads.push({ q: [a, b, c, d], d: (a.d + b.d + c.d + d.d) / 4, v: (a.v + b.v + c.v + d.v) / 4 });
    }
    quads.sort((u, w) => w.d - u.d);
    ctx.lineWidth = 0.4; ctx.strokeStyle = "rgba(20,24,28,0.22)";
    for (const { q, v } of quads) {
      const k = Math.round(Math.max(0, Math.min(1, (v / vm + 1) / 2)) * 255) * 3;
      ctx.fillStyle = `rgb(${L[k]},${L[k + 1]},${L[k + 2]})`;
      ctx.beginPath(); ctx.moveTo(q[0].sx, q[0].sy);
      for (let m = 1; m < 4; m++) ctx.lineTo(q[m].sx, q[m].sy);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    // the real channel as a thin ghost mesh, so you see the gap close
    ctx.strokeStyle = css("--ink"); ctx.globalAlpha = 0.35; ctx.lineWidth = 0.6;
    for (let j = 0; j < M; j += 2) { ctx.beginPath(); for (let i = 0; i < M; i++) { const p = ref[j * M + i]; i ? ctx.lineTo(p.sx, p.sy) : ctx.moveTo(p.sx, p.sy); } ctx.stroke(); }
    for (let i = 0; i < M; i += 2) { ctx.beginPath(); for (let j = 0; j < M; j++) { const p = ref[j * M + i]; j ? ctx.lineTo(p.sx, p.sy) : ctx.moveTo(p.sx, p.sy); } ctx.stroke(); }
    ctx.globalAlpha = 1;
  }
  let drag = null;
  cv3.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; cv3.setPointerCapture(e.pointerId); });
  cv3.addEventListener("pointermove", (e) => {
    if (!drag) return;
    V.az += (e.clientX - drag.x) * 0.012;
    V.elev = Math.max(0.15, Math.min(1.45, V.elev + (e.clientY - drag.y) * 0.008));
    drag = { x: e.clientX, y: e.clientY };
    draw3d();
  });
  cv3.addEventListener("pointerup", () => { drag = null; }); cv3.addEventListener("pointercancel", () => { drag = null; });

  const acc = { re: new Float64Array(N), im: new Float64Array(N) };
  let n = 0, applied = 0;

  function paintGrid() {
    const ctx = grid.getContext("2d");
    const img = ctx.createImageData(res, res);
    const on = new Uint8Array(N);
    for (let j = 0; j < n; j++) { on[cen(waves[j].i)] = 1; on[cen(waves[j].p)] = 1; }
    for (let i = 0; i < N; i++) {
      const c = cen(i);
      const col = base[i];
      const f = on[c] ? 1 : 0.14;
      img.data.set([ground[0] + (col[0] - ground[0]) * f, ground[1] + (col[1] - ground[1]) * f, ground[2] + (col[2] - ground[2]) * f, 255], c * 4);
    }
    ctx.putImageData(img, 0, 0);
  }

  const cutoffOutline = () => {
    const o = over.getContext("2d");
    o.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink-3").trim() || "#888";
    o.lineWidth = 1.5;
    const sc = over.width / res;
    const seg = (a, b, c, d) => { o.beginPath(); o.moveTo(a, b); o.lineTo(c, d); o.stroke(); };
    for (let a = 0; a < res; a++) for (let b = 0; b < res; b++) {
      if (!km[a * res + b]) continue;
      if (a === 0 || !km[(a - 1) * res + b]) seg(b * sc, a * sc, (b + 1) * sc, a * sc);
      if (a === res - 1 || !km[(a + 1) * res + b]) seg(b * sc, (a + 1) * sc, (b + 1) * sc, (a + 1) * sc);
      if (b === 0 || !km[a * res + b - 1]) seg(b * sc, a * sc, b * sc, (a + 1) * sc);
      if (b === res - 1 || !km[a * res + b + 1]) seg((b + 1) * sc, a * sc, (b + 1) * sc, (a + 1) * sc);
    }
  };

  function highlight() {
    const o = over.getContext("2d");
    o.clearRect(0, 0, over.width, over.height);
    cutoffOutline();
    if (!n) return;
    const sc = over.width / res;
    const w = waves[n - 1];
    o.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--spectral").trim();
    o.strokeStyle = "#fff";
    o.lineWidth = 1.5;
    for (const i of w.i === w.p ? [w.i] : [w.i, w.p]) {
      const c = cen(i), a = Math.floor(c / res), b = c % res;
      const pad = 4;
      o.beginPath();
      o.rect(b * sc - pad, a * sc - pad, sc + 2 * pad, sc + 2 * pad);
      o.fill(); o.stroke();
    }
  }

  const slider = el("input", { type: "range", min: "0", max: "1000", step: "1", value: "0", "aria-label": "Number of waves added" });
  const count = el("b", { class: "mono" }, "");
  const info = el("div", { class: "bu-info mono" });
  const marker = el("span", { class: "bu-tick", "aria-hidden": "true" }, el("i", {}), el("span", {}, ""));
  const prev = el("button", { class: "bu-btn", "aria-label": "Remove the last wave" }, "−1");
  const next = el("button", { class: "bu-btn", "aria-label": "Add the next wave" }, "+1");
  const play = el("button", { class: "bu-btn play" }, "▶ Play");

  let keptPairs = 0, keptE = 0;
  for (const w of waves) if (km[cen(w.i)] || km[cen(w.p)]) { keptPairs++; if (w.i !== 0) keptE += w.e; }
  const keptShare = totalAC > 0 ? keptE / totalAC : 0;
  const posOf = (k) => Math.round((1000 * Math.log(k + 1)) / Math.log(NW + 1));
  const nOf = (p) => Math.round(Math.exp((p / 1000) * Math.log(NW + 1)) - 1);

  marker.style.left = `calc(8px + (100% - 16px) * ${posOf(keptPairs) / 1000})`;
  marker.lastChild.textContent = `network keeps ${keptPairs} pairs`;
  function update() {
    // Bring the running sum from `applied` waves to `n` waves.
    while (applied < n) { const w = waves[applied++]; for (const i of w.i === w.p ? [w.i] : [w.i, w.p]) { acc.re[i] = F.re[i]; acc.im[i] = F.im[i]; } }
    while (applied > n) { const w = waves[--applied]; for (const i of w.i === w.p ? [w.i] : [w.i, w.p]) { acc.re[i] = 0; acc.im[i] = 0; } }
    cur = ifft2(acc, res).re;
    paintField(recCv, cur, res, { vmax: vm });
    draw3d();
    if (n) {
      const w = waves[n - 1];
      const one = { re: new Float64Array(N), im: new Float64Array(N) };
      for (const i of w.i === w.p ? [w.i] : [w.i, w.p]) { one.re[i] = F.re[i]; one.im[i] = F.im[i]; }
      const single = ifft2(one, res).re;
      paintField(newCv, single, res, { vmax: Math.max(maxAbs(single), 1e-9) });
    } else {
      newCv.getContext("2d").clearRect(0, 0, newCv.width, newCv.height);
    }
    paintGrid();
    highlight();
    slider.value = String(posOf(n));
    count.textContent = `${n.toLocaleString("en-US")} of ${NW.toLocaleString("en-US")} waves`;
    prev.disabled = n === 0;
    next.disabled = n === NW;
    const got = totalAC > 0 ? cum[n] / totalAC : 0;
    if (!n) {
      info.replaceChildren(el("div", {}, "Nothing added yet. Press +1, drag the slider, or play."));
    } else {
      const w = waves[n - 1];
      const a = Math.floor(w.i / res), b = w.i % res;
      const kx = a <= h ? a : a - res, ky = b <= h ? b : b - res;
      const inside = km[cen(w.i)] || km[cen(w.p)];
      info.replaceChildren(
        el("div", {}, w.i === 0 ? "wave " + n + ": the flat average" : `wave ${n}: kx ${Math.abs(kx) === 0 ? 0 : kx}, ky ${ky} · ${fmt(Math.hypot(kx, ky), 1)} cycles across`),
        el("div", {}, `energy captured so far: ${pct(got, 1)}`, el("span", { class: "bu-dim" }, " (not counting the flat average)")),
        el("div", { class: inside ? "t-spectral" : "" }, inside ? "inside the network's kept block" : "outside the network's kept block"),
        el("div", { class: "bu-dim" }, `the network's kept block holds ${keptPairs} of these waves and ${pct(keptShare, 1)} of the energy`));
    }
  }

  function set(k) { n = Math.max(0, Math.min(NW, k)); update(); }

  slider.addEventListener("input", () => { stop(); set(nOf(+slider.value)); });
  prev.addEventListener("click", () => { stop(); set(n - 1); });
  next.addEventListener("click", () => { stop(); set(n + 1); });

  let raf = 0, t0 = 0, from = 0;
  const DUR = 9000;
  function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; play.textContent = "▶ Play"; }
  function tick(t) {
    const u = Math.min(1, (t - t0) / DUR);
    set(Math.round(Math.exp(u * Math.log(NW + 1 - from) ) - 1 + from));
    if (u < 1) raf = requestAnimationFrame(tick); else stop();
  }
  play.addEventListener("click", () => {
    if (raf) { stop(); return; }
    if (n >= NW) set(0);
    from = n; t0 = performance.now(); play.textContent = "❚❚ Pause";
    raf = requestAnimationFrame(tick);
  });

  const tile = (cv, label, sub) => el("figure", { class: "ftile bu-tile" }, el("div", { class: "tile" }, cv),
    el("figcaption", {}, el("span", { class: "cap-t" }, label), sub ? el("span", { class: "cap-s" }, sub) : null));

  const out = el("div", { class: "bu" },
    el("h4", { class: "drill-sub" }, "2 · Putting a channel back together"),
    el("p", { class: "lead-in" }, `Those squares really are the input. To see it, take channel ${best} and add its waves back one at a time, strongest first. The rebuild gets closer and closer to the real channel.`),
    el("div", { class: "bu-row" },
      el("figure", { class: "ftile bu-tile bu-gridfig" },
        el("div", { class: "heat-stack bu-stack" }, grid, over),
        el("figcaption", {}, el("span", { class: "cap-t" }, "its waves, strongest first"), el("span", { class: "cap-s" }, "orange: the one just added · grey outline: the network's kept block"))),
      tile(recCv, "rebuilt from the waves so far", "one colour scale"),
      tile(realCv, "the real channel", "the target"),
      tile(newCv, "the wave just added", "scaled to itself")),
    el("div", { class: "bu-ctl" },
      el("div", { class: "tw-slider" }, el("span", {}, "waves added, strongest first"), count),
      el("div", { class: "bu-ctlrow" }, prev, el("div", { class: "tw bu-sl" }, slider, marker), next, play)),
    info,
    el("div", { class: "bu-3dwrap" }, cv3,
      el("p", { class: "cap-s" }, "The same rebuild as a 3-D surface, where height is the channel's value. The thin black mesh is the real channel, so you can watch the surface settle onto it. Drag to turn it.")),
    el("p", { class: "cap-s" }, `Each wave goes in with its real size and shift. Waves come in mirror pairs, so each step lights up two squares. The slider is stretched so the first few waves, which do most of the work, get more room. The orange tick marks how many pairs the network keeps: ${keptPairs}, which is the ${app.manifest.stats.n_kept_modes} squares inside the outline, paired up (the flat average is its own pair). Strongest first isn't quite the same order as slowest first, so a few waves the network drops show up before the tick. Once all ${NW.toLocaleString("en-US")} are in, the rebuild matches the channel exactly.`));

  update();
  return out;
}
