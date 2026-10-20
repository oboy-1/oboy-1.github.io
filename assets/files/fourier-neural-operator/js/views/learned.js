// Stages for the three steps between "waves" and "the network": how a picture is stored as waves,
// what a layer does to each wave (the learned complex number R), and where all of this stops working.
// Everything here is read off the real rock, the real trained weights, or the exported manifest.
import { el, paintField, paintBinary, paintInput, words, maxAbs, fmt, pct } from "../render.js";
import { fft2, MODES } from "../fourier.js";

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const TAU = 2 * Math.PI;
const wave = (res, kx, ky, amp, phase) => {
  const w = new Float64Array(res * res);
  for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) w[i * res + j] = amp * Math.cos((TAU * (kx * i + ky * j)) / res + phase);
  return w;
};
const deg = (r) => `${Math.round((r * 180) / Math.PI)}°`;

// ---------------------------------------------------------------- stored
/** Each wave is just a few numbers. Show them for this rock, strongest first. */
export function storedStage(viz, app) {
  const res = app.state.res;
  const input = app.sample.io[res].input;
  const N = res * res;
  const F = fft2(input, res);
  const mean = F.re[0] / N;

  // Canonical half of the spectrum: ky > 0, or ky = 0 with kx > 0 (the other half is its mirror image).
  const rows = [];
  for (let kx = MODES.kxLo; kx <= MODES.kxHi; kx++) {
    for (let ky = MODES.kyLo; ky <= MODES.kyHi; ky++) {
      if (ky === 0 && kx <= 0) continue;
      const i = (((kx % res) + res) % res) * res + ky;
      const re = F.re[i] / N, im = F.im[i] / N;
      const h = 2 * Math.hypot(re, im); // height of the real wave this mode and its mirror make together
      rows.push({ kx, ky, h, phase: Math.atan2(im, re), E: h * h });
    }
  }
  let total = 0;
  for (const r of rows) total += r.E;
  // Energy of everything outside the kept block, so the "everything else" line is honest.
  let all = 0;
  for (let x = 0; x < res; x++) for (let y = 0; y < res; y++) {
    if (x === 0 && y === 0) continue;
    all += (F.re[x * res + y] ** 2 + F.im[x * res + y] ** 2) / (N * N);
  }
  all *= 2; // 2 * variance = sum of squared wave heights
  const top = rows.sort((a, b) => b.E - a.E).slice(0, 8);
  const vmax = top[0].h;
  const shown = top.reduce((s, r) => s + r.E, 0);

  const head = el("div", { class: "st-row st-head" },
    ...["", "wave", "repeats", "height", "starts at", "share of energy"].map((t) => el("span", {}, t)));
  const body = top.map((r) => {
    const cv = el("canvas");
    paintField(cv, wave(res, r.kx, r.ky, r.h, r.phase), res, { vmax });
    const cyc = Math.hypot(r.kx, r.ky);
    return el("div", { class: "st-row" },
      el("span", { class: "st-tile" }, cv),
      el("span", { class: "mono" }, `kx ${r.kx}, ky ${r.ky}`),
      el("span", { class: "mono" }, `${fmt(cyc, 1)}× across`),
      el("span", { class: "mono" }, fmt(r.h, 3)),
      el("span", { class: "mono" }, deg(r.phase)),
      el("span", { class: "st-share" }, el("span", { class: "st-bar", style: `width:${(r.E / top[0].E) * 100}%` }), el("span", { class: "mono" }, pct(r.E / all, 1))));
  });

  const rockCv = el("canvas");
  paintInput(rockCv, input, res);
  viz.append(el("div", { class: "panel-body stored" },
    el("h4", { class: "drill-sub" }, `This ${words().input}, as a list of waves`),
    el("div", { class: "st-top" },
      el("figure", { class: "ftile st-rock" }, el("div", { class: "tile" }, rockCv), el("figcaption", {}, el("span", { class: "cap-t" }, `the ${words().input}`), el("span", { class: "cap-s" }, `${res}×${res}, ${N.toLocaleString("en-US")} pixels`))),
      el("div", { class: "st-readout" },
        el("div", { class: "st-big mono" }, `${top.length} waves`),
        el("div", { class: "cap-s" }, `already hold ${pct(shown / all, 0)} of this map's energy (not counting its average, ${fmt(mean, 3)}).`))),
    el("div", { class: "st-table" }, head, ...body),
    el("p", { class: "cap-s" }, `Each row is one wave of the ${words().input}, strongest first. Each tile is that wave on its own, drawn at the height and shift in the row. The network keeps ${app.manifest.stats.n_kept_modes} waves per channel in all, so that is ${app.manifest.stats.n_kept_modes * 2} numbers (a height and a shift each) standing in for ${N.toLocaleString("en-US")} pixels.`)));
}

// ---------------------------------------------------------------- multiply
const S = { l: 2, b: 0, e: 0, ix: 9, iy: 1 }; // survives re-renders; ix = kx + 8

/** One learned number R for one wave: it scales the wave and slides it sideways. */
export function multiplyStage(viz, app) {
  const m = app.manifest;
  const W = app.model?.[`L${S.l - 1}.spec.w`];
  if (!W) { viz.append(el("div", { class: "panel-body" }, el("p", { class: "cap-s" }, "Loading the trained weights…"))); return; }
  const [cin, cout, M0, M1] = W.shape;
  const at = (b, e, ix, iy) => { const k = (((b * cout + e) * M0 + ix) * M1 + iy) * 2; return [W.data[k], W.data[k + 1]]; };

  const host = el("div", { class: "panel-body mult" });
  viz.append(host);

  function draw() {
    const [rr, ri] = at(S.b, S.e, S.ix, S.iy);
    const mag = Math.hypot(rr, ri), ph = Math.atan2(ri, rr);
    const kx = S.ix - M0 / 2, ky = S.iy;
    let lo = Infinity, hi = 0, mx = 0;
    for (let ix = 0; ix < M0; ix++) for (let iy = 0; iy < M1; iy++) {
      const v = Math.hypot(...at(S.b, S.e, ix, iy));
      lo = Math.min(lo, v); hi = Math.max(hi, v); mx = Math.max(mx, v);
    }

    // chooser: layer, channels
    const seg = el("div", { class: "seg", role: "group", "aria-label": "Layer" },
      ...[1, 2, 3, 4].map((l) => {
        const b = el("button", { "aria-pressed": String(l === S.l) }, `Layer ${l}`);
        b.addEventListener("click", () => { S.l = l; viz.replaceChildren(); multiplyStage(viz, app); });
        return b;
      }));
    const sel = (label, key) => {
      const s = el("select", { "aria-label": label }, ...Array.from({ length: 32 }, (_, i) => el("option", { value: String(i), ...(i === S[key] ? { selected: "" } : {}) }, String(i))));
      s.addEventListener("change", () => { S[key] = +s.value; draw(); });
      return el("label", { class: "mult-sel" }, el("span", {}, label), s);
    };

    // |R| for every wave of this channel pair, click to choose
    const CELL = 26;
    const heat = el("canvas", { width: M1 * CELL, height: M0 * CELL, class: "mult-heat", "aria-label": "Size of R for every wave" });
    const g = heat.getContext("2d");
    const acc = css("--accent");
    g.fillStyle = css("--panel") || "#fff"; g.fillRect(0, 0, heat.width, heat.height);
    for (let ix = 0; ix < M0; ix++) for (let iy = 0; iy < M1; iy++) {
      const t = Math.hypot(...at(S.b, S.e, ix, iy)) / (mx || 1);
      g.globalAlpha = 0.08 + 0.92 * t; g.fillStyle = acc;
      g.fillRect(iy * CELL + 1, ix * CELL + 1, CELL - 2, CELL - 2);
    }
    g.globalAlpha = 1; g.lineWidth = 3; g.strokeStyle = css("--ink");
    g.strokeRect(S.iy * CELL + 1.5, S.ix * CELL + 1.5, CELL - 3, CELL - 3);
    heat.addEventListener("click", (ev) => {
      const r = heat.getBoundingClientRect();
      const iy = Math.floor(((ev.clientX - r.left) / r.width) * M1), ix = Math.floor(((ev.clientY - r.top) / r.height) * M0);
      if (ix >= 0 && iy >= 0 && ix < M0 && iy < M1) { S.ix = ix; S.iy = iy; draw(); }
    });

    // input wave, output wave, same colour scale
    const T = 64, vmax = Math.max(1, mag);
    const inCv = el("canvas"), outCv = el("canvas");
    paintField(inCv, wave(T, kx, ky, 1, 0), T, { vmax });
    paintField(outCv, wave(T, kx, ky, mag, ph), T, { vmax });

    // one slice through both, so the sliding is visible
    const cyc = Math.max(Math.hypot(kx, ky), 1), Wd = 420, Ht = 130, pad = 8;
    const path = (a, p) => Array.from({ length: 241 }, (_, i) => {
      const t = i / 240;
      return `${i ? "L" : "M"}${(pad + t * (Wd - 2 * pad)).toFixed(1)},${(Ht / 2 - (a * Math.cos(TAU * cyc * t + p) / vmax) * (Ht / 2 - pad)).toFixed(1)}`;
    }).join("");
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${Wd} ${Ht}`); svg.setAttribute("class", "mult-slice");
    svg.innerHTML = `<line x1="${pad}" x2="${Wd - pad}" y1="${Ht / 2}" y2="${Ht / 2}" stroke="${css("--rule-strong")}" stroke-dasharray="3 4"/>`
      + `<path d="${path(1, 0)}" fill="none" stroke="${css("--ink-3")}" stroke-width="1.6"/>`
      + `<path d="${path(mag, ph)}" fill="none" stroke="${acc}" stroke-width="2.6"/>`;

    const frac = ph / TAU;
    host.replaceChildren(
      el("h4", { class: "drill-sub" }, "One learned number, applied to one wave"),
      el("div", { class: "mult-pick" }, seg, sel("channel in", "b"), sel("channel out", "e")),
      el("div", { class: "mult-top" },
        el("div", { class: "mult-heatcol" }, heat,
          el("p", { class: "cap-s" }, `One number R for each of ${M0 * M1} stored waves. Darker means the layer scales that wave up more. Click a square to choose a wave.`)),
        el("div", { class: "mult-right" },
          el("div", { class: "mult-tiles" },
            el("figure", { class: "ftile" }, el("div", { class: "tile" }, inCv), el("figcaption", {}, el("span", { class: "cap-t" }, "a wave goes in"), el("span", { class: "cap-s mono" }, "height 1, no shift"))),
            el("span", { class: "wave-intro-eq", "aria-hidden": "true" }, "×R →"),
            el("figure", { class: "ftile" }, el("div", { class: "tile" }, outCv), el("figcaption", {}, el("span", { class: "cap-t" }, "and comes out"), el("span", { class: "cap-s mono" }, `height ${fmt(mag, 2)}, shifted ${deg(ph)}`)))),
          svg,
          el("p", { class: "cap-s" }, `Grey: the wave going in. Orange: what comes out. The size of R, ${fmt(mag, 2)}, scales it. The angle of R, ${deg(ph)}, slides its peaks ${fmt(Math.abs(frac), 2)} of a wavelength ${frac > 0 ? "earlier" : "later"}.`))),
      el("p", { class: "cap-s" }, `This is wave (${kx}, ${ky}) in layer ${S.l}, channel ${S.b} → channel ${S.e}. Across all ${M0 * M1} waves of this one pair, R ranges in size from ${fmt(lo, 2)} to ${fmt(hi, 2)}: each wave gets its own number. In the real layer every one of the ${cin} input channels has such a number for each of the ${cout} output channels, and the results are added, so one wave of one output channel is ${cin} of these multiplications summed.`));
  }
  draw();
}

// ---------------------------------------------------------------- limits
/** What the Fourier part of the model quietly assumes, and what it throws away, with the numbers. */
export function limitsStage(viz, app) {
  const ns = app.ds.ns.manifest, dc = app.ds.darcy.manifest;
  const nsIn = app.sampleOf("ns").io[64].input, dcIn = app.sampleOf("darcy").io[64].input;

  // 2 x 2 tilings: what the FFT believes each input looks like beyond its edges.
  const tiling = (input, paint) => {
    const res = 64, T = 2 * res, tiled = new Float64Array(T * T);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) tiled[i * T + j] = input[(i % res) * res + (j % res)];
    const cv = el("canvas"); paint(cv, tiled, T); return cv;
  };
  const nsTile = tiling(nsIn, (cv, f, T) => paintField(cv, f, T, { vmax: maxAbs(f) }));
  const dcTile = tiling(dcIn, (cv, f, T) => paintBinary(cv, f, T));

  const keptRange = (m) => { const ex = m.stats.truncation_kept_excl_mean_by_layer; return [Math.min(...ex), Math.max(...ex)]; };
  const a = ns.analysis;

  // card 2: how much of the input the cut keeps, as a bar
  const keptRow = (name, m) => {
    const [lo, hi] = keptRange(m), mid = (lo + hi) / 2;
    return el("div", { class: "lim-bar" },
      el("span", { class: "lim-name" }, name),
      el("span", { class: "lim-track", role: "img", "aria-label": `${name}: spectral path keeps ${pct(lo, 1)} to ${pct(hi, 1)}` },
        el("span", { class: "lim-kept", style: `width:${mid * 100}%` }),
        el("span", { class: "lim-cut", style: `width:${(1 - mid) * 100}%` })),
      el("span", { class: "lim-val mono" }, lo > 0.995 ? pct(mid, 1) : `${pct(lo, 0)}–${pct(hi, 0)}`));
  };
  // card 3: test error at each grid size
  const allErr = [ns, dc].flatMap((m) => m.resolutions.map((r) => m.stats.test_error[String(r)].mean));
  const eMax = Math.max(...allErr);
  const errRow = (name, m) => el("div", { class: "lim-errs" },
    el("span", { class: "lim-name" }, name),
    ...m.resolutions.map((r) => {
      const e = m.stats.test_error[String(r)].mean;
      return el("span", { class: `lim-col${r === m.model.trained_resolution ? " trained" : ""}` },
        el("span", { class: "lim-colbar", style: `height:${Math.max(6, (e / eMax) * 56)}px` }),
        el("span", { class: "mono" }, pct(e, 1)), el("span", { class: "lim-res" }, `${r}²`));
    }));

  const card = (n, title, visual, text) => el("section", { class: "lim-card" },
    el("div", { class: "lim-vis" }, visual),
    el("div", { class: "lim-text" }, el("h5", {}, el("span", { class: "lim-n" }, n), title), el("p", {}, text)));

  viz.append(el("div", { class: "panel-body limits" },
    el("h4", { class: "drill-sub" }, "Three places this model is not as general as it sounds"),
    el("div", { class: "lim-grid" },
      card("1", "The grid wraps around",
        el("div", { class: "lim-pair" }, el("div", { class: "tile lim-tile" }, nsTile), el("div", { class: "tile lim-tile" }, dcTile)),
        "Each input as the Fourier transform sees it: the same map, repeated. The fluid (left) has no seam, because it was simulated on a domain whose edges join. The rock (right) jumps where copies meet, and a jump needs waves of every speed."),
      card("2", "The cut throws detail away",
        el("div", { class: "lim-bars" }, keptRow("fluid", ns), keptRow("rock", dc),
          el("div", { class: "lim-key" }, el("span", {}, el("i", { class: "k" }), "kept by the spectral path"), el("span", {}, el("i", { class: "c" }), "only the local path sees it"))),
        `Share of each layer's input energy that survives keeping ${ns.stats.n_kept_modes} of ${ns.stats.n_total_modes.toLocaleString("en-US")} waves (average excluded; range over layers). The fluid loses almost nothing, the rock loses about 7%.`),
      card("3", "Other grid sizes, measured",
        el("div", { class: "lim-chart" }, errRow("fluid", ns), errRow("rock", dc)),
        `Test error at ${ns.resolutions.map((r) => `${r}×${r}`).join(", ")}, on ${ns.stats.test_error["64"].n} unseen samples each. Both models were trained at 64×64 only. The errors at the other sizes are measured on this test data, so data that looks different could do worse.`))));
}
