// Zoom level 3: focused explainer panels for nodes inside a layer.
import { el, M, paintField, paintBinary, paintInput, isNS, words, maxAbs, fmt, pct, lut } from "../render.js";
import { truncate, keptMask, splitAtCutoff, fft2, ifft2 } from "../fourier.js";
import { mountTruncationWidget } from "../widgets/truncation.js";

const maskCache = new Map();
function mask(res) {
  if (!maskCache.has(res)) maskCache.set(res, keptMask(res));
  return maskCache.get(res);
}

function tile(field, res, vmax, label, sub, cls = "") {
  const cv = el("canvas");
  paintField(cv, field, res, { vmax });
  return el("figure", { class: `ftile ${cls}` }, el("div", { class: "tile" }, cv),
    el("figcaption", {}, el("span", { class: "cap-t" }, label), sub ? el("span", { class: "cap-s" }, sub) : null));
}

const DECADES = 6;
function blues() { return lut("Blues"); }

function close(app) {
  const b = el("button", { class: "drill-close", "aria-label": "Close panel" }, "Close ×");
  b.addEventListener("click", () => app.set({ node: null }));
  return b;
}

export function renderDrill(app, node, ctx) {
  const panel = { fft: fftPanel, truncate: truncatePanel, r: rPanel, conv: convPanel }[node];
  const body = panel ? panel(app, ctx) : [el("p", {}, "No panel for this step.")];
  return el("section", { class: `drill drill-${node}`, "aria-label": `Explanation: ${node}` }, close(app), ...body);
}

// ---------- FFT ----------
// Optional depth: none of this is needed to follow "the spectral path cannot draw an edge".
function fftPanel() {
  const host = el("div", { class: "widget-host", "data-slot": "fft-phase-widget" });
  mountPhaseWidget(host);
  return [
    el("h3", { class: "drill-title" }, "FFT: the parts the argument doesn't need"),
    el("p", { class: "aside-flag" }, "Optional: you can skip this and still follow the rest."),
    el("h4", { class: "drill-sub" }, "Why a wave needs two numbers"),
    el("p", {}, "A wave of a given speed can sit anywhere: peak in the middle, peak at the edge, anything between. Speed alone doesn't pin it down. So each frequency gets two coefficients — how much cosine and how much sine — and their combination slides the wave along."),
    host,
    el("h4", { class: "drill-sub" }, "Complex packaging"),
    el("p", {}, "Those two numbers are stored as one complex number ", M(String.raw`a + bi`), ". Its magnitude ", M(String.raw`\sqrt{a^2 + b^2}`), " is how strong the wave is, and its angle ", M(String.raw`\operatorname{atan2}(b, a)`), " is where the peaks sit. That's the only reason complex numbers show up here: they're a compact way to store a wave's size and shift together."),
    el("p", {}, "It's also why ", M(String.raw`R`), ", the learned weight, is complex: multiplying by a complex number rescales a wave ", el("em", {}, "and"), " shifts it, in one operation."),
    el("h4", { class: "drill-sub" }, "Why it's called fast"),
    el("p", {}, "Measuring each of ", M(String.raw`n`), " frequencies against ", M(String.raw`n`), " samples is ", M(String.raw`n^2`), " work. The FFT uses the fact that the even-indexed and odd-indexed samples give overlapping sub-problems. It splits the transform in two and repeats, which brings the work down to ", M(String.raw`n \log n`), "."),
    el("p", { class: "drill-note" }, "The FFT only makes this faster. A slow transform and a fast one give identical coefficients, so truncation would blur edges either way."),
    el("h4", { class: "drill-sub" }, "The convolution theorem"),
    el("p", {}, "Sliding a filter across an image — a convolution — is the same thing as multiplying coefficients one by one in frequency space. That's why the spectral path is worth the two transforms. With one multiply per kept wave, you get a filter as wide as the whole grid, so the layer can move information across the whole domain in a single step."),
  ];
}

/** Two coefficients, one wave: cos and sin combine into a wave of any shift. */
function mountPhaseWidget(host) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 600 150");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "A cosine and a sine adding up to a shifted wave");
  const slider = el("input", { type: "range", id: "phase-shift", min: "0", max: "100", step: "1", value: "28", "aria-label": "Shift" });
  const lab = el("div", { class: "tw-slider" }, el("span", {}, "shift the peak"), el("b", { class: "mono" }, ""));
  const read = el("p", { class: "cap-s" }, "");
  host.replaceChildren(el("div", { class: "tw" }, svg, lab, slider, read));

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
  const K = 3, N = 400;

  function draw() {
    const phi = (+slider.value / 100) * 2 * Math.PI;
    const a = Math.cos(phi), b = Math.sin(phi);
    const ink3 = css("--ink-3"), rule = css("--rule"), sp = css("--spectral"), lo = css("--local");
    svg.replaceChildren();
    const x0 = 24, x1 = 580, yM = 76, amp = 52;
    svg.append(mk("line", { x1: x0, x2: x1, y1: yM, y2: yM, stroke: rule, "stroke-width": 1 }));
    const line = (fn, color, w, dash) => {
      const pts = Array.from({ length: N }, (_, i) => {
        const t = i / (N - 1);
        return `${(x0 + t * (x1 - x0)).toFixed(1)},${(yM - fn(t) * amp).toFixed(1)}`;
      }).join(" ");
      const p = mk("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": w, "stroke-linejoin": "round" });
      if (dash) p.setAttribute("stroke-dasharray", dash);
      svg.append(p);
    };
    line((t) => a * Math.cos(2 * Math.PI * K * t), ink3, 1.2, "4,4");
    line((t) => -b * Math.sin(2 * Math.PI * K * t), lo, 1.2, "4,4");
    line((t) => a * Math.cos(2 * Math.PI * K * t) - b * Math.sin(2 * Math.PI * K * t), sp, 2.2);
    const t1 = mk("text", { x: x0, y: 140, "font-size": 10, fill: ink3 }); t1.textContent = `${fmt(a, 2)} × cosine`; svg.append(t1);
    const t2 = mk("text", { x: 150, y: 140, "font-size": 10, fill: lo }); t2.textContent = `+ ${fmt(-b, 2)} × sine`; svg.append(t2);
    const t3 = mk("text", { x: 290, y: 140, "font-size": 10, fill: sp, "font-weight": 600 }); t3.textContent = "= one wave, same speed, shifted"; svg.append(t3);
    lab.lastChild.textContent = `${Math.round((phi * 180) / Math.PI)}°`;
    read.textContent = `Same frequency throughout: ${K} cycles across the slice. Only the mix of the two coefficients changes, and the wave slides. Stored as one complex number, this is magnitude 1.00 at angle ${Math.round((phi * 180) / Math.PI)}°.`;
  }
  slider.addEventListener("input", draw);
  draw();
}

// ---------- truncation ----------
function truncatePanel(app, ctx) {
  return [el("h3", { class: "drill-title" }, "Truncation: why an edge needs fast waves"),
    ...truncateWidgetPanel(), ...truncateRealPanel(app, ctx)];
}

/** The 1-D edge-vs-bump widget on its own. */
export function truncateWidgetPanel() {
  const widget = el("div", { class: "widget-host" });
  mountTruncationWidget(widget, { cutoff: 8, n: 64 });
  return [widget];
}

/** The kept-mode block drawn at grid scale: which of the res x res waves survive. */
export function keptMaskFigure(res) {
  const cv = el("canvas", { width: res, height: res, class: "maskfig", "aria-label": "Kept modes" });
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(res, res);
  const m = mask(res);
  const css = getComputedStyle(document.documentElement);
  const hex = (v) => { const h = css.getPropertyValue(v).trim().replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
  const on = hex("--spectral"), off = hex("--rule");
  for (let i = 0; i < res * res; i++) img.data.set([...(m[i] ? on : off), 255], i * 4);
  ctx.putImageData(img, 0, 0);
  const kept = m.reduce((s, v) => s + (v ? 1 : 0), 0);
  return el("figure", { class: "ftile maskwrap" }, el("div", { class: "tile" }, cv),
    el("figcaption", {}, el("span", { class: "cap-t" }, `${kept} kept of ${(res * res).toLocaleString("en-US")}`),
      el("span", { class: "cap-s" }, `every wave in the grid, at ${res}×${res}; centre is the flat average`)));
}

/** The cut applied to the layer's real input: before, after, and what was removed. */
export function truncateRealPanel(app, { l, part = "both" }) {
  const s = app.sample;
  const res = s.live ? s.live.res : 64;
  const stats = app.sample.live?.layerStats?.[l - 1] ?? s.entry.layers[l - 1];
  const inFields = s.live ? (l === 1 ? s.live.lifted : s.live.layers[l - 2].out) : s.fields[l === 1 ? "lifted" : `layer${l - 1}.out`];
  // Show the input channel with the most energy above the cutoff, so the cut is visible.
  const m = mask(res);
  let best = 0, bestHi = -1;
  const splits = inFields.map((f) => splitAtCutoff(f, res, m));
  splits.forEach((sp, c) => { let e = 0; for (const v of sp.high) e += v * v; if (e > bestHi) { bestHi = e; best = c; } });
  const x = inFields[best];
  const kept = truncate(x, res);
  const cut = x.map((v, i) => v - kept[i]);
  const vm = maxAbs(x);

  // The removed part is tiny on smooth data, so it gets its own scale (and says so); before and after share one.
  const vmRem = maxAbs(cut) || 1;
  const out = [
    el("h4", { class: "drill-sub" }, "3 · Keeping only the slow block"),
    el("p", { class: "lead-in" }, `Now the cut itself. We keep only the waves inside the orange block and drop everything else. Here's what that does to one of layer ${l}'s input channels, the one that loses the most.`),
    el("div", { class: "drill-row" },
      tile(x, res, vm, `input channel ${best}`, "before the cut"),
      el("span", { class: "op-chip" }, "keep |k| ≤ 8"),
      tile(kept, res, vm, "after the cut", `same scale ±${fmt(vm, 2)}`),
      el("span", { class: "op-chip" }, "difference"),
      tile(cut, res, vmRem, "what was removed", `own scale ±${fmt(vmRem, 3)}`)),
    el("p", { class: "drill-note" },
      `This channel is smooth, so the cut barely changes it. The last tile has its own, much finer colour scale (±${fmt(vmRem, 3)}, against ±${fmt(vm, 2)} for the other two), just so the difference is visible at all. Across all 32 channels, the cut keeps ${pct(stats.truncation_kept_excl_mean, 1)} of this input's energy, not counting the flat average.`),
  ];

  if (part === "real") return out;
  if (part === "rock") out.length = 0;
  // A smooth fluid has few fast waves to lose, so show the same cut on an image that does: a rock map with sharp edges.
  try {
    const rock = app.sampleOf("darcy").io[64].input;
    const rk = truncate(rock, 64), rcut = Float64Array.from(rock, (v, i) => v - rk[i]);
    const rv = maxAbs(rock) || 1, rvr = maxAbs(rcut) || 1;
    const inCv = el("canvas"), keptCv = el("canvas"), cutCv = el("canvas");
    paintBinary(inCv, rock, 64);
    paintField(keptCv, rk, 64, { vmax: rv });
    paintField(cutCv, rcut, 64, { vmax: rvr });
    const fig = (cv, label, sub) => el("figure", { class: "ftile" }, el("div", { class: "tile" }, cv),
      el("figcaption", {}, el("span", { class: "cap-t" }, label), sub ? el("span", { class: "cap-s" }, sub) : null));
    out.push(
      el("h4", { class: "drill-sub" }, "5 · The same cut on sharp edges"),
      el("p", { class: "lead-in" }, "The cut costs so little here because the fluid is smooth. Here's the same cut on an image with sharp edges, a map of two kinds of rock."),
      el("div", { class: "drill-row" },
        fig(inCv, "a rock map", "two materials, sharp edges"),
        el("span", { class: "op-chip" }, "keep |k| ≤ 8"),
        fig(keptCv, "after the cut", `same scale ±${fmt(rv, 2)}`),
        el("span", { class: "op-chip" }, "difference"),
        fig(cutCv, "what was removed", `own scale ±${fmt(rvr, 2)}`)),
      el("p", { class: "drill-note" }, "This time the cut does real damage. The edges blur and ripple, and everything it removed sits right along them. Sharp edges are made of fast waves, and those are exactly what the cut throws away."));
  } catch (e) { /* the rock data is only needed for this comparison */ }
  return out;
}

// ---------- step "waves": grounding wave + energy before they get used unexplained ----------
const ONE_WAVE = { kx: 11, ky: 11 };

function variance(f) {
  let m = 0;
  for (let i = 0; i < f.length; i++) m += f[i];
  m /= f.length;
  let s = 0;
  for (let i = 0; i < f.length; i++) s += (f[i] - m) ** 2;
  return s / f.length;
}

/** Grounds "wave" and "energy" against the actual rock image for this sample, before the full
 *  spectrum panel needs them. Two real, live-computed pieces: what one wave looks like, and an
 *  honest low/high energy split of this exact input — the same split truncation performs a few
 *  steps later, so this doubles as a preview. Tracks the resolution control: not a fixed demo. */
export function waveIntroPanel(app) {
  const res = app.state.res;
  const input = app.sample.io[res].input;
  const rockCv = el("canvas");
  paintInput(rockCv, input, res, { vmax: maxAbs(input) });

  // One wave, drawn plainly: no energy claim attached, this tile is just vocabulary.
  const w = new Float64Array(res * res);
  for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) w[i * res + j] = Math.cos((2 * Math.PI * (ONE_WAVE.kx * i + ONE_WAVE.ky * j)) / res);
  const waveCv = el("canvas");
  paintField(waveCv, w, res, { vmax: 1 });
  const kmag = Math.hypot(ONE_WAVE.kx, ONE_WAVE.ky);

  // The real split: reconstruct from only the low (kept) modes, and look at what's left over.
  // low + high === input exactly, and because the two occupy disjoint frequency bands, their
  // energy shares add to the whole — this is the same accounting the model itself performs.
  const low = truncate(input, res);
  const high = Float64Array.from(input, (v, i) => v - low[i]);
  const varIn = variance(input);
  const shareHigh = variance(high) / varIn;
  const shareLow = 1 - shareHigh;
  const vm = maxAbs(input);

  // A progressive reconstruction: add waves of increasing speed, one group at a time, and
  // watch it converge on the real rock map. The first four steps use a simple circular cutoff,
  // purely to illustrate "more waves kept, more detail kept" — the network's own cutoff (next
  // step, "low") uses a differently shaped block, not a circle, which the caption says plainly.
  const steps = [
    { r: 0, label: "just the average" },
    { r: 1, label: "+ 1 more wave" },
    { r: 3, label: "+ a few more" },
    { r: 6, label: "+ more still" },
  ].map(({ r, label }) => ({ label, field: radialLowpass(input, res, r) }));
  steps.push({ label: "the network's cutoff", field: low, highlight: true });
  steps.push({ label: "every wave", field: input, exact: true });

  const buildRow = el("div", { class: "wave-intro-row build" },
    ...steps.map(({ label, field, highlight, exact }) =>
      el("figure", { class: `ftile wave-intro-tile${highlight ? " on" : ""}` },
        el("div", { class: "tile" }, exact && !isNS() ? binaryCv(field, res) : tileCv(field, res, vm)),
        el("figcaption", {}, el("span", { class: "cap-t" }, label), exact ? el("span", { class: "cap-s" }, `= this ${words().input}, exactly`) : null))));

  const splitRow = el("div", { class: "wave-intro-row" },
    el("figure", { class: "ftile wave-intro-tile" }, el("div", { class: "tile" }, tileCv(low, res, vm)),
      el("figcaption", {}, el("span", { class: "cap-t" }, "kept, at the network's cutoff"), el("span", { class: "cap-s" }, el("strong", {}, pct(shareLow, 1)), " of the energy"))),
    el("span", { class: "wave-intro-eq", "aria-hidden": "true" }, "+"),
    el("figure", { class: "ftile wave-intro-tile" }, el("div", { class: "tile" }, tileCv(high, res, vm)),
      el("figcaption", {}, el("span", { class: "cap-t" }, "left out"), el("span", { class: "cap-s" }, el("strong", {}, pct(shareHigh, 1)), " of the energy"))),
    el("span", { class: "wave-intro-eq", "aria-hidden": "true" }, "="),
    el("figure", { class: "ftile wave-intro-tile" }, el("div", { class: "tile" }, rockCv),
      el("figcaption", {}, el("span", { class: "cap-t" }, `this ${words().input}`), el("span", { class: "cap-s" }, `${res}×${res}, this sample`))));

  return [
    el("figure", { class: "ftile wave-intro-tile one-wave" }, el("div", { class: "tile" }, waveCv),
      el("figcaption", {}, el("span", { class: "cap-t" }, "one wave, by itself"),
        el("span", { class: "cap-s mono" }, `${fmt(kmag, 1)} cycles across the grid, ${fmt(res / kmag, 1)} px wavelength`))),
    el("p", { class: "cap-s" }, "Every image on this page — this one included — is a sum of waves like that one, at every speed and angle the grid can hold. Adding more of them, in order from slow to fast, is what reconstructs an image."),
    buildRow,
    el("p", { class: "cap-s" }, "Each tile adds more, faster waves to the last one. By the last tile every wave the grid can hold is included, and the reconstruction matches the original exactly (up to rounding)."),
    splitRow,
    el("p", { class: "cap-s" }, `Group those same waves into two sets at the network's own cutoff instead of building up gradually: ${pct(shareLow, 1)} of the energy in the kept set, ${pct(shareHigh, 1)} in the set left out, for this sample at ${res}×${res}.`),
  ];
}

/** A simple circular low-pass, for illustrating convergence only — not the model's own
 *  (differently shaped) kept-mode block, which `truncate()` computes elsewhere in this file. */
function radialLowpass(field, res, radius) {
  const F = fft2(field, res);
  const N = res * res;
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let x = 0; x < res; x++) {
    const kx = x <= res / 2 ? x : x - res;
    for (let y = 0; y < res; y++) {
      const ky = y <= res / 2 ? y : y - res;
      if (kx * kx + ky * ky <= radius * radius) {
        const i = x * res + y;
        re[i] = F.re[i]; im[i] = F.im[i];
      }
    }
  }
  return ifft2({ re, im }, res).re;
}

function tileCv(field, res, vmax) {
  const cv = el("canvas");
  paintField(cv, field, res, { vmax });
  return cv;
}

/** The reconstructed field is real-valued and only near-binary (FFT round-trip), so round it
 *  before handing it to the binary rock-map painter — this tile is meant to look identical to
 *  the reference rock-map tile, not merely close. */
function binaryCv(field, res) {
  const cv = el("canvas");
  paintBinary(cv, field, res);
  return cv;
}

// ---------- x R ----------
export function rPanel(app, { l, truncOnly = false }) {
  const s = app.sample;
  const m = app.manifest;
  const spectra = s.live?.spectra ?? s.spectra;
  const res = s.live ? s.live.res : 64;
  const kinds = truncOnly ? [["0", "layer input"], ["1", "after truncation"], ["3", "what we throw away"]]
    : [["0", "layer input"], ["1", "after truncation"], ["2", "after × R and back"]];
  const key = truncOnly ? "tKind" : "rKind";
  let kind = app.state[key] ?? "0";
  if (!kinds.some(([k]) => k === kind)) kind = "0";

  const heat = el("canvas", { class: "heat", width: res, height: res, "aria-label": "Mode energy heatmap" });
  const overlay = el("canvas", { class: "heat-overlay", width: res * 8, height: res * 8, "aria-hidden": "true" });
  const wave = el("canvas", { class: "wave", width: 64, height: 64, "aria-label": "Plane wave for the hovered mode" });
  const info = el("div", { class: "mode-info mono" }, "Hover a cell to see its wave.");
  const seg = el("div", { class: "seg", role: "group", "aria-label": "Spectrum stage" });
  const kmask = mask(res);

  function draw() {
    // "what we throw away": the same spectrum with the kept block greyed and the rest stretched to its own range
    const lost = kind === "3";
    const E = spectra[l - 1][lost ? 0 : +kind];
    let hi = -Infinity;
    for (let i = 0; i < E.length; i++) if (E[i] > -29 && !(lost && kmask[i])) hi = Math.max(hi, E[i]);
    const lo = hi - (truncOnly ? 10 : DECADES);
    const ctx = heat.getContext("2d");
    const img = ctx.createImageData(res, res);
    const lut = blues();
    for (let i = 0; i < res * res; i++) {
      if (E[i] <= -29 || (lost && kmask[i])) { img.data.set([206, 209, 214, 255], i * 4); continue; }
      const t = Math.max(0, Math.min(1, (E[i] - lo) / (hi - lo)));
      const j = Math.round(t * 255) * 3;
      img.data.set([lut[j], lut[j + 1], lut[j + 2], 255], i * 4);
    }
    ctx.putImageData(img, 0, 0);
    const o = overlay.getContext("2d");
    o.clearRect(0, 0, overlay.width, overlay.height);
    o.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--spectral").trim();
    o.lineWidth = 2;
    const sc = overlay.width / res;
    for (let x = 0; x < res; x++) for (let y = 0; y < res; y++) {
      if (!kmask[x * res + y]) continue;
      if (x === 0 || !kmask[(x - 1) * res + y]) seg4(o, y * sc, x * sc, (y + 1) * sc, x * sc);
      if (x === res - 1 || !kmask[(x + 1) * res + y]) seg4(o, y * sc, (x + 1) * sc, (y + 1) * sc, (x + 1) * sc);
      if (y === 0 || !kmask[x * res + y - 1]) seg4(o, y * sc, x * sc, y * sc, (x + 1) * sc);
      if (y === res - 1 || !kmask[x * res + y + 1]) seg4(o, (y + 1) * sc, x * sc, (y + 1) * sc, (x + 1) * sc);
    }
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.k === kind)));
  }
  function seg4(o, x0, y0, x1, y1) { o.beginPath(); o.moveTo(x0, y0); o.lineTo(x1, y1); o.stroke(); }

  kinds.forEach(([k, label]) => {
    const b = el("button", { "data-k": k, "aria-pressed": String(k === kind) }, label);
    b.addEventListener("click", () => { kind = k; app.state[key] = k; draw(); });
    seg.append(b);
  });

  const stack = el("div", { class: "heat-stack" }, heat, overlay);
  function onMove(e) {
    const r = heat.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * res);
    const row = Math.floor(((e.clientY - r.top) / r.height) * res);
    if (col < 0 || row < 0 || col >= res || row >= res) return;
    showMode(row - res / 2, col - res / 2);
  }
  stack.addEventListener("pointermove", onMove);
  draw();
  // Rest state: show one kept wave before any hover.
  showMode(3, 2);
  function showMode(kx, ky) {
    const w = new Float64Array(64 * 64);
    for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) w[i * 64 + j] = Math.cos(2 * Math.PI * (kx * i + ky * j) / 64);
    paintField(wave, w, 64, { vmax: 1 });
    const row = kx + res / 2, col = ky + res / 2;
    const E = spectra[l - 1][+kind][row * res + col];
    const kmag = Math.hypot(kx, ky);
    info.replaceChildren(
      el("div", {}, `kx ${kx}, ky ${ky}`),
      el("div", {}, kmag === 0 ? "the mean (constant)" : `${fmt(kmag, 1)} cycles across the grid`),
      el("div", {}, kmag === 0 ? "" : `wavelength ${fmt(res / kmag, 1)} px`),
      el("div", {}, E <= -29 ? "discarded at this stage" : `log10 energy ${fmt(E, 2)}`),
      el("div", { class: kmask[row * res + col] ? "t-spectral" : "" }, kmask[row * res + col] ? "kept by truncation" : "outside the kept block"));
  }

  const W = m.weights.tensors[`L${l - 1}.spec.w`].shape;
  // The strongest waves outside the block, drawn as the waves they are, next to the strongest one inside it.
  const lostRow = () => {
    const E0 = spectra[l - 1][0], h = res / 2;
    let total = 0;
    for (let i = 0; i < E0.length; i++) if (E0[i] > -29) total += 10 ** E0[i];
    const pick = (inside, n) => {
      const idx = [];
      for (let i = 0; i < E0.length; i++) if (E0[i] > -29 && !!kmask[i] === inside && i !== h * res + h) idx.push(i);
      idx.sort((a, b) => E0[b] - E0[a]);
      const out = [], seen = new Set();
      for (const i of idx) {
        if (seen.has(i)) continue;
        const r = Math.floor(i / res), c = i % res;
        seen.add(((res - r) % res) * res + ((res - c) % res)); seen.add(i);
        out.push(i); if (out.length === n) break;
      }
      return out;
    };
    const fig = (i, tag) => {
      const r = Math.floor(i / res), c = i % res, kx = r - h, ky = c - h;
      const w = new Float64Array(res * res);
      for (let a = 0; a < res; a++) for (let b = 0; b < res; b++) w[a * res + b] = Math.cos((2 * Math.PI * (kx * a + ky * b)) / res);
      const share = (2 * 10 ** E0[i]) / total;
      const txt = share >= 1e-4 ? pct(share, 2) : `${(share * 100).toExponential(1)}%`;
      return tile(w, res, 1, `${tag} kx ${kx}, ky ${ky}`, `${fmt(Math.hypot(kx, ky), 1)} cycles · ${txt} of the energy`);
    };
    const kept = pick(true, 1), cut = pick(false, 5);
    return el("div", { class: "lost-waves" },
      el("h4", { class: "drill-sub" }, "4 · What the cut throws away"),
      el("p", { class: "lead-in" }, "So what exactly did we lose? Here are the biggest waves outside the block, drawn out, next to the biggest wave we keep."),
      el("div", { class: "drill-row" }, ...(kept.length ? [fig(kept[0], "kept:"), el("span", { class: "op-chip" }, "vs")] : []), ...cut.map((i) => fig(i, "cut:"))),
      el("p", { class: "drill-note" }, "Even the biggest wave we throw away holds a tiny sliver of the energy, compared with the biggest wave we keep. In smooth data like this fluid, there just isn't much up here to lose."));
  };

  if (truncOnly) return [
    el("h3", { class: "drill-title" }, "1 · The input, broken into waves"),
    el("p", { class: "lead-in" }, `First, the FFT. This is layer ${l}'s input, broken into its waves. Every square on the map below is one wave, and together they make up the whole input.`),
    el("div", { class: "drill-grid" },
      el("div", { class: "heat-col" }, seg, stack,
        el("p", { class: "cap-s" }, `Brightness is on a log scale and summed over all 32 channels, so even very faint waves show up. Outlined: the ${m.stats.n_kept_modes} waves we keep, out of ${(res * res).toLocaleString("en-US")}.`)),
      el("div", { class: "wave-col" }, el("div", { class: "cap-t" }, "The square you're pointing at, as a wave"), wave, info),
      el("div", { class: "drill-text" },
        el("p", {}, "The middle square is the flat average. The further a square sits from the middle, the faster its wave wiggles, and which way it sits tells you which way the stripes run. The brighter a square, the more of that wave is in the input. Point at any square to see its wave drawn out."),
        el("p", {}, `The orange outline is the block of slow waves we keep: the same ${m.stats.n_kept_modes} waves for every image. Click “after truncation” to see only what's inside it, or “what we throw away” to see only what's outside it, brightened so you can see it.`))),
    lostRow(),
  ];
  return [
    el("h3", { class: "drill-title" }, "× R: one learned mixing matrix per kept wave"),
    el("div", { class: "drill-grid" },
      el("div", { class: "heat-col" }, seg, stack,
        el("p", { class: "cap-s" }, `log10 energy summed over 32 channels · 6 decades · (0,0) at center · outlined: ${m.stats.n_kept_modes} kept modes of ${(res * res).toLocaleString("en-US")}`)),
      el("div", { class: "wave-col" }, el("div", { class: "cap-t" }, "Hovered mode as a wave"), wave, info),
      el("div", { class: "drill-text" },
        el("p", {}, "Each cell is one wave. Its position says how fast it oscillates and in which direction: the center is the mean, the far edges are the finest ripples the grid can show."),
        el("p", {}, `R is a complex array of shape ${W.join(" × ")}. For each kept wave it holds a ${W[0]}×${W[1]} matrix that mixes the ${W[0]} input channels into ${W[1]} output channels. Complex numbers let it rescale each wave and shift where its peaks sit.`),
        el("p", {}, "Switch to “after truncation”: everything outside the outline turns grey. Switch to “after × R”: the energy is redistributed inside the block but never leaves it. Multiplying a wave's coefficient by a number can't create a different wave."),
        el("p", { class: "drill-note" }, "The strong horizontal and vertical lines through the center come from the square's borders. The FFT treats the grid as if it wrapped around, and opposite edges of the rock don't match. The model's own FFT sees these lines too."))),
  ];
}

// ---------- 1x1 conv ----------
export function convPanel(app, { l, c, F }) {
  const res = F.res;
  const stats = app.sample.live?.layerStats?.[l - 1] ?? app.sample.entry.layers[l - 1];
  const m = mask(res);
  const sp = splitAtCutoff(F.spectral, res, m);
  const lc = splitAtCutoff(F.local, res, m);
  const vm = Math.max(maxAbs(sp.high), maxAbs(lc.high)) || 1;
  const exact = !!app.sample.live;
  return [
    el("h3", { class: "drill-title" }, "1×1 conv: the path that keeps the edges"),
    el("p", {}, "A 1×1 convolution looks at one pixel at a time and mixes the 32 channel values found there. It has no notion of frequency, so it can't lose any. Whatever sharpness its input has, it passes on."),
    el("h4", { class: "drill-sub" }, `Only the detail above the cutoff, channel ${c}`),
    el("div", { class: "drill-row" },
      tile(sp.high, res, vm, "Spectral path, above cutoff", "", "p-spectral"),
      tile(lc.high, res, vm, "Local path, above cutoff", `shared scale ±${fmt(vm, 3)}`, "p-local")),
    el("p", { class: "drill-note" },
      `The left tile is blank because the spectral path has nothing there to show. Over all 32 channels of this layer, the local path supplies ${pct(stats.outside_cutoff_share_local, 1)} of the energy above the cutoff. ${pct(stats.local_own_energy_outside_cutoff, 1)} of the local path's own energy lives up there, and ${pct(stats.spectral_own_energy_outside_cutoff, 1)} of the spectral path's.`,
      exact ? "" : " The two tiles above come from the 8-bit precomputed fields, so the left one holds a little rounding noise, far below this colour scale. The percentages come from the full-precision export."),
  ];
}

/** Energy per wave after the × R step, as one tile: grey outside the kept block. For the "back to pixels" slide. */
export function heatTile(app, { l, kind = 2 }) {
  const s = app.sample, res = s.live ? s.live.res : 64;
  const E = (s.live?.spectra ?? s.spectra)[l - 1][kind];
  let hi = -Infinity;
  for (const v of E) if (v > -29) hi = Math.max(hi, v);
  const lo = hi - DECADES, cv = el("canvas", { width: res, height: res, class: "maskfig" });
  const ctx = cv.getContext("2d"), img = ctx.createImageData(res, res), L = blues();
  for (let i = 0; i < res * res; i++) {
    if (E[i] <= -29) { img.data.set([206, 209, 214, 255], i * 4); continue; }
    const t = Math.max(0, Math.min(1, (E[i] - lo) / (hi - lo))), j = Math.round(t * 255) * 3;
    img.data.set([L[j], L[j + 1], L[j + 2], 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  return el("figure", { class: "ftile" }, el("div", { class: "tile" }, cv),
    el("figcaption", {}, el("span", { class: "cap-t" }, "waves after × R"), el("span", { class: "cap-s" }, "all 32 channels; grey = discarded")));
}
