// Field and heatmap painting. Colormap LUTs come from the manifest (sampled from matplotlib),
// so browser tiles use the same RdBu_r / Blues as the matplotlib renders.

import katex from "../vendor/katex/katex.mjs";

let LUT = null;

export function setColormaps(colormaps) {
  LUT = {};
  for (const [name, rows] of Object.entries(colormaps)) {
    const u8 = new Uint8ClampedArray(256 * 3);
    rows.forEach((rgb, i) => u8.set(rgb, i * 3));
    LUT[name] = u8;
  }
}

// Which PDE the active bundle solves. Darcy bundles carry no manifest.problem; Navier-Stokes
// bundles do. The app calls setProblem whenever a step switches datasets.
let PROBLEM = null;
export function setProblem(p) { PROBLEM = p ?? null; }
export const isNS = () => PROBLEM?.id === "navier_stokes";

/** Input field as its physics reads: two rock materials, or a signed forcing pattern. */
export function paintInput(canvas, field, res, { vmax = null } = {}) {
  if (!isNS()) return paintBinary(canvas, field, res);
  return paintField(canvas, field, res, { vmax });
}

/** One shared color scale for output fields, as paintField options: pressure (>= 0) or vorticity (signed). */
export function outScale(...fields) {
  const m = Math.max(...fields.map(maxAbs)) || 1;
  return isNS() ? { mode: "diverging", vmax: m } : { mode: "sequential", vmin: 0, vmax: m };
}

/** Plain-language names for the active problem's fields. */
export function words() {
  return isNS()
    ? { input: "forcing", inputSym: "f", output: "vorticity", outputLong: "vorticity at T = 5", thing: "flow", things: "flows" }
    : { input: "rock map", inputSym: "a", output: "pressure", outputLong: "pressure", thing: "rock", things: "rocks" };
}

export function lut(name) {
  return LUT[name];
}

export function maxAbs(f) {
  let m = 0;
  for (let i = 0; i < f.length; i++) { const v = Math.abs(f[i]); if (v > m) m = v; }
  return m;
}

/**
 * Paint a square field into a canvas at native resolution; CSS scales it (pixelated).
 * mode "diverging": value in [-vmax, vmax] -> RdBu_r, 0 is the colormap midpoint.
 * mode "sequential": value in [vmin, vmax] -> Blues.
 */
export function paintField(canvas, field, res, { mode = "diverging", vmax = null, vmin = 0 } = {}) {
  if (canvas.width !== res) { canvas.width = res; canvas.height = res; }
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(res, res);
  const lut = LUT[mode === "diverging" ? "RdBu_r" : "Blues"];
  const hi = vmax ?? (mode === "diverging" ? maxAbs(field) || 1 : Math.max(...field));
  const lo = mode === "diverging" ? -hi : vmin;
  const span = hi - lo || 1;
  for (let i = 0; i < res * res; i++) {
    const t = Math.min(1, Math.max(0, (field[i] - lo) / span));
    const j = Math.round(t * 255) * 3;
    img.data[i * 4] = lut[j]; img.data[i * 4 + 1] = lut[j + 1]; img.data[i * 4 + 2] = lut[j + 2]; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return hi;
}

/** Binary rock map: two fixed material colors (not a data colormap). */
export function paintBinary(canvas, field, res, colors = [[226, 222, 212], [92, 84, 72]]) {
  if (canvas.width !== res) { canvas.width = res; canvas.height = res; }
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(res, res);
  for (let i = 0; i < res * res; i++) {
    const c = colors[field[i] > 0.5 ? 1 : 0];
    img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function fmt(v, digits = 2) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 10 ** -digits || a >= 1e4)) return v.toExponential(1);
  return v.toFixed(digits);
}

export function pct(v, digits = 1) {
  return `${(v * 100).toFixed(digits)}%`;
}

/** LaTeX, rendered with the bundled KaTeX. M = inline, D = display block. */
export function M(tex) {
  const s = document.createElement("span");
  s.className = "m";
  s.innerHTML = katex.renderToString(tex, { throwOnError: false, output: "html" });
  return s;
}
export function D(tex) {
  const d = document.createElement("div");
  d.className = "m-block";
  d.innerHTML = katex.renderToString(tex, { displayMode: true, throwOnError: false, output: "html" });
  return d;
}

export function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "dataset") Object.assign(n.dataset, v);
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}
