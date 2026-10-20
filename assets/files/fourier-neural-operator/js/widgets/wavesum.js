// "How waves add up": three small experiments that build the picture the spectral path relies on.
//   mix     — five waves, five sliders. The thick curve is their sum. (Also reads out energy.)
//   build   — pick a shape, add waves one at a time until it is rebuilt.
//   surface — the same idea in 2-D: a rotatable height-field rebuilt from stripes, k = 0 .. 16.
// All of it is computed live; nothing here is a recording. State lives at module scope so a
// re-render of the stage (new sample, new grid) does not throw the reader's sliders away.
import { el, paintField, lut } from "../render.js";

const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const TAU = 2 * Math.PI;

const NP = 480;
const xs = Array.from({ length: NP + 1 }, (_, i) => i / NP);
// One cool family, dark to light as the waves get faster, so the warm orange sum is the only thing that pops.
const WAVE_COLORS = ["hsl(226 48% 36%)", "hsl(214 52% 46%)", "hsl(200 55% 50%)", "hsl(186 48% 52%)", "hsl(172 42% 56%)"];

const S = {
  amps: [0, 0.8, 0.3, 0, 0], // amplitude of the k = 1..5 waves: the recipe from the text
  kind: "step", n: 3, playing: null,
  K: 3, az: -0.65, elev: 0.82,
  tab: {},
};

// ---------------------------------------------------------------- 1-D machinery
const sinSum = (amps, x) => {
  let v = 0;
  for (let k = 0; k < amps.length; k++) v += amps[k] * Math.sin(TAU * (k + 1) * x);
  return v;
};

function target1d(kind) {
  // off-centre on purpose: a centred step has no even waves at all, and a skipped wave is a confusing thing to watch
  if (kind === "step") return xs.map((x) => (x >= 0.2 && x < 0.55 ? 1 : 0));
  if (kind === "saw") return xs.map((x) => x);
  return xs.map((x) => Math.exp(-((x - 0.5) ** 2) / (2 * 0.1 * 0.1)));
}

const NK = 24;
function fourier1d(f) {
  const n = NP; // NP samples on [0,1); the last (x = 1) point repeats the first
  const a = [], b = [];
  for (let k = 0; k <= NK; k++) {
    let ar = 0, br = 0;
    for (let i = 0; i < n; i++) {
      const g = (TAU * k * i) / n;
      ar += f[i] * Math.cos(g); br += f[i] * Math.sin(g);
    }
    if (k === 0) { a.push(ar / n); b.push(0); } else { a.push((2 * ar) / n); b.push((2 * br) / n); }
  }
  return { a, b };
}
const build1dCache = {};
function build1d(kind) {
  if (!build1dCache[kind]) {
    const f = target1d(kind);
    const c = fourier1d(f);
    const e = [0];
    for (let k = 1; k <= NK; k++) e.push(c.a[k] ** 2 + c.b[k] ** 2);
    build1dCache[kind] = { f, c, e, total: e.reduce((s, v) => s + v, 0) };
  }
  return build1dCache[kind];
}
const wave1d = (c, k, x) => c.a[k] * Math.cos(TAU * k * x) + c.b[k] * Math.sin(TAU * k * x);

function frame(svg, { x0, x1, yB, yT, ymax, rule, ink3 }) {
  const Y = (v) => (yB + yT) / 2 - (v / ymax) * ((yB - yT) / 2);
  svg.append(svgEl("line", { x1: x0, x2: x1, y1: Y(0), y2: Y(0), stroke: rule, "stroke-width": 1 }));
  for (const v of [-1, 1]) {
    if (Math.abs(v) > ymax) continue;
    svg.append(svgEl("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: rule, "stroke-width": 0.6, "stroke-dasharray": "2,4" }));
    const t = svgEl("text", { x: x0 - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 10, fill: ink3 });
    t.textContent = v > 0 ? "+1" : "−1";
    svg.append(t);
  }
  const z = svgEl("text", { x: x0 - 8, y: Y(0) + 4, "text-anchor": "end", "font-size": 10, fill: ink3 }); z.textContent = "0"; svg.append(z);
  return Y;
}
function curve(svg, vals, X, Y, color, w, dash, op = 1) {
  const pts = vals.map((v, i) => `${X(i / (vals.length - 1)).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  const p = svgEl("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": w, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: op });
  if (dash) p.setAttribute("stroke-dasharray", dash);
  svg.append(p);
}

// ---------------------------------------------------------------- tab 1: mix it yourself
function mixTab({ showEnergy }) {
  const svg = svgEl("svg", { viewBox: "0 0 600 220", role: "img", "aria-label": "Waves and their sum" });
  const sliderRows = [], valEls = [];
  const energyBox = el("div", { class: "ws-energy" });

  S.amps.forEach((a, i) => {
    const k = i + 1;
    const val = el("span", { class: "ws-val mono" });
    const input = el("input", { type: "range", min: "-1", max: "1", step: "0.05", value: String(a), "aria-label": `Amount of the wave that repeats ${k} time${k > 1 ? "s" : ""}` });
    input.addEventListener("input", () => { S.amps[i] = +input.value; draw(); });
    valEls.push({ val, input });
    sliderRows.push(el("label", { class: "ws-row" },
      el("span", { class: "ws-dot", style: `background:${WAVE_COLORS[i]}` }),
      el("span", { class: "ws-lab" }, `repeats ${k}×`),
      input, val));
  });

  const preset = (label, amps) => {
    const b = el("button", {}, label);
    b.addEventListener("click", () => { amps.forEach((v, i) => (S.amps[i] = v)); sync(); draw(); });
    return b;
  };
  const presets = el("div", { class: "seg ws-presets", role: "group", "aria-label": "Recipes" },
    preset("0.8 and 0.3", [0, 0.8, 0.3, 0, 0]),
    preset("a bit of everything", [0.5, 0.5, 0.5, 0.5, 0.5]),
    preset("steps (1, ⅓, ⅕)", [1, 0, 1 / 3, 0, 1 / 5]),
    preset("flat", [0, 0, 0, 0, 0]));

  function sync() { valEls.forEach(({ val, input }, i) => { input.value = String(S.amps[i]); val.textContent = S.amps[i].toFixed(2); }); }

  function draw() {
    const rule = css("--rule"), ink3 = css("--ink-3"), ink = css("--accent");
    const f = xs.map((x) => sinSum(S.amps, x));
    const mx = Math.max(...f.map(Math.abs));
    const ymax = Math.max(1.3, mx * 1.12);
    svg.replaceChildren();
    const x0 = 34, x1 = 592;
    const X = (t) => x0 + t * (x1 - x0);
    const Y = frame(svg, { x0, x1, yB: 208, yT: 12, ymax, rule, ink3 });
    S.amps.forEach((a, i) => {
      if (a === 0) return;
      curve(svg, xs.map((x) => a * Math.sin(TAU * (i + 1) * x)), X, Y, WAVE_COLORS[i], 1.4, null, 0.75);
    });
    curve(svg, f, X, Y, ink, 2.8);
    const lab = svgEl("text", { x: x1, y: 14, "text-anchor": "end", "font-size": 10.5, fill: css("--accent-text") }); lab.textContent = "the sum — the thick curve";
    svg.append(lab);
    valEls.forEach(({ val }, i) => (val.textContent = S.amps[i].toFixed(2)));

    if (showEnergy) {
      const E = S.amps.map((a) => a * a);
      const total = E.reduce((s, v) => s + v, 0);
      // energy measured straight off the drawn curve: 2 x the average of the squared height,
      // so that one wave of height a reads a^2. If energies add, this equals `total`.
      let ms = 0, mean = 0;
      for (let i = 0; i < NP; i++) { ms += f[i] * f[i]; mean += f[i]; }
      ms = (2 * ms) / NP; mean /= NP;
      const slow = E.findIndex((e) => e > 0);
      const share = total > 0 && slow >= 0 ? E[slow] / total : null;
      const maxE = Math.max(...E, 1e-9);
      energyBox.replaceChildren(
        el("div", { class: "ws-bars", role: "img", "aria-label": "Energy of each wave" },
          ...E.map((e, i) => el("div", { class: "ws-bar" },
            el("span", { class: "ws-bar-lab mono" }, `${i + 1}×`),
            el("span", { class: "ws-track" }, el("span", { class: "ws-fill", style: `width:${(e / maxE) * 100}%;background:${WAVE_COLORS[i]}` })),
            el("span", { class: "ws-bar-val mono" }, e.toFixed(3))))),
        el("div", { class: "statgrid three" },
          el("div", { class: "stat" }, el("span", { class: "k" }, "average height"), el("span", { class: "v" }, Math.abs(mean) < 5e-4 ? "0.00" : mean.toFixed(2)), el("span", { class: "s" }, "ups cancel downs")),
          el("div", { class: "stat" }, el("span", { class: "k" }, "energies, added"), el("span", { class: "v" }, total.toFixed(3)), el("span", { class: "s" }, "sum of the bars")),
          el("div", { class: "stat" }, el("span", { class: "k" }, "energy of the curve"), el("span", { class: "v" }, ms.toFixed(3)), el("span", { class: "s" }, share != null ? `slowest wave holds ${Math.round(share * 100)}%` : "measured straight off it"))));
    }
  }

  const root = el("div", { class: "tw ws-mix" }, svg, el("div", { class: "ws-sliders" }, ...sliderRows), presets,
    showEnergy ? energyBox : null,
    el("p", { class: "cap-s" }, showEnergy
      ? "Energy of a wave = its height squared. “Energy of the curve” is measured directly off the drawn sum (twice its average squared height, so a single wave of height a reads a²)."
      : "Each thin curve is one wave, scaled by its slider. The thick curve is what you get when you add them up, point by point."));
  sync(); draw();
  return { root, redraw: draw };
}

// ---------------------------------------------------------------- tab 2: build a shape
function buildTab() {
  const main = svgEl("svg", { viewBox: "0 0 600 200", role: "img", "aria-label": "A shape rebuilt from waves" });
  const strip = svgEl("svg", { viewBox: "0 0 600 78", role: "img", "aria-label": "The wave just added" });
  const nLab = el("b", { class: "mono" });
  const slider = el("input", { type: "range", id: "ws-n", min: "0", max: String(NK), step: "1", value: String(S.n), "aria-label": "How many waves to add" });
  const sN = el("span", { class: "v" }), sAmp = el("span", { class: "v" }), sEn = el("span", { class: "v" });
  const kinds = [["step", "Step"], ["saw", "Ramp"], ["bump", "Bump"]];
  const kindBtns = kinds.map(([k, t]) => {
    const b = el("button", { "aria-pressed": String(S.kind === k) }, t);
    b.addEventListener("click", () => { S.kind = k; kindBtns.forEach((x, i) => x.setAttribute("aria-pressed", String(kinds[i][0] === k))); draw(); });
    return b;
  });
  const play = el("button", { class: "ws-play" }, "Add them one at a time");
  const stop = () => { if (S.playing) { clearInterval(S.playing); S.playing = null; } play.textContent = "Add them one at a time"; };
  play.addEventListener("click", () => {
    if (S.playing) return stop();
    S.n = 0; slider.value = "0"; draw(); play.textContent = "Pause";
    S.playing = setInterval(() => {
      if (S.n >= NK) return stop();
      S.n += 1; slider.value = String(S.n); draw();
    }, 520);
  });
  slider.addEventListener("input", () => { stop(); S.n = +slider.value; draw(); });

  function draw() {
    const rule = css("--rule"), ink3 = css("--ink-3"), acc = css("--accent"), accText = css("--accent-text"), sp = css("--spectral");
    const D = build1d(S.kind);
    const rec = xs.map((x) => { let v = D.c.a[0]; for (let k = 1; k <= S.n; k++) v += wave1d(D.c, k, x); return v; });
    const x0 = 34, x1 = 592, X = (t) => x0 + t * (x1 - x0);
    main.replaceChildren();
    // target lives between 0 and 1, so frame the plot from -0.4 to 1.4 rather than centring on zero
    const Y = (v) => 190 - ((v + 0.4) / 1.8) * 180;
    for (const v of [0, 1]) {
      main.append(svgEl("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: rule, "stroke-width": 1, ...(v ? { "stroke-dasharray": "2,4" } : {}) }));
      const t = svgEl("text", { x: x0 - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 10, fill: ink3 }); t.textContent = String(v); main.append(t);
    }
    curve(main, D.f, X, Y, ink3, 1.6, "5,4");
    curve(main, rec, X, Y, acc, 2.8);
    const lab = svgEl("text", { x: x1, y: 14, "text-anchor": "end", "font-size": 10.5, fill: accText }); lab.textContent = `${S.n} wave${S.n === 1 ? "" : "s"} added, plus the average`;
    main.append(lab);

    strip.replaceChildren();
    const sy0 = 48, amp = 22;
    strip.append(svgEl("line", { x1: x0, x2: x1, y1: sy0, y2: sy0, stroke: rule }));
    const wv = S.n >= 1 ? xs.map((x) => wave1d(D.c, S.n, x)) : null;
    const wmax = wv ? Math.max(...wv.map(Math.abs), 1e-9) : 1;
    if (wv) curve(strip, wv.map((v) => (v / wmax) * 1), X, (v) => sy0 - v * amp, sp, 2);
    const sl = svgEl("text", { x: x0 - 8, y: sy0 + 4, "text-anchor": "end", "font-size": 10, fill: ink3 }); sl.textContent = "new"; strip.append(sl);
    const st = svgEl("text", { x: x1, y: 11, "text-anchor": "end", "font-size": 10, fill: ink3 });
    st.textContent = S.n >= 1 ? `the wave just added: repeats ${S.n}×, drawn at full height` : "no waves yet: just the average";
    strip.append(st);

    nLab.textContent = String(S.n);
    let kept = 0;
    for (let k = 1; k <= S.n; k++) kept += D.e[k];
    const all = D.total;
    sN.textContent = `${S.n} / ${NK}`;
    sAmp.textContent = S.n >= 1 ? Math.hypot(D.c.a[S.n], D.c.b[S.n]).toFixed(3) : "—";
    sEn.textContent = `${all > 0 ? ((100 * kept) / all).toFixed(1) : "100.0"}%`;
  }

  const root = el("div", { class: "tw ws-build" },
    el("div", { class: "ws-toprow" }, el("div", { class: "seg", role: "group", "aria-label": "Shape" }, ...kindBtns), play),
    main, strip,
    el("label", { class: "tw-slider", for: "ws-n" }, el("span", {}, "waves added, 1 through k"), nLab),
    slider,
    el("div", { class: "statgrid three" },
      el("div", { class: "stat" }, el("span", { class: "k" }, "waves added"), sN),
      el("div", { class: "stat" }, el("span", { class: "k" }, "size of the newest"), sAmp),
      el("div", { class: "stat" }, el("span", { class: "k" }, "energy captured"), sEn)),
    el("p", { class: "cap-s" }, "Dashed: the shape you want. Orange: what the waves have built so far. Each new wave is faster and smaller than the last, and tightens the match. (“Energy captured” is the next-but-one step.)"));
  draw();
  return { root, redraw: draw, stop };
}

// ---------------------------------------------------------------- tab 3: 2-D surface
const G = 32, DISP = 44, KMAX = 16;
let SURF = null;
function surface() {
  if (SURF) return SURF;
  // a toy "rock": two lumps of one material on a background of the other. Not a dataset sample.
  const h = new Float64Array(G * G);
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const x = (i + 0.5) / G, y = (j + 0.5) / G;
    const a = (x - 0.36) ** 2 + (y - 0.4) ** 2 < 0.2 ** 2;
    const b = (x - 0.7) ** 2 + (y - 0.68) ** 2 < 0.14 ** 2;
    const c = x > 0.55 && x < 0.9 && y > 0.18 && y < 0.3;
    h[j * G + i] = a || b || c ? 1 : 0;
  }
  // 2-D DFT, direct (32^4 = 1M multiply-adds, once)
  const re = new Float64Array(G * G), im = new Float64Array(G * G);
  const ct = Array.from({ length: G }, (_, n) => Array.from({ length: G }, (_, k) => Math.cos((TAU * n * k) / G)));
  const st = Array.from({ length: G }, (_, n) => Array.from({ length: G }, (_, k) => Math.sin((TAU * n * k) / G)));
  // rows first: R[j][kx]
  const Rr = new Float64Array(G * G), Ri = new Float64Array(G * G);
  for (let j = 0; j < G; j++) for (let kx = 0; kx < G; kx++) {
    let a = 0, b = 0;
    for (let i = 0; i < G; i++) { const v = h[j * G + i]; a += v * ct[i][kx]; b -= v * st[i][kx]; }
    Rr[j * G + kx] = a; Ri[j * G + kx] = b;
  }
  for (let ky = 0; ky < G; ky++) for (let kx = 0; kx < G; kx++) {
    let a = 0, b = 0;
    for (let j = 0; j < G; j++) {
      const c = ct[j][ky], s = -st[j][ky];
      const rr = Rr[j * G + kx], ri = Ri[j * G + kx];
      a += rr * c - ri * s; b += rr * s + ri * c;
    }
    re[ky * G + kx] = a / (G * G); im[ky * G + kx] = b / (G * G);
  }
  // twiddles for display points p: e^{2 pi i f x_p / G}, f in -16..16
  const tc = [], ts = [];
  for (let f = -KMAX; f <= KMAX; f++) {
    const c = new Float64Array(DISP), s = new Float64Array(DISP);
    for (let p = 0; p < DISP; p++) { const x = (p * G) / DISP; c[p] = Math.cos((TAU * f * x) / G); s[p] = Math.sin((TAU * f * x) / G); }
    tc.push(c); ts.push(s);
  }
  let totalE = 0;
  for (let ky = 0; ky < G; ky++) for (let kx = 0; kx < G; kx++) if (ky || kx) totalE += re[ky * G + kx] ** 2 + im[ky * G + kx] ** 2;
  SURF = { h, re, im, tc, ts, totalE, cache: new Map() };
  return SURF;
}
const freqs = (K) => {
  const out = [];
  if (K >= KMAX) { for (let f = -15; f <= 16; f++) out.push(f); return out; }
  for (let f = -K; f <= K; f++) out.push(f);
  return out;
};
function rebuild2d(K) {
  const D = surface();
  if (D.cache.has(K)) return D.cache.get(K);
  const fs = freqs(K);
  const bin = (f) => (f + G) % G;
  // pass 1: along x, for each kept ky
  const A = fs.map((fy) => {
    const ar = new Float64Array(DISP), ai = new Float64Array(DISP);
    for (const fx of fs) {
      const r = D.re[bin(fy) * G + bin(fx)], m = D.im[bin(fy) * G + bin(fx)];
      const c = D.tc[fx + KMAX], s = D.ts[fx + KMAX];
      for (let p = 0; p < DISP; p++) { ar[p] += r * c[p] - m * s[p]; ai[p] += r * s[p] + m * c[p]; }
    }
    return [ar, ai];
  });
  const out = new Float64Array(DISP * DISP);
  fs.forEach((fy, a) => {
    const [ar, ai] = A[a];
    const c = D.tc[fy + KMAX], s = D.ts[fy + KMAX];
    for (let q = 0; q < DISP; q++) for (let p = 0; p < DISP; p++) out[q * DISP + p] += ar[p] * c[q] - ai[p] * s[q];
  });
  let kept = 0;
  for (const fy of fs) for (const fx of fs) if (fy || fx) kept += D.re[bin(fy) * G + bin(fx)] ** 2 + D.im[bin(fy) * G + bin(fx)] ** 2;
  const res = { field: out, frac: kept / D.totalE, nWaves: fs.length * fs.length };
  D.cache.set(K, res);
  return res;
}

function surfaceTab({ cutoff }) {
  const W = 560, H = 330;
  const cv = el("canvas", { class: "ws-3d", width: W * 2, height: H * 2, role: "img", "aria-label": "Rotatable 3-D surface rebuilt from waves. Drag to rotate." });
  const tgt = el("canvas", { class: "ws-thumb" }), cur = el("canvas", { class: "ws-thumb" });
  const kLab = el("b", { class: "mono" });
  const slider = el("input", { type: "range", id: "ws-k", min: "0", max: String(KMAX), step: "1", value: String(S.K), "aria-label": "Highest wave number kept, in each direction" });
  const sW = el("span", { class: "v" }), sE = el("span", { class: "v" }), sK = el("span", { class: "v" });

  function draw3d(field) {
    const ctx = cv.getContext("2d");
    ctx.setTransform(2, 0, 0, 2, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const L = lut("Blues"), line = css("--rule-strong");
    const ca = Math.cos(S.az), sa = Math.sin(S.az), ce = Math.cos(S.elev), se = Math.sin(S.elev);
    const SC = 118, cx = W / 2, cy = H * 0.54;
    const P = (i, j, hgt) => {
      const x = (i / (DISP - 1)) * 2 - 1, y = (j / (DISP - 1)) * 2 - 1, z = (hgt - 0.35) * 0.8;
      const X = x * ca - y * sa, Y = x * sa + y * ca;
      return { sx: cx + SC * X, sy: cy - SC * (z * ce + Y * se * 0.9), d: Y * ce - z * se, z: hgt };
    };
    const pts = new Array(DISP * DISP);
    for (let j = 0; j < DISP; j++) for (let i = 0; i < DISP; i++) pts[j * DISP + i] = P(i, j, field[j * DISP + i]);
    // floor outline, so the eye has something to hold on to
    const corners = [[0, 0], [DISP - 1, 0], [DISP - 1, DISP - 1], [0, DISP - 1]].map(([i, j]) => P(i, j, -0.1));
    ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.beginPath();
    corners.forEach((c, k) => (k ? ctx.lineTo(c.sx, c.sy) : ctx.moveTo(c.sx, c.sy))); ctx.closePath(); ctx.stroke();
    const quads = [];
    for (let j = 0; j < DISP - 1; j++) for (let i = 0; i < DISP - 1; i++) {
      const a = pts[j * DISP + i], b = pts[j * DISP + i + 1], c = pts[(j + 1) * DISP + i + 1], d = pts[(j + 1) * DISP + i];
      quads.push({ q: [a, b, c, d], d: (a.d + b.d + c.d + d.d) / 4, z: (a.z + b.z + c.z + d.z) / 4 });
    }
    quads.sort((u, v) => v.d - u.d); // far first
    ctx.lineWidth = 0.4; ctx.strokeStyle = "rgba(20,24,28,0.28)";
    for (const { q, z } of quads) {
      const t = Math.max(0, Math.min(1, (z + 0.2) / 1.5));
      const k = Math.round(t * 255) * 3;
      ctx.fillStyle = `rgb(${L[k]},${L[k + 1]},${L[k + 2]})`;
      ctx.beginPath(); ctx.moveTo(q[0].sx, q[0].sy);
      for (let n = 1; n < 4; n++) ctx.lineTo(q[n].sx, q[n].sy);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }

  function draw() {
    const D = surface();
    const r = rebuild2d(S.K);
    draw3d(r.field);
    // same colours as the rebuild beside it, so the two can be compared at a glance
    paintField(tgt, D.h, G, { mode: "sequential", vmin: -0.2, vmax: 1.2 });
    paintField(cur, r.field, DISP, { mode: "sequential", vmin: -0.2, vmax: 1.2 });
    kLab.textContent = String(S.K);
    sK.textContent = S.K === 0 ? "none" : S.K >= KMAX ? "all of them" : `up to ${S.K}`;
    sW.textContent = `${r.nWaves.toLocaleString("en-US")} / ${(G * G).toLocaleString("en-US")}`;
    sE.textContent = `${(100 * r.frac).toFixed(1)}%`;
  }
  slider.addEventListener("input", () => { S.K = +slider.value; draw(); });

  // drag to rotate
  let drag = null;
  cv.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    S.az += (e.clientX - drag.x) * 0.012;
    S.elev = Math.max(0.15, Math.min(1.45, S.elev + (e.clientY - drag.y) * 0.008));
    drag = { x: e.clientX, y: e.clientY };
    draw3d(rebuild2d(S.K).field);
  });
  const end = () => { drag = null; };
  cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);

  const root = el("div", { class: "tw ws-surf" },
    el("div", { class: "ws-surf-top" },
      cv,
      el("div", { class: "ws-thumbs" },
        el("figure", { class: "ws-goal" }, el("figcaption", {}, el("strong", {}, "What we want"), el("span", { class: "cap-s" }, "the shape to rebuild")), el("div", { class: "tile" }, tgt)),
        el("figure", {}, el("figcaption", {}, el("strong", {}, "What the waves give"), el("span", { class: "cap-s" }, "so far, with the slider below")), el("div", { class: "tile" }, cur)))),
    el("label", { class: "tw-slider", for: "ws-k" }, el("span", {}, "highest wave number kept, in each direction"), kLab),
    slider,
    cutoff ? el("p", { class: "cap-s ws-cut" }, `The model's own cutoff is ${cutoff}: set the slider to ${cutoff} to see how much of this shape it could keep.`) : null,
    el("div", { class: "statgrid three" },
      el("div", { class: "stat" }, el("span", { class: "k" }, "waves kept"), sK, el("span", { class: "s" }, "per direction")),
      el("div", { class: "stat" }, el("span", { class: "k" }, "stripe patterns"), sW),
      el("div", { class: "stat" }, el("span", { class: "k" }, "energy captured"), sE, el("span", { class: "s" }, "not counting the average"))),
    el("p", { class: "cap-s" }, "Drag the surface to turn it. Height is the shape's value at each spot. The shape is a toy of two lumps of one material on a background of the other. It's a made-up shape for this demo."));
  draw();
  return { root, redraw: draw };
}

// ---------------------------------------------------------------- public
/**
 * @param host    element to fill
 * @param opts    { id, tabs: ["mix"|"build"|"surface"...], showEnergy, cutoff }
 */
export function mountWaveSum(host, { id, tabs, showEnergy = false, cutoff = null }) {
  const labels = { mix: "Mix it yourself", build: "Build a shape", surface: "In 2-D" };
  let tab = S.tab[id] && tabs.includes(S.tab[id]) ? S.tab[id] : tabs[0];
  const body = el("div", { class: "ws-body" });
  let current = null;
  const btns = tabs.map((t) => {
    const b = el("button", { "aria-pressed": String(t === tab) }, labels[t]);
    b.addEventListener("click", () => { tab = t; S.tab[id] = t; show(); });
    return b;
  });
  function show() {
    if (current?.stop) current.stop();
    btns.forEach((b, i) => b.setAttribute("aria-pressed", String(tabs[i] === tab)));
    current = tab === "mix" ? mixTab({ showEnergy }) : tab === "build" ? buildTab() : surfaceTab({ cutoff });
    body.replaceChildren(current.root);
  }
  host.replaceChildren(el("div", { class: "ws" }, tabs.length > 1 ? el("div", { class: "seg ws-tabs", role: "group", "aria-label": "Experiment" }, ...btns) : null, body));
  show();
}
