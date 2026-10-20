// The convolution section (slides 9–19), between "Resolution invariance" and "The whole network":
// what a kernel does to one wave, why that makes it a per-wave multiply, what it costs, and
// how far this model's prediction moves when one spot of its input is poked.
// The stage re-renders on every app render, so each panel is built once and kept at module scope;
// a render only re-attaches it. Canvases read the colour tokens at draw time.
import { el, M, D, fmt, pct, paintField, paintBinary, maxAbs } from "../render.js";
import { forward } from "../fno.js";
import FFT from "../../vendor/fft.mjs";

const TAU = 2 * Math.PI;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const tokens = () => ({ ink: css("--ink"), ink3: css("--ink-3"), rule: css("--rule"), sp: css("--spectral"), soft: css("--accent-soft"), panel: css("--panel"), ground: css("--ground") });
const FONT = (px, w = 400) => `${w} ${px}px ${css("--font-mono") || "monospace"}`;

/** Size a canvas to its laid-out width at device resolution. Null while it isn't laid out. */
function fit(cv, h) {
  const w = cv.clientWidth;
  if (!w) return null;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  cv.style.height = `${h}px`;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  return { ctx, w, h };
}

const panels = new Map();
function panel(id, build) {
  if (!panels.has(id)) {
    const p = build();
    panels.set(id, p);
    // Redraw when the panel's width changes.
    const ro = new ResizeObserver(() => { if (p.root.isConnected) requestAnimationFrame(() => p.draw()); });
    p.root.querySelectorAll("canvas").forEach((c) => ro.observe(c));
  }
  return panels.get(id);
}
// Colours are read at draw time, so a flip of the colour scheme only needs a redraw.
const redrawAll = () => requestAnimationFrame(() => { for (const p of panels.values()) if (p.root.isConnected) p.draw(); });
new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);

/** A sweep from 0 to `steps` over time. It stops on its own once the canvas has left the page. */
function sweeper(steps, msPerStep, draw, isLive) {
  const s = { pos: steps, raf: 0, t0: 0 };
  const tick = (now) => {
    if (!isLive()) { s.raf = 0; return; }
    s.pos = Math.min(steps, (now - s.t0) / msPerStep);
    draw();
    s.raf = s.pos < steps ? requestAnimationFrame(tick) : 0;
  };
  s.play = () => {
    cancelAnimationFrame(s.raf); s.raf = 0;
    if (reduced()) { s.pos = steps; draw(); return; }
    s.pos = 0; s.t0 = performance.now();
    s.raf = requestAnimationFrame(tick);
  };
  /** Pick up where it left off if it was cut short by a re-render. */
  s.resume = () => {
    if (s.raf || s.pos >= steps) return draw();
    s.t0 = performance.now() - s.pos * msPerStep;
    s.raf = requestAnimationFrame(tick);
  };
  s.done = () => s.pos >= steps;
  return s;
}

const seg = (options, get, set) => {
  const btns = options.map(([v, label]) => el("button", { type: "button", onclick: () => { set(v); sync(); } }, label));
  const sync = () => btns.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i][0] === get())));
  sync();
  return el("div", { class: "seg" }, ...btns);
};
const stat = (k, v) => el("div", { class: "stat" }, el("span", { class: "k" }, k), v);

// ---------- shared: periodic 1-D signals and kernels ----------
// 32 samples, so each one is wide enough to point at.
const N = 32;
const HALF = N / 2;
/** Kernel taps as [u, κ(u)]: y(n) = Σ_u κ(u) x(n − u), indices wrapping around. */
const KERNELS = {
  average: { label: "average", taps: [[-2, 1 / 16], [-1, 4 / 16], [0, 6 / 16], [1, 4 / 16], [2, 1 / 16]],
    about: "Five weights that add up to 1. The middle sample counts most and its neighbours less, so each output is an average of a sample and the ones around it." },
  difference: { label: "difference", taps: [[-1, 1], [0, 0], [1, -1]],
    about: "Weights −1 and +1 on either side of the middle. Each output is the sample ahead minus the sample behind, so it is large where the signal changes fast: positive going up, negative going down." },
  shift: { label: "shift by 3", taps: [[3, 1]],
    about: "A single weight of 1, three samples back. Each output just copies the input from three samples earlier." },
  onesided: { label: "one-sided average", taps: [[0, 0.5], [1, 0.3], [2, 0.15], [3, 0.05]],
    about: "Four weights that add up to 1, all on one side. Each output is an average of a sample and the ones just before it." },
};
const KOPTS = Object.entries(KERNELS).map(([k, v]) => [k, v.label]);
const mod = (a, n) => ((a % n) + n) % n;

function convolve(x, taps) {
  const y = new Float64Array(N);
  for (let n = 0; n < N; n++) { let s = 0; for (const [u, w] of taps) s += w * x[mod(n - u, N)]; y[n] = s; }
  return y;
}
/** What the kernel does to wave k: C = Σκ(u)cos(2πku/N), S = Σκ(u)sin(2πku/N), R = C − iS. */
function response(taps, k) {
  let C = 0, S = 0;
  for (const [u, w] of taps) { C += w * Math.cos((TAU * k * u) / N); S += w * Math.sin((TAU * k * u) / N); }
  return { C, S, mag: Math.hypot(C, S), arg: -Math.atan2(S, C) };
}
/**
 * How far wave k moves, in points. A shift is only defined up to whole wavelengths, so pick the one
 * nearest the kernel's centre of weight: "shift by 3" then reads as 3 for every wave, not a wrapped −1.
 */
function shiftOf(taps, k, R) {
  if (k === 0 || R.mag < 1e-9) return 0;
  const lam = N / k, sum = taps.reduce((a, [, w]) => a + w, 0);
  const centre = Math.abs(sum) > 1e-9 ? taps.reduce((a, [u, w]) => a + u * w, 0) / sum : 0;
  const d = (-R.arg * N) / (TAU * k);
  return d + Math.round((centre - d) / lam) * lam;
}
const n2 = (v) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));

function plotLine(ctx, vals, X, Y, n = vals.length) {
  ctx.beginPath();
  for (let i = 0; i < n; i++) i ? ctx.lineTo(X(i), Y(vals[i])) : ctx.moveTo(X(i), Y(vals[i]));
  ctx.stroke();
}
function label(ctx, t, x, y, c, align = "left", size = 11) {
  ctx.font = FONT(size); ctx.fillStyle = c; ctx.textAlign = align; ctx.textBaseline = "alphabetic"; ctx.fillText(t, x, y);
}
function dot(ctx, x, y, r, c) { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill(); }

// ---------- convwave … convout: what a convolution does to one wave ----------
/** Five scroll steps (convwave … convout) share this one panel; each step is one part of the story. */
export const WAVE_PARTS = ["convwave", "convkernel", "convone", "convslide", "convout"];

function buildConvWave() {
  const PHI = 0.6;
  const st = { kernel: "average", k: 4, n: 8, seen: new Set([8]), part: 0 };
  const cv = el("canvas", { class: "cv-canvas", role: "img", "aria-label": "A wave, the kernel lined up under it, the products, and the output" });
  const kLab = el("b", { class: "mono" });
  const slider = el("input", { type: "range", min: "1", max: "12", step: "1", value: String(st.k), "aria-label": "How many times the wave repeats, k" });
  const sw = sweeper(N, 200, () => p.draw(), () => cv.isConnected);
  const replay = el("button", { type: "button", class: "cv-btn", onclick: () => sw.play() }, "↻ play again");
  const move = (d) => { st.n = mod(st.n + d, N); st.seen.add(st.n); p.draw(); };
  const nav = el("div", { class: "cv-ctl" },
    el("button", { type: "button", class: "cv-btn", onclick: () => move(-1) }, "◀ move left"),
    el("button", { type: "button", class: "cv-btn", onclick: () => move(1) }, "move right ▶"));
  const about = el("p", { class: "cs-about" });
  const math = el("p", { class: "cs-math mono" });
  const verdict = el("p", { class: "cs-verdict" });
  // the proof sits under the plot on the last part
  const tex = String.raw;
  // live numbers for the kernel being shown, so the algebra lands on the plot above it
  const vC = el("span", { class: "cv-live" }), vS = el("span", { class: "cv-live" });
  const vA = el("span", { class: "cv-live" }), vD = el("span", { class: "cv-live" });
  const proof = el("div", { class: "cv-proof" },
    el("h5", {}, "Why a sine wave always comes out"),
    el("p", {}, el("strong", {}, "The input. "), "The grey wave repeats k times across N points, so its frequency is ", M(tex`\omega = 2\pi k/N`), ". ", M(tex`\varphi`), " is where it starts:"),
    D(tex`x(n) = \cos(\omega n + \varphi)`),
    el("p", {}, el("strong", {}, "The convolution. "), "Call the kernel's weights ", M(tex`\kappa(u)`), ": ", M(tex`\kappa(u)`), " is the weight that multiplies the sample u points back from the output point. So ", M(tex`\kappa(0)`), " sits on the point itself, ", M(tex`\kappa(1)`), " one point back, and ", M(tex`\kappa(-1)`), " one point ahead. The average kernel, for example, is ", M(tex`\kappa(0) = 0.375`), ", ", M(tex`\kappa(\pm 1) = 0.25`), ", ", M(tex`\kappa(\pm 2) = 0.0625`), ". The output at point n multiplies each weight by its sample and adds them up. Going u points back only subtracts ", M(tex`\omega u`), " from the angle:"),
    D(tex`\begin{aligned} y(n) &= \sum_u \kappa(u)\,x(n-u) \\ &= \sum_u \kappa(u)\cos(\omega n + \varphi - \omega u) \end{aligned}`),
    el("p", {}, el("strong", {}, "Pull n out of the sum. "), "Expand each cosine with the angle-difference rule, ", M(tex`\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta`), ", using ", M(tex`\alpha = \omega n + \varphi`), " and ", M(tex`\beta = \omega u`), ":"),
    D(tex`\begin{aligned} y(n) &= \sum_u \kappa(u)\big[\cos(\omega n + \varphi)\cos\omega u \\ &\qquad\qquad + \sin(\omega n + \varphi)\sin\omega u\big] \end{aligned}`),
    el("p", {}, M(tex`\cos(\omega n + \varphi)`), " and ", M(tex`\sin(\omega n + \varphi)`), " don't depend on u, so they come out of the sum:"),
    D(tex`\begin{aligned} y(n) &= \cos(\omega n + \varphi)\sum_u \kappa(u)\cos\omega u \\ &\quad + \sin(\omega n + \varphi)\sum_u \kappa(u)\sin\omega u \end{aligned}`),
    el("p", {}, "Call those two sums C and S:"),
    D(tex`C = \sum_u \kappa(u)\cos\omega u`),
    D(tex`S = \sum_u \kappa(u)\sin\omega u`),
    D(tex`y(n) = C\cos(\omega n + \varphi) + S\sin(\omega n + \varphi)`),
    el("p", {}, "C and S don't depend on n at all. They're two fixed numbers, set by the kernel's weights and the wave's frequency. For the kernel shown above, ", vC, " and ", vS, "."),
    el("p", {}, el("strong", {}, "Fold it back into one wave. "), "Write C and S as a length A and an angle θ, the way you'd turn a point (C, S) into a distance and a direction:"),
    D(tex`C = A\cos\theta, \qquad S = A\sin\theta`),
    D(tex`A = \sqrt{C^2 + S^2}, \qquad \theta = \operatorname{atan2}(S, C)`),
    el("p", {}, "Substitute those in, and the same angle-difference rule, run backwards, turns the two terms into one:"),
    D(tex`\begin{aligned} y(n) &= A\big[\cos\theta\cos(\omega n + \varphi) \\ &\qquad + \sin\theta\sin(\omega n + \varphi)\big] \\ &= A\cos(\omega n + \varphi - \theta) \end{aligned}`),
    el("p", {}, el("strong", {}, "What it means. "), "That's the input wave again with the same frequency ", M(tex`\omega`), ". It's just ", M("A"), " times as tall and shifted by ", M(tex`\theta/\omega`), " points. For the kernel shown above that's ", vA, " and ", vD, ", which is exactly the orange wave in the plot."),
    );
  // the live bits that belong to each part: what the kernel is, the sum at one spot, replay, the result
  const live = el("div", { class: "cs-live" });
  const LIVE = [[], [about], [math, nav], [replay], [verdict]];
  const changed = () => (st.part === 3 ? sw.play() : p.draw());

  slider.addEventListener("input", () => { st.k = +slider.value; changed(); });
  cv.addEventListener("click", (e) => {
    if (st.part !== 2) return;
    const b = cv.getBoundingClientRect(), L = 40, Rt = b.width - 14;
    st.n = Math.max(0, Math.min(N - 1, Math.round(((e.clientX - b.left - L) / (Rt - L)) * (N - 1))));
    st.seen.add(st.n); p.draw();
  });

  const p = {
    root: el("div", { class: "panel-body cv" },
      el("h4", { class: "drill-sub" }, "What a kernel does to a single wave"),
      el("div", { class: "cv-ctl" }, el("span", { class: "cv-lab" }, "kernel"), seg(KOPTS, () => st.kernel, (v) => { st.kernel = v; changed(); })),
      el("div", { class: "tw" }, el("label", { class: "tw-slider" }, el("span", {}, "how many times the wave repeats, k"), kLab), slider),
      live,
      cv,
      proof),
    show(fresh, part) {
      st.part = part;
      live.replaceChildren(...LIVE[part]);
      live.hidden = !LIVE[part].length;
      cv.classList.toggle("cv-click", part === 2);
      if (part === 3) fresh ? sw.play() : sw.resume(); else p.draw();
    },
    draw() {
      const taps = KERNELS[st.kernel].taps, k = st.k;
      const x = Float64Array.from({ length: N }, (_, n) => Math.cos((TAU * k * n) / N + PHI));
      const y = convolve(x, taps), R = response(taps, k);
      kLab.textContent = k; slider.value = String(k);
      about.textContent = KERNELS[st.kernel].about;
      // the arithmetic at the current spot, one product per weight, left to right as drawn
      const terms = taps.filter(([, w]) => w !== 0).sort((p, q) => q[0] - p[0]).map(([u, w]) => { const v = x[mod(st.n - u, N)]; return `${n2(w)} × ${v < 0 ? `(${n2(v)})` : n2(v)}`; });
      math.textContent = `output at point ${st.n} = ${terms.join(" + ")} = ${n2(y[st.n])}`;
      const shift = shiftOf(taps, k, R);
      const where = Math.abs(shift) < 0.005 ? "in the same place" : `moved ${fmt(Math.abs(shift), 2)} points to the ${shift > 0 ? "right" : "left"}`;
      verdict.replaceChildren(el("strong", {}, `A sine wave in, a sine wave out.`),
        ` Both repeat ${k} times. The one coming out is ${fmt(R.mag, 2)}× as tall and ${where}.`);
      vC.replaceChildren(M(tex`C = ${fmt(R.C, 3)}`)); vS.replaceChildren(M(tex`S = ${fmt(R.S, 3)}`));
      vA.textContent = `${fmt(R.mag, 2)}× as tall`;
      vD.textContent = Math.abs(shift) < 0.005 ? "no shift" : `a shift of ${fmt(Math.abs(shift), 2)} points to the ${shift > 0 ? "right" : "left"}`;
      p.check = { y, R, PHI, k, shift };
      proof.hidden = st.part !== 4;
      if (st.part === 4) drawOut(cv, { y, R, k, PHI, shift, taps, name: KERNELS[st.kernel].label });
      else drawWave(cv, { x, y, taps, R, k, PHI, stage: st.part, n: st.n, seen: st.seen, pos: sw.pos, done: sw.done() });
    },
  };
  return p;
}

function drawWave(cv, { x, y, taps, R, k, PHI, stage, n, seen, pos, done }) {
  const mid3 = stage === 1 || stage === 2 || (stage === 3 && !done);
  const rows = [["input", 96], ...(mid3 ? [["kernel", 78]] : []), ...(mid3 && stage >= 2 ? [["products", 78]] : []), ...(stage >= 2 ? [["output", 110]] : []), ...(stage === 1 ? [["block", 110]] : [])];
  const f = fit(cv, rows.reduce((a, [, h]) => a + h + 22, 6));
  if (!f) return;
  const { ctx, w } = f, T = tokens();
  const L = 40, Rt = w - 14, X = (i) => L + (i / (N - 1)) * (Rt - L), dx = (Rt - L) / (N - 1);
  const top = {}; let yy = 20;
  for (const [name, h] of rows) { top[name] = [yy, h]; yy += h + 22; }
  const mid = (name) => top[name][0] + top[name][1] / 2;
  const scaleY = (name, lim) => (v) => mid(name) - (v / lim) * (top[name][1] / 2 - 6);
  const axis = (name) => { ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, mid(name)); ctx.lineTo(Rt, mid(name)); ctx.stroke(); };
  const head = (name, t, c = T.ink3) => label(ctx, t, L, top[name][0] - 6, c);

  // where the kernel sits: the current spot in parts 2–3, the sweep's spot in part 4
  const sweeping = stage === 3 && !done;
  const at = stage === 3 ? Math.min(N - 1, Math.floor(pos)) : n;
  const showWin = stage === 1 || stage === 2 || sweeping;
  const cols = taps.map(([u]) => mod(at - u, N));
  if (showWin) {
    const last = stage >= 2 ? "products" : "kernel", y0 = top.input[0], y1 = top[last][0] + top[last][1];
    ctx.fillStyle = T.soft;
    for (const c of cols) ctx.fillRect(X(c) - dx / 2 + 1, y0, dx - 2, y1 - y0);
  }

  const Yi = scaleY("input", 1.1);
  head("input", "input");
  axis("input");
  ctx.strokeStyle = T.rule; ctx.lineWidth = 1.2; plotLine(ctx, x, X, Yi);
  for (let i = 0; i < N; i++) dot(ctx, X(i), Yi(x[i]), showWin && cols.includes(i) ? 3.6 : 2.6, showWin && cols.includes(i) ? T.ink : T.ink3);

  if (top.kernel) {
    const kl = Math.max(...taps.map(([, wt]) => Math.abs(wt))) * 1.6, Yk = scaleY("kernel", kl);
    head("kernel", "kernel: one weight under each sample it reads");
    axis("kernel");
    if (showWin) {
      taps.forEach(([, wt], i) => {
        ctx.strokeStyle = T.sp; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(X(cols[i]), mid("kernel")); ctx.lineTo(X(cols[i]), Yk(wt)); ctx.stroke();
        dot(ctx, X(cols[i]), Yk(wt), 3.2, T.sp);
        if (dx >= 26) label(ctx, n2(wt), X(cols[i]), wt >= 0 ? Yk(wt) - 7 : Yk(wt) + 15, T.sp, "center", 10);
      });
    }
  }

  // the same kernel on a block with sharp sides: average rounds the corners, difference spikes at the
  // two jumps, shift moves the block, one-sided average smears only one side
  if (top.block) {
    const xb = Float64Array.from({ length: N }, (_, i) => (i >= 10 && i <= 21 ? 1 : 0)), yb = convolve(xb, taps);
    const Yb = scaleY("block", 1.15);
    head("block", w < 560 ? "on a block: grey in, orange out" : "the same kernel on a block (grey), and what comes out (orange)");
    axis("block");
    ctx.strokeStyle = T.ink3; ctx.lineWidth = 1.6; ctx.beginPath();
    for (let i = 0; i < N; i++) { const a = X(i) - dx / 2, b = X(i) + dx / 2; i ? ctx.lineTo(a, Yb(xb[i])) : ctx.moveTo(a, Yb(xb[i])); ctx.lineTo(b, Yb(xb[i])); }
    ctx.stroke();
    ctx.strokeStyle = T.sp; ctx.lineWidth = 2; plotLine(ctx, yb, X, Yb);
    for (let i = 0; i < N; i++) dot(ctx, X(i), Yb(yb[i]), 2.6, T.sp);
  }

  if (top.products) {
    const kl = Math.max(...taps.map(([, wt]) => Math.abs(wt))) * 1.15, Yp = scaleY("products", kl);
    head("products", "weight × sample above it");
    axis("products");
    if (showWin) {
      taps.forEach(([, wt], i) => {
        const v = wt * x[cols[i]];
        ctx.strokeStyle = T.ink3; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(X(cols[i]), mid("products")); ctx.lineTo(X(cols[i]), Yp(v)); ctx.stroke();
      });
    }
  }
  if (stage >= 2) {

    const lim = Math.max(1, ...Array.from({ length: HALF + 1 }, (_, kk) => response(taps, kk).mag)) * 1.08, Yo = scaleY("output", lim);
    head("output", stage === 4 ? "output (orange) against the input (grey)" : "output: the sum of the products, one point per spot");
    axis("output");
    if (stage === 4) {
      ctx.strokeStyle = T.rule; ctx.lineWidth = 2; plotLine(ctx, x, X, Yo);
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); ctx.beginPath();
      for (let i = 0; i <= 400; i++) { const t = (i / 400) * (N - 1), v = R.mag * Math.cos((TAU * k * t) / N + PHI + R.arg); i ? ctx.lineTo(X(t), Yo(v)) : ctx.moveTo(X(t), Yo(v)); }
      ctx.stroke(); ctx.setLineDash([]);
    }
    const shown = stage === 2 ? [...seen] : stage === 3 ? Array.from({ length: done ? N : at + 1 }, (_, i) => i) : Array.from({ length: N }, (_, i) => i);
    if (stage >= 3) { ctx.strokeStyle = T.sp; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.5; plotLine(ctx, y, X, Yo, stage === 3 && !done ? at + 1 : N); ctx.globalAlpha = 1; }
    for (const i of shown) dot(ctx, X(i), Yo(y[i]), 3, T.sp);
    if (showWin) {
      ctx.strokeStyle = T.sp; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(X(at), top.products[0] + top.products[1]); ctx.lineTo(X(at), Yo(y[at])); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = T.sp; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(at), Yo(y[at]), 6, 0, TAU); ctx.stroke();
      if (stage === 2) label(ctx, n2(y[at]), X(at) + (X(at) > w - 60 ? -10 : 10), Yo(y[at]) - 8, T.sp, X(at) > w - 60 ? "right" : "left", 11);
    }
  }
}

/** The result, on its own: the kernel, the sine wave going in, the sine wave coming out, and how they differ. */
function drawOut(cv, { y, R, k, PHI, shift, taps, name }) {
  const kTop = cv.clientWidth < 560 ? 72 : 56, kH = 96;
  const f = fit(cv, (cv.clientWidth < 560 ? 340 : 320) + kH + 8);
  if (!f) return;
  const { ctx, w, h } = f, T = tokens();
  const L = 20, Rt = w - 20, top = kTop + kH + 30, bot = h - 34;
  const X = (t) => L + (t / (N - 1)) * (Rt - L), dx = (Rt - L) / (N - 1);

  // the kernel, at the same point spacing as the waves below, so its width reads against one wavelength.
  // κ(u) reads the sample u points back, so positive u is drawn to the left of the output point.
  const kl = Math.max(...taps.map(([, wt]) => Math.abs(wt))), neg = taps.some(([, wt]) => wt < 0);
  const kBase = neg ? kTop + 18 + (kH - 18) / 2 : kTop + kH - 4, kAmp = neg ? (kH - 18) / 2 - 6 : kH - 30;
  const c0 = (N - 1) / 2, Kx = (u) => X(c0 - u), Ky = (wt) => kBase - (wt / kl) * kAmp;
  label(ctx, cv.clientWidth < 560 ? `kernel: ${name}` : `kernel: ${name}, centred on the dashed line (the output point)`, L, kTop + 4, T.ink, "left", 11);
  ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, kBase); ctx.lineTo(Rt, kBase); ctx.stroke();
  ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X(c0), kBase - kAmp - 4); ctx.lineTo(X(c0), kBase + (neg ? kAmp + 4 : 4)); ctx.stroke(); ctx.setLineDash([]);
  for (const [u, wt] of taps) {
    ctx.strokeStyle = T.sp; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(Kx(u), kBase); ctx.lineTo(Kx(u), Ky(wt)); ctx.stroke();
    dot(ctx, Kx(u), Ky(wt), 3.2, T.sp);
    if (dx >= 22 && wt !== 0) label(ctx, n2(wt), Kx(u), wt >= 0 ? Ky(wt) - 7 : Ky(wt) + 14, T.sp, "center", 10);
  }

  const lim = Math.max(1, R.mag) * 1.08, Y = (v) => (top + bot) / 2 - (v / lim) * ((bot - top) / 2);
  const wave = (a, ph, color, lw) => {
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
    for (let i = 0; i <= 400; i++) { const t = (i / 400) * (N - 1), v = a * Math.cos((TAU * k * t) / N + ph); i ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)); }
    ctx.stroke();
  };
  // legend on the left; the two numbers that differ on the right, or on a third line when narrow
  const narrow = w < 560;
  label(ctx, "grey: the sine wave going in", L, 16, T.ink3, "left", 11);
  label(ctx, "orange: the sine wave coming out", L, 32, T.sp, "left", 11);
  label(ctx, `both repeat ${k}× · orange is ${fmt(R.mag, 2)}× as tall`, narrow ? L : Rt, narrow ? 48 : 16, T.ink, narrow ? "left" : "right", 11);
  ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(Rt, Y(0)); ctx.stroke();
  wave(1, PHI, T.ink3, 2.2);
  wave(R.mag, PHI + R.arg, T.sp, 3);
  for (let n = 0; n < N; n++) dot(ctx, X(n), Y(y[n]), 2.4, T.ink);

  // one input peak near the middle, and the output peak it turned into
  const lam = N / k;
  let tin = ((-PHI / TAU) * N) / k;
  while (tin < (N - 1) * 0.3) tin += lam;
  while (tin > (N - 1) * 0.3 + lam) tin -= lam;
  const tout = tin + shift;
  ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2;
  ctx.strokeStyle = T.ink3; ctx.beginPath(); ctx.moveTo(X(tin), Y(0)); ctx.lineTo(X(tin), Y(1)); ctx.stroke();
  ctx.strokeStyle = T.sp; ctx.beginPath(); ctx.moveTo(X(tout), Y(0)); ctx.lineTo(X(tout), Y(R.mag)); ctx.stroke();
  ctx.setLineDash([]);
  dot(ctx, X(tin), Y(1), 4, T.ink3);
  dot(ctx, X(tout), Y(R.mag), 4.5, T.sp);
  if (Math.abs(shift) > 0.05) {
    const ya = Y(lim) + 2, a = X(tin), b = X(tout), dir = Math.sign(b - a);
    ctx.strokeStyle = T.sp; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(a, ya); ctx.lineTo(b, ya); ctx.lineTo(b - dir * 6, ya - 4); ctx.moveTo(b, ya); ctx.lineTo(b - dir * 6, ya + 4); ctx.stroke();
    label(ctx, `moved ${fmt(Math.abs(shift), 2)} points`, (a + b) / 2, ya - 7, T.sp, "center", 11);
  }
  label(ctx, w < 560 ? "dots: the output, all on the orange wave" : `dots: the convolution's output at each of the ${N} points, all on the orange wave`, L, h - 8, T.ink3, "left", 10.5);
}

// ---------- convsum … convadd: any signal is a sum of waves, so change each wave ----------
/** The example signal: secretly three waves, slow, medium and fast, so the split is easy to see. */
const PARTS3 = [[1, 1.0, 0.3], [3, 0.6, 1.0], [8, 0.5, 0.4]];
const signal3 = () => Float64Array.from({ length: N }, (_, n) => PARTS3.reduce((s, [k, a, ph]) => s + a * Math.cos((TAU * k * n) / N + ph), 0));
const SUM_SAYS = {
  average: "Average barely touches the slow wave and almost wipes out the fast one.",
  difference: "Difference does the opposite: it almost wipes out the slow wave and doubles the fast one.",
  shift: "Shift by 3 leaves every height alone and moves every wave 3 points along.",
  onesided: "One-sided average shrinks the fast wave most and nudges each wave along a little.",
};
const fft = new FFT(N);
function spectrum(x) {
  const a = fft.createComplexArray(), out = fft.createComplexArray();
  for (let i = 0; i < N; i++) a[2 * i] = x[i];
  fft.transform(out, a);
  return out;
}
/** Output by the waves: FFT the input, multiply wave k by R_k, inverse FFT. */
function viaWaves(x, taps) {
  const X = spectrum(x), Y = fft.createComplexArray(), y = fft.createComplexArray();
  for (let k = 0; k < N; k++) {
    let rr = 0, ri = 0;
    for (const [u, w] of taps) { rr += w * Math.cos((TAU * k * u) / N); ri -= w * Math.sin((TAU * k * u) / N); }
    const xr = X[2 * k], xi = X[2 * k + 1];
    Y[2 * k] = xr * rr - xi * ri; Y[2 * k + 1] = xr * ri + xi * rr;
  }
  fft.inverseTransform(y, Y);
  return Float64Array.from({ length: N }, (_, i) => y[2 * i]);
}
/** The input as HALF + 1 real waves: a_k·cos(2πkn/N + φ_k), which add back up to it exactly. */
function waves(x) {
  const X = spectrum(x);
  return Array.from({ length: HALF + 1 }, (_, k) => ({
    k, amp: ((k === 0 || k === HALF ? 1 : 2) * Math.hypot(X[2 * k], X[2 * k + 1])) / N, phase: Math.atan2(X[2 * k + 1], X[2 * k]),
  }));
}

/** Four scroll steps (convsum … convadd) share this one panel: the input, split, change each wave, add back up. */
export const SUM_PARTS = ["convsum", "convsplit", "convchange", "convadd"];

function buildConvSum() {
  const st = { kernel: "average", part: 0 };
  const cv = el("canvas", { class: "cv-canvas", role: "img", "aria-label": "A signal, the three waves it is made of, each wave after the kernel, and their sum" });
  const diff = el("span", { class: "v" });
  const morph = sweeper(1, 2400, () => p.draw(), () => cv.isConnected);
  const adder = sweeper(3, 1400, () => p.draw(), () => cv.isConnected);
  const says = el("p", { class: "cs-verdict" });
  const gapBox = el("div", { class: "statgrid cv-one" }, stat("largest gap between the sum and sliding directly", diff));
  const LIVE = [[], [],
    [says, el("button", { type: "button", class: "cv-btn", onclick: () => morph.play() }, "↻ play again")],
    [el("button", { type: "button", class: "cv-btn", onclick: () => adder.play() }, "↻ play again"), gapBox]];
  const live = el("div", { class: "cs-live" });
  const anim = () => (st.part === 2 ? morph : st.part === 3 ? adder : null);
  const changed = () => (anim() ? anim().play() : p.draw());
  const p = {
    root: el("div", { class: "panel-body cv" },
      el("h4", { class: "drill-sub" }, "Split a signal into waves, change each wave, add them back up"),
      el("div", { class: "cv-ctl" }, el("span", { class: "cv-lab" }, "kernel"), seg(KOPTS, () => st.kernel, (v) => { st.kernel = v; changed(); })),
      live,
      cv),
    show(fresh, part) {
      st.part = part;
      live.replaceChildren(...LIVE[part]);
      live.hidden = !LIVE[part].length;
      const a = anim();
      if (a) fresh ? a.play() : a.resume(); else p.draw();
    },
    draw() {
      const taps = KERNELS[st.kernel].taps, x = signal3();
      const direct = convolve(x, taps), byWaves = viaWaves(x, taps);
      let d = 0; for (let i = 0; i < N; i++) d = Math.max(d, Math.abs(direct[i] - byWaves[i]));
      diff.textContent = d.toExponential(1);
      says.textContent = SUM_SAYS[st.kernel];
      gapBox.hidden = !adder.done(); // only once the sum is complete
      p.check = { maxDiff: d };
      // the waves the transform finds; everything else is zero to rounding
      const W = waves(x).filter((wv) => wv.amp > 1e-9).map((wv) => { const R = response(taps, wv.k); return { ...wv, R, shift: shiftOf(taps, wv.k, R) }; });
      const tau = st.part < 2 ? 0 : st.part === 2 ? morph.pos : 1;
      drawSum(cv, { x, W, direct, stage: st.part, tau, added: st.part === 3 ? adder.pos : 0, done: adder.done() });
    },
  };
  return p;
}

function drawSum(cv, { x, W, direct, stage, tau, added, done }) {
  const ROW = 84, nW = W.length, inH = 110, outH = 130;
  const H = 22 + inH + (stage >= 1 ? 36 + nW * ROW : 0) + (stage >= 3 ? 46 + outH : 0) + 6;
  const f = fit(cv, H);
  if (!f) return;
  const { ctx, w } = f, T = tokens();
  const L = 14, Rt = w - 14;
  const Xt = (t) => L + (t / (N - 1)) * (Rt - L);
  // part 3 grows or shrinks each wave and slides it by its shift, in step
  const at = (wv, t, s) => (1 + s * (wv.R.mag - 1)) * wv.amp * Math.cos((TAU * wv.k * (t - s * wv.shift)) / N + wv.phase);
  const curve = (fn, Y) => { ctx.beginPath(); for (let i = 0; i <= 320; i++) { const t = (i / 320) * (N - 1); i ? ctx.lineTo(Xt(t), Y(fn(t))) : ctx.moveTo(Xt(t), Y(fn(t))); } ctx.stroke(); };
  const zero = (y) => { ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(Rt, y); ctx.stroke(); };
  // how much of wave r is in the running sum: 0 before its turn, 1 once added
  const share = (r) => Math.max(0, Math.min(1, added - r));

  let y0 = 22;
  const lim = Math.max(...x.map(Math.abs)) * 1.12, Yin = (v) => y0 + inH / 2 - (v / lim) * (inH / 2);
  label(ctx, "input", L, y0 - 8, T.ink3);
  zero(Yin(0));
  ctx.strokeStyle = T.ink; ctx.lineWidth = 1.8; plotLine(ctx, x, Xt, Yin);
  for (let i = 0; i < N; i++) dot(ctx, Xt(i), Yin(x[i]), 2.4, T.ink);
  if (stage === 1) {
    ctx.strokeStyle = T.sp; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
    plotLine(ctx, Float64Array.from({ length: N }, (_, i) => W.reduce((s, wv) => s + at(wv, i, 0), 0)), Xt, Yin);
    ctx.setLineDash([]);
    label(ctx, "dashed: the three waves added up", Rt, y0 - 8, T.sp, "right", 10.5);
  }
  if (stage < 1) return;

  // one row per wave, on one height scale so their sizes compare
  y0 += inH + 36;
  const big = Math.max(...W.map((wv) => wv.amp * (stage >= 2 ? Math.max(1, wv.R.mag) : 1))) || 1;
  const half = ROW / 2 - 14;
  const NAMES = ["slow", "medium", "fast"];
  W.forEach((wv, r) => {
    const top = y0 + r * ROW, c = top + 12 + ROW / 2 - 6, Y = (v) => c - (v / big) * half;
    const on = stage === 3 && share(r) > 0 && share(r) < 1;
    const narrow = w < 560;
    label(ctx, `${NAMES[r] ?? ""} wave${narrow ? "" : ` · repeats ${wv.k}×`}${on ? "  ← adding" : ""}`, L, top + 4, on ? T.sp : T.ink3, "left", 10.5);
    if (stage >= 2) {
      const moved = Math.abs(wv.shift) < 0.05 ? "" : ` · moved ${fmt(Math.abs(wv.shift), 1)} ${narrow ? "pts" : "points"} ${wv.shift > 0 ? "right" : "left"}`;
      label(ctx, `×${wv.R.mag.toFixed(2)}${narrow ? "" : " as tall"}${moved}`, Rt, top + 4, tau > 0.02 ? T.sp : T.ink3, "right", 10.5);
    }
    zero(c);
    ctx.strokeStyle = stage >= 2 ? T.ink3 : T.ink; ctx.lineWidth = stage >= 2 ? 1.2 : 1.8; ctx.globalAlpha = stage >= 2 ? 0.45 : 1;
    curve((t) => at(wv, t, 0), Y); ctx.globalAlpha = 1;
    if (stage >= 2) {
      ctx.strokeStyle = T.sp; ctx.lineWidth = on ? 3.2 : 2.2;
      ctx.globalAlpha = stage === 3 && share(r) === 0 ? 0.35 : 1;
      curve((t) => at(wv, t, tau), Y); ctx.globalAlpha = 1;
    }
  });
  if (stage >= 2) label(ctx, "grey: before · orange: after the kernel", L, y0 - 14, T.ink3, "left", 10);
  if (stage < 3) return;

  // the changed waves, added into the output one at a time
  y0 += nW * ROW + 46;
  const ol = Math.max(...direct.map(Math.abs)) * 1.15 || 1, Yo = (v) => y0 + outH / 2 - (v / ol) * (outH / 2);
  const names = W.map((_, r) => NAMES[r]).filter((_, r) => share(r) > 0);
  label(ctx, done ? "output: the three changed waves added up" : `output so far: ${names.length ? names.join(" + ") : "nothing yet"}`, L, y0 - 8, T.ink3);
  zero(Yo(0));
  ctx.strokeStyle = T.sp; ctx.lineWidth = 3; curve((t) => W.reduce((s, wv, r) => s + share(r) * at(wv, t, 1), 0), Yo);
  if (done) {
    for (let i = 0; i < N; i++) dot(ctx, Xt(i), Yo(direct[i]), 2.6, T.ink);
    label(ctx, "dots: sliding the kernel along the input directly, all on the orange curve", L, y0 + outH + 4, T.ink3, "left", 10.5);
  }
}

// ---------- convcost: why go through the waves ----------
const WINDOWS = [[3, "3×3"], [5, "5×5"], [9, "9×9"], [16, "whole grid"]];
const G = 16;
const SIZES = [64, 128, 256];
// Every real multiply and every real add, checked by counting them in a plain radix-2 FFT and a plain
// sliding loop (16×16, 32×32 and 64×64 all match these formulas exactly).
// Sliding: each output takes K² multiplies and K² − 1 adds; a whole-grid window has K² = N.
const perPixelSliding = (side, K) => (K === 16 ? 2 * side * side - 1 : 2 * K * K - 1);
const directOps = (side, K) => side * side * perPixelSliding(side, K);
// Waves: two FFTs at 5·N·log₂N each (textbook complex radix-2, every twiddle counted) and one complex
// multiply per wave (6). The kernel's waves aren't counted: an FNO learns them directly.
const fftOps = (side) => { const n = side * side; return 10 * n * Math.log2(n) + 6 * n; };
const big = (v) => v >= 1e9 ? `${fmt(v / 1e9, 1)}B` : v >= 1e6 ? `${fmt(v / 1e6, 1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}k` : String(Math.round(v));

/** The working behind both counts, collapsed until the reader asks for it. */
function howCounted() {
  const tex = String.raw;
  const sec = (title, ...body) => el("div", { class: "cv-how-sec" }, el("h6", {}, title), ...body);
  return el("details", { class: "cv-proof cv-how" },
    el("summary", {}, "How we count the operations"),
    el("p", {}, "We count every real multiply and every real add, for one channel on an ", M(tex`n \times n`), " grid with ", M(tex`N = n^2`), " pixels."),
    sec(el("span", {}, "Sliding a ", M(tex`K \times K`), " kernel"),
      el("p", {}, "Each output pixel multiplies ", M(tex`K^2`), " weights by the ", M(tex`K^2`), " samples under them, then adds the products together. That's ", M(tex`K^2`), " multiplies and ", M(tex`K^2 - 1`), " adds, for each of the ", M(tex`N`), " pixels:"),
      D(tex`\text{sliding} = N\,(2K^2 - 1)`),
      el("p", {}, "A kernel as wide as the grid has ", M(tex`K^2 = N`), ", so every pixel needs every other pixel: ", M(tex`N(2N - 1)`), ". At 64 × 64 that's 4,096 × 8,191 = 33,550,336.")),
    sec("Through the waves",
      el("p", {}, "Three steps: turn the input into waves, multiply each wave by its ", M(tex`R`), ", and turn the result back into pixels."),
      el("p", {}, el("strong", {}, "One FFT. "), "An FFT on ", M(tex`N`), " numbers runs ", M(tex`\log_2 N`), " rounds of ", M(tex`N/2`), " butterflies. A butterfly turns two numbers ", M(tex`a`), " and ", M(tex`b`), " into ", M(tex`a + wb`), " and ", M(tex`a - wb`), ", for a fixed angle ", M(tex`w`), ". That's one complex multiply (4 multiplies and 2 adds) and two complex adds (2 adds each), so 10 operations:"),
      D(tex`\tfrac{N}{2} \cdot \log_2 N \cdot 10 = 5N\log_2 N`),
      el("p", {}, "On an ", M(tex`n \times n`), " grid, we FFT every row and then every column. That's ", M(tex`2n`), " FFTs of size ", M(tex`n`), ", which comes to the same thing, because ", M(tex`\log_2 N = 2\log_2 n`), ":"),
      D(tex`2n \cdot 5n\log_2 n = 5N\log_2 N`),
      el("p", {}, el("strong", {}, "Multiply each wave by its ", M(tex`R`), ". "), "One complex multiply per wave, 6 operations each, so ", M(tex`6N`), "."),
      el("p", {}, el("strong", {}, "Turn it back. "), "The inverse FFT costs the same as the first one. Altogether:"),
      D(tex`\text{waves} = 10N\log_2 N + 6N`),
      el("p", {}, "At 64 × 64, ", M(tex`N = 4{,}096`), " and ", M(tex`\log_2 N = 12`), ", so 491,520 + 24,576 = 516,096, whatever the kernel's size.")),
    sec("Checked, and what's left out",
      el("p", {}, "We ran a plain sliding loop and a plain FFT that count each operation as they go. At 16 × 16, 32 × 32 and 64 × 64 the counts match these formulas exactly, and the two routes give the same output to within ", M(tex`4 \times 10^{-15}`), "."),
      el("p", {}, "Not counted: the kernel's own waves, because an FNO learns them directly, and the inverse FFT's division by ", M(tex`N`), ", which can be folded into ", M(tex`R`), ". The count includes multiplies by 1 and doesn't use the cheaper FFT for real-valued inputs, so if anything it overstates the cost of the waves. These count operations. The actual run time also depends on memory and hardware.")));
}

function buildConvCost() {
  const st = { K: 3 };
  const grid = el("canvas", { class: "cv-canvas cv-grid", role: "img", "aria-label": "A kernel window sliding over a 16 by 16 grid, wrapping at the edges" });
  const bars = el("canvas", { class: "cv-canvas", role: "img", "aria-label": "Total multiplies and adds, sliding against through the waves, on a log scale" });
  const count = el("span", { class: "v" }), perD = el("span", { class: "v" }), perF = el("span", { class: "v" });
  const sw = sweeper(G * G, 22, () => p.draw(), () => grid.isConnected);
  const replay = el("button", { type: "button", class: "cv-btn", onclick: () => sw.play() }, "↻ replay");
  const p = {
    root: el("div", { class: "panel-body cv" },
      el("h4", { class: "drill-sub" }, "One window position per output pixel, at any kernel size"),
      el("div", { class: "cv-ctl" }, el("span", { class: "cv-lab" }, "kernel"), seg(WINDOWS, () => st.K, (v) => { st.K = v; sw.play(); }), replay),
      el("div", { class: "cv-cost" },
        el("div", { class: "cv-cost-l" }, grid, el("div", { class: "statgrid cv-one" }, stat("multiplies + adds so far, sliding on this 16×16 grid", count))),
        el("div", { class: "cv-cost-r" }, el("span", { class: "cap-s" }, "total multiplies + adds, log scale"), bars)),
      el("div", { class: "statgrid" }, stat("per output pixel at 64×64, sliding", perD), stat("per output pixel at 64×64, through the waves", perF)),
      howCounted()),
    show(fresh) { fresh ? sw.play() : sw.resume(); },
    draw() {
      const K = st.K, per = perPixelSliding(G, K), done = Math.floor(sw.pos);
      count.textContent = `${done.toLocaleString("en-US")} × ${per} = ${(done * per).toLocaleString("en-US")}`;
      perD.textContent = perPixelSliding(64, K).toLocaleString("en-US");
      perF.textContent = String(fftOps(64) / 4096);
      drawGrid(grid, K, done, sw.done());
      drawCostBars(bars, K);
    },
  };
  return p;
}

function drawGrid(cv, K, done, finished) {
  const side = Math.min(cv.clientWidth, 300), f = fit(cv, side);
  if (!f) return;
  const { ctx } = f, T = tokens(), c = side / G;
  ctx.fillStyle = T.soft;
  for (let i = 0; i < done; i++) ctx.fillRect((i % G) * c, Math.floor(i / G) * c, c, c);
  if (!finished) {
    const at = Math.min(done, G * G - 1), r0 = Math.floor(at / G), c0 = at % G, h = Math.floor(K / 2);
    ctx.strokeStyle = T.sp; ctx.lineWidth = 1.5;
    for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) {
      const rr = mod(r0 - h + a, G), cc = mod(c0 - h + b, G);
      ctx.strokeRect(cc * c + 2, rr * c + 2, c - 4, c - 4);
    }
    ctx.fillStyle = T.sp; ctx.fillRect(c0 * c + c * 0.3, r0 * c + c * 0.3, c * 0.4, c * 0.4);
  }
  ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
  for (let i = 0; i <= G; i++) { ctx.beginPath(); ctx.moveTo(i * c, 0); ctx.lineTo(i * c, side); ctx.moveTo(0, i * c); ctx.lineTo(side, i * c); ctx.stroke(); }
}

function drawCostBars(cv, K) {
  const f = fit(cv, 250);
  if (!f) return;
  const { ctx, w, h } = f, T = tokens();
  const lo = 3, hiExp = 10, B = h - 34, Tp = 10, L = 30;
  const Y = (v) => B - ((Math.log10(v) - lo) / (hiExp - lo)) * (B - Tp);
  ctx.lineWidth = 1;
  for (let e = lo; e <= hiExp; e++) {
    ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(L, Y(10 ** e)); ctx.lineTo(w - 4, Y(10 ** e)); ctx.stroke();
    if (e % 3 === 0) label(ctx, `1e${e}`, L - 4, Y(10 ** e) + 4, T.ink3, "right", 9.5);
  }
  const gw = (w - L - 4) / SIZES.length, bw = Math.min(28, gw * 0.28);
  SIZES.forEach((s, i) => {
    const cx = L + gw * (i + 0.5), d = directOps(s, K), fo = fftOps(s);
    ctx.fillStyle = T.ink3; ctx.fillRect(cx - bw - 3, Y(d), bw, B - Y(d));
    ctx.fillStyle = T.sp; ctx.fillRect(cx + 3, Y(fo), bw, B - Y(fo));
    label(ctx, big(d), cx - bw / 2 - 3, Y(d) - 4, T.ink3, "center", 9.5);
    label(ctx, big(fo), cx + bw / 2 + 3, Y(fo) - 4, T.sp, "center", 9.5);
    label(ctx, `${s}×${s}`, cx, B + 14, T.ink, "center", 10.5);
  });
  ctx.fillStyle = T.ink3; ctx.fillRect(L, h - 11, 9, 9); label(ctx, "sliding", L + 13, h - 3, T.ink3, "left", 10);
  ctx.fillStyle = T.sp; ctx.fillRect(L + 74, h - 11, 9, 9); label(ctx, "through the waves", L + 87, h - 3, T.sp, "left", 10);
}

// ---------- convwide: why this network wants a wide kernel ----------
const REACH = 4; // four layers, each a 3×3 window, reach ±1 pixel apiece
const R64 = 64;
const wideCache = new Map();
let appRef = null;

function poke(key, input, at) {
  const out = Float64Array.from(input);
  const [r0, c0] = at;
  if (key === "ns") {
    const amp = 0.9 * maxAbs(input), s2 = 2 * 2 ** 2;
    for (let r = 0; r < R64; r++) for (let c = 0; c < R64; c++) {
      const dr = Math.min(Math.abs(r - r0), R64 - Math.abs(r - r0)), dc = Math.min(Math.abs(c - c0), R64 - Math.abs(c - c0));
      out[r * R64 + c] += amp * Math.exp(-(dr * dr + dc * dc) / s2);
    }
  } else {
    let lo = Infinity, hi = -Infinity;
    for (const v of input) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const rs = Math.max(1, Math.min(R64 - 2, r0)), cs = Math.max(1, Math.min(R64 - 2, c0));
    for (let r = rs - 1; r <= rs + 1; r++) for (let c = cs - 1; c <= cs + 1; c++) { const i = r * R64 + c; out[i] = input[i] > (lo + hi) / 2 ? lo : hi; }
  }
  return out;
}
/** Chebyshev distance, wrapping for the fluid (its domain is periodic), not for the rock. */
const cheb = (key, r, c, [r0, c0]) => {
  let dr = Math.abs(r - r0), dc = Math.abs(c - c0);
  if (key === "ns") { dr = Math.min(dr, R64 - dr); dc = Math.min(dc, R64 - dc); }
  return Math.max(dr, dc);
};
function measure(key, delta, at) {
  let tot = 0, out = 0;
  const byD = new Float64Array(R64);
  for (let r = 0; r < R64; r++) for (let c = 0; c < R64; c++) {
    const e = delta[r * R64 + c] ** 2, d = cheb(key, r, c, at);
    tot += e; byD[d] += e; if (d > REACH) out += e;
  }
  let acc = 0, r90 = 0;
  for (let d = 0; d < R64; d++) { acc += byD[d]; if (acc >= 0.9 * tot) { r90 = d; break; } }
  return { outside: tot ? out / tot : 0, r90 };
}

function buildConvWide() {
  const st = { key: "ns", at: { ns: [22, 40], darcy: [30, 30] }, pending: null };
  const inCv = el("canvas"), inOv = el("canvas", { class: "cv-ov" });
  const dCv = el("canvas"), dOv = el("canvas", { class: "cv-ov" });
  const status = el("p", { class: "cap-s cv-status" });
  const vOut = el("span", { class: "v" }), vR90 = el("span", { class: "v" });
  const inFig = el("div", { class: "cv-field cv-click", title: "Click to move the poke" }, inCv, inOv);
  inFig.addEventListener("click", (e) => {
    const b = inFig.getBoundingClientRect();
    st.at[st.key] = [Math.min(R64 - 1, Math.floor(((e.clientY - b.top) / b.height) * R64)), Math.min(R64 - 1, Math.floor(((e.clientX - b.left) / b.width) * R64))];
    p.draw();
  });
  const p = {
    root: el("div", { class: "panel-body cv" },
      el("h4", { class: "drill-sub" }, "Poke one spot of the input, see where this model's prediction moves"),
      el("div", { class: "cv-ctl" }, seg([["ns", "fluid"], ["darcy", "rock"]], () => st.key, (v) => { st.key = v; p.draw(); })),
      el("div", { class: "cv-wide" },
        el("figure", { class: "cv-fig" }, inFig, el("figcaption", { class: "cap-s" }, "input, poked at the marker · click to move it")),
        el("figure", { class: "cv-fig" }, el("div", { class: "cv-field" }, dCv, dOv), el("figcaption", { class: "cap-s" }, "|Δ prediction|, square-root scale"))),
      status,
      el("div", { class: "statgrid" }, stat("squared change outside the dashed square", vOut), stat("90% of the squared change sits within", vR90)),
      el("p", { class: "cap-s" }, "Measured on this model, for this poke. The dashed square is ±4 pixels: four layers of 3×3 windows can carry a change one pixel per layer, so a network built only from them could not move anything outside it. Its steps before and after the four layers look at one pixel at a time, so they add no reach.")),
    show() { p.draw(); },
    draw() {
      const app = appRef, key = st.key, at = st.at[key];
      const ds = app.ds[key], sample = app.sampleOf(key), model = ds.model, m = ds.manifest;
      const input = sample.io[64].input, idx = sample.entry.test_index;
      const pk = poke(key, input, at);
      if (key === "ns") paintField(inCv, pk, R64, { vmax: maxAbs(pk) }); else paintBinary(inCv, pk, R64);
      drawMarks(inOv, key, at, false);
      const id = `${key}|${idx}|${at.join(",")}`, hit = wideCache.get(id);
      if (hit) {
        status.textContent = `${key === "ns" ? "fluid" : "rock"} · test sample #${idx} · poke at row ${at[0]}, column ${at[1]}`;
        paintHeat(dCv, hit.delta);
        drawMarks(dOv, key, at, true);
        vOut.textContent = pct(hit.outside, 1);
        vR90.textContent = `±${hit.r90} pixels`;
        p.check = { ...hit, delta: undefined, id };
        return;
      }
      vOut.textContent = "—"; vR90.textContent = "—";
      dCv.getContext("2d").clearRect(0, 0, dCv.width, dCv.height);
      drawMarks(dOv, key, at, true);
      if (!model) { status.textContent = "loading the model…"; return; }
      status.textContent = "running the model…";
      if (st.pending === id) return;
      st.pending = id;
      setTimeout(() => {
        const norm = { yMean: m.model.y_mean, yStd: m.model.y_std };
        const baseId = `${key}|${idx}`;
        if (!wideCache.has(baseId)) wideCache.set(baseId, forward(model, Float64Array.from(input), R64, norm).prediction);
        const p1 = forward(model, pk, R64, norm).prediction, p0 = wideCache.get(baseId);
        const delta = Float64Array.from(p1, (v, i) => v - p0[i]);
        wideCache.set(id, { delta, ...measure(key, delta, at) });
        st.pending = null;
        if (p.root.isConnected) p.draw();
      }, 30);
    },
  };
  return p;
}

// Always white to orange, in both themes, like the input map beside it. Fading from the dark-mode
// background instead turns the middle of the scale a muddy brown.
const HEAT_LO = [255, 255, 255], HEAT_HI = [234, 88, 12];
function paintHeat(cv, delta) {
  if (cv.width !== R64) { cv.width = R64; cv.height = R64; }
  const ctx = cv.getContext("2d"), img = ctx.createImageData(R64, R64);
  const a = HEAT_LO, b = HEAT_HI, top = Math.sqrt(maxAbs(delta)) || 1;
  for (let i = 0; i < R64 * R64; i++) {
    const t = Math.sqrt(Math.abs(delta[i])) / top;
    for (let ch = 0; ch < 3; ch++) img.data[i * 4 + ch] = a[ch] + (b[ch] - a[ch]) * t;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function drawMarks(cv, key, [r0, c0], square) {
  const side = cv.clientWidth, f = fit(cv, side);
  if (!f) return;
  const { ctx } = f, T = tokens(), px = side / R64;
  const cx = (c0 + 0.5) * px, cy = (r0 + 0.5) * px;
  ctx.lineWidth = 1.5;
  if (square) {
    // wrapped copies for the fluid, so the square reads right when the poke sits by an edge
    const shifts = key === "ns" ? [-1, 0, 1] : [0];
    // fixed dark ink: the map under it is white to orange in both themes
    ctx.strokeStyle = "#27272a"; ctx.setLineDash([5, 4]);
    for (const a of shifts) for (const b of shifts) ctx.strokeRect((c0 - REACH + b * R64) * px, (r0 - REACH + a * R64) * px, (2 * REACH + 1) * px, (2 * REACH + 1) * px);
    ctx.setLineDash([]);
    const ly = (r0 - REACH) * px - 6 > 12 ? (r0 - REACH) * px - 6 : (r0 + REACH + 1) * px + 14;
    ctx.font = FONT(10.5, 500); ctx.textAlign = cx > side / 2 ? "right" : "left";
    const tx = cx > side / 2 ? Math.min(side - 4, (c0 + REACH + 1) * px) : Math.max(4, (c0 - REACH) * px);
    const t = "all that four 3×3 layers could reach", wt = ctx.measureText(t).width;
    ctx.fillStyle = "#ffffff"; ctx.globalAlpha = 0.85;
    ctx.fillRect(ctx.textAlign === "right" ? tx - wt - 3 : tx - 3, ly - 11, wt + 6, 15); ctx.globalAlpha = 1;
    ctx.fillStyle = "#27272a"; ctx.fillText(t, tx, ly);
  } else {
    // wide enough to ring the flipped 3×3 block without covering it
    const rad = Math.max(8, 2.6 * px);
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx, cy, rad, 0, TAU); ctx.stroke();
    ctx.strokeStyle = T.sp; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, rad, 0, TAU); ctx.stroke();
  }
}

// ---------- stage entry points ----------
export function convWaveStage(viz, fresh, id) { const p = panel("convwave", buildConvWave); viz.append(p.root); p.show(fresh, WAVE_PARTS.indexOf(id)); }
export function convSumStage(viz, fresh, id) { const p = panel("convsum", buildConvSum); viz.append(p.root); p.show(fresh, SUM_PARTS.indexOf(id)); }
export function convCostStage(viz, fresh) { const p = panel("convcost", buildConvCost); viz.append(p.root); p.show(fresh); }
export function convWideStage(viz, app) { appRef = app; const p = panel("convwide", buildConvWide); viz.append(p.root); p.show(); }
/** For the verification harness. */
export const convPanels = panels;
