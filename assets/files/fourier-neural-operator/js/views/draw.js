// Draw your own input on a 64x64 grid and re-run the model as you paint.
// Darcy: two rock materials. Navier-Stokes: a signed forcing pattern, painted in soft strokes.
import { el, paintInput, isNS, maxAbs } from "../render.js";

let brush = 3;
let material = 1;
let spin = 1; // NS: sign of the forcing painted (red +, blue −)

// Stroke strength per stamp, in the data's own units: a fraction of this sample's peak forcing,
// so a few strokes reach the range the model saw in training.
const SWIRL_FRACTION = 0.35;

export function renderDraw(app) {
  const N = 64;
  const ns = isNS();
  const cv = el("canvas", { class: "paint", width: N, height: N, "aria-label": ns ? "Forcing map. Drag to paint." : "Permeability map. Drag to paint." });
  // Fixed colour scale while painting, so a stroke visibly adds rather than the map renormalising.
  const peak = () => Math.max(maxAbs(app.sample.io[64].input), 1e-6);
  const paint = () => (ns ? paintInput(cv, app.drawing, N, { vmax: peak() }) : paintInput(cv, app.drawing, N));
  paint();
  const status = el("span", { class: "mono draw-status" }, app.sample.live ? `model ran in ${Math.round(app.sample.live.ms)} ms at ${app.sample.live.res}×${app.sample.live.res}` : "");

  let painting = false, pending = false, last = 0;
  const stamp = (e) => {
    const r = cv.getBoundingClientRect();
    const cx = ((e.clientX - r.left) / r.width) * N;
    const cy = ((e.clientY - r.top) / r.height) * N;
    if (ns) {
      // A Gaussian bump of forcing, wrapped around the edges: the domain is periodic.
      const sigma = brush * 1.4, amp = spin * SWIRL_FRACTION * peak() * 0.25, cap = 1.6 * peak();
      const R = Math.ceil(3 * sigma);
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
        const x = ((Math.floor(cx) + dx) % N + N) % N, y = ((Math.floor(cy) + dy) % N + N) % N;
        const g = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        const i = y * N + x;
        app.drawing[i] = Math.max(-cap, Math.min(cap, app.drawing[i] + amp * g));
      }
    } else {
      const rad = brush - 1;
      const ix = Math.floor(cx), iy = Math.floor(cy);
      for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
        if (dx * dx + dy * dy > rad * rad + rad) continue;
        const x = ix + dx, y = iy + dy;
        if (x >= 0 && y >= 0 && x < N && y < N) app.drawing[y * N + x] = material;
      }
    }
    paint();
  };
  // Re-run at most every 250 ms while painting, and once more on release.
  const rerun = () => {
    const now = performance.now();
    if (now - last < 250) { if (!pending) { pending = true; setTimeout(() => { pending = false; rerun(); }, 260); } return; }
    last = now;
    app.set({ redraw: true });
  };
  cv.addEventListener("pointerdown", (e) => { painting = true; cv.setPointerCapture(e.pointerId); stamp(e); });
  cv.addEventListener("pointermove", (e) => { if (painting) { stamp(e); rerun(); } });
  cv.addEventListener("pointerup", () => { painting = false; last = 0; app.set({ redraw: true }); });

  const seg = (label, options, get, setv) => el("div", { class: "seg", role: "group", "aria-label": label },
    options.map(([v, text]) => {
      const b = el("button", { "aria-pressed": String(get() === v) }, text);
      b.addEventListener("click", () => { setv(v); app.render(); });
      return b;
    }));

  const action = (text, fn) => { const b = el("button", { class: "btn" }, text); b.addEventListener("click", fn); return b; };

  const title = ns ? "Paint a forcing pattern" : "Draw a permeability map";
  const help = ns
    ? "Drag to push the fluid. Red and blue push in opposite directions. The model re-runs as you paint and every column below updates. No forcing like this was in the training data."
    : "Drag to paint. The model re-runs as you draw, and every column below updates. This map was never in the training data.";
  return el("section", { class: "draw-panel", "aria-label": title },
    el("div", { class: "paint-wrap" }, cv),
    el("div", { class: "draw-tools" },
      el("h3", { class: "draw-title" }, title),
      el("p", { class: "draw-help" }, help),
      ns
        ? el("div", { class: "tool-row" }, el("span", { class: "tool-lab" }, "Push"),
          seg("Push", [[1, "red (+)"], [-1, "blue (−)"]], () => spin, (v) => { spin = v; }))
        : el("div", { class: "tool-row" }, el("span", { class: "tool-lab" }, "Material"),
          seg("Material", [[1, "dark rock"], [0, "light rock"]], () => material, (v) => { material = v; })),
      el("div", { class: "tool-row" }, el("span", { class: "tool-lab" }, "Brush"),
        ns
          ? seg("Brush size", [[2, "small"], [3, "medium"], [5, "large"]], () => brush, (v) => { brush = v; })
          : seg("Brush size", [[1, "1 px"], [3, "5 px"], [6, "11 px"]], () => brush, (v) => { brush = v; })),
      el("div", { class: "tool-row" },
        action("Start from sample", () => { app.drawing = Float64Array.from(app.sample.io[64].input); app.set({ redraw: true }); }),
        action(ns ? "Flip the sign" : "Invert", () => { app.drawing = app.drawing.map((v) => (ns ? -v : 1 - v)); app.set({ redraw: true }); }),
        action(ns ? "No forcing" : "Clear", () => { app.drawing = new Float64Array(N * N); app.set({ redraw: true }); }),
        action("Back to test samples", () => app.set({ draw: false }))),
      status));
}
