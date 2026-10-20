// "Why an edge needs high frequencies": adapted from the standalone truncation widget
// (claude.ai/artifact/87JkV7yjwqf6mnXGViCKv6). Same math and verdict logic; restyled with the
// explainer's tokens, and the cutoff this model uses is marked.
import { el } from "../render.js";

const NP = 640;
const xs = Array.from({ length: NP }, (_, i) => i / NP);
const SVG = "http://www.w3.org/2000/svg";

function target(kind) {
  if (kind === "edge") return xs.map((x) => (x >= 0.25 && x < 0.75 ? 1 : 0));
  return xs.map((x) => {
    const d = Math.min(Math.abs(x - 0.5), 1 - Math.abs(x - 0.5));
    return Math.exp(-(d * d) / (2 * 0.075 * 0.075));
  });
}

function coeffs(f, kmax) {
  const a = [], b = [];
  for (let k = 0; k <= kmax; k++) {
    let ar = 0, br = 0;
    for (let n = 0; n < NP; n++) {
      const ang = (2 * Math.PI * k * n) / NP;
      ar += f[n] * Math.cos(ang); br += f[n] * Math.sin(ang);
    }
    if (k === 0) { a.push(ar / NP); b.push(0); } else { a.push((2 * ar) / NP); b.push((2 * br) / NP); }
  }
  return { a, b };
}

function rebuild({ a, b }, K) {
  return xs.map((x) => {
    let v = a[0];
    for (let k = 1; k <= K; k++) { const ang = 2 * Math.PI * k * x; v += a[k] * Math.cos(ang) + b[k] * Math.sin(ang); }
    return v;
  });
}

function svgEl(tag, attrs) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

export function mountTruncationWidget(host, { cutoff = 8, n = 64 } = {}) {
  const KMAX = n / 2;
  const DATA = {};
  for (const t of ["edge", "bump"]) {
    const f = target(t);
    const c = coeffs(f, KMAX);
    const energy = [];
    for (let k = 1; k <= KMAX; k++) energy.push(c.a[k] ** 2 + c.b[k] ** 2);
    DATA[t] = { f, c, energy, total: energy.reduce((s, v) => s + v, 0) };
  }
  let kind = "edge";
  let K = cutoff;

  const segBtns = ["edge", "bump"].map((t) => {
    const b = el("button", { "aria-pressed": String(t === kind) }, t === "edge" ? "Sharp edge" : "Smooth bump");
    b.addEventListener("click", () => { kind = t; segBtns.forEach((x, i) => x.setAttribute("aria-pressed", String(["edge", "bump"][i] === kind))); update(); });
    return b;
  });
  const main = svgEl("svg", { viewBox: "0 0 600 180", role: "img", "aria-label": "Shape rebuilt from waves" });
  const bars = svgEl("svg", { viewBox: "0 0 600 86", role: "img", "aria-label": "Energy in each wave" });
  const slider = el("input", { type: "range", id: "trunc-k", min: "0", max: String(KMAX), step: "1", value: String(K), "aria-label": "Waves used" });
  const kLab = el("b", { class: "mono" }, String(K));
  const sKept = el("span", { class: "v" }), sEnergy = el("span", { class: "v" }), sErr = el("span", { class: "v" });
  const verdict = el("p", { class: "verdict" });

  host.replaceChildren(
    el("div", { class: "tw" },
      el("div", { class: "seg" }, ...segBtns),
      main,
      el("label", { class: "tw-slider", for: "trunc-k" }, el("span", {}, "waves used, 0 through k"), kLab),
      slider,
      el("div", { class: "statgrid three" },
        el("div", { class: "stat" }, el("span", { class: "k" }, "waves kept"), sKept),
        el("div", { class: "stat" }, el("span", { class: "k" }, "energy captured"), sEnergy),
        el("div", { class: "stat" }, el("span", { class: "k" }, "worst error"), sErr)),
      bars,
      verdict));
  slider.addEventListener("input", () => { K = +slider.value; update(); });

  // Read tokens from the root: the host may not be attached yet on first draw.
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  function update() {
    kLab.textContent = K;
    const D = DATA[kind];
    const rec = rebuild(D.c, K);
    const ink3 = css("--ink-3"), rule = css("--rule"), sp = css("--spectral"), lo = css("--local");

    main.replaceChildren();
    const x0 = 30, x1 = 590, yB = 150, yT = 14, vlo = -0.35, vhi = 1.35;
    const Y = (v) => yB - ((v - vlo) / (vhi - vlo)) * (yB - yT);
    for (const v of [0, 1]) {
      main.append(svgEl("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: rule, "stroke-width": 1 }));
      const t = svgEl("text", { x: x0 - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 10, fill: ink3 }); t.textContent = v; main.append(t);
    }
    const line = (vals, color, w, dash) => {
      const pts = vals.map((v, i) => `${(x0 + (i / (NP - 1)) * (x1 - x0)).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
      const p = svgEl("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": w, "stroke-linejoin": "round" });
      if (dash) p.setAttribute("stroke-dasharray", dash);
      main.append(p);
    };
    line(D.f.map((v, i) => (v - rec[i]) * 0.55 - 0.22), lo, 1.3);
    const rl = svgEl("text", { x: x1, y: Y(-0.22) + 26, "text-anchor": "end", "font-size": 10, fill: lo }); rl.textContent = "what's missing (scaled)"; main.append(rl);
    line(D.f, ink3, 1.5, "5,4");
    line(rec, sp, 2.2);

    bars.replaceChildren();
    const bx0 = 30, bx1 = 590, byB = 58, maxH = 46, nb = D.energy.length, bw = (bx1 - bx0) / nb;
    const mx = Math.max(...D.energy) || 1;
    bars.append(svgEl("line", { x1: bx0, x2: bx1, y1: byB, y2: byB, stroke: rule }));
    D.energy.forEach((e, i) => {
      const k = i + 1, h = Math.sqrt(e / mx) * maxH;
      bars.append(svgEl("rect", { x: (bx0 + i * bw + 0.8).toFixed(1), y: (byB - h).toFixed(1), width: Math.max(1, bw - 1.6).toFixed(1), height: Math.max(0.6, h).toFixed(1), rx: 1.5, fill: k <= K ? sp : rule }));
    });
    const cx = bx0 + cutoff * bw;
    bars.append(svgEl("line", { x1: cx, x2: cx, y1: 2, y2: byB + 4, stroke: lo, "stroke-width": 1.5, "stroke-dasharray": "3,3" }));
    const ct = svgEl("text", { x: cx + 5, y: 11, "font-size": 10, fill: lo, "font-weight": 600 }); ct.textContent = `this FNO keeps k ≤ ${cutoff}`; bars.append(ct);
    for (const k of [1, 8, 16, 24, 32]) {
      if (k > nb) continue;
      const t = svgEl("text", { x: (bx0 + (k - 0.5) * bw).toFixed(1), y: byB + 14, "text-anchor": "middle", "font-size": 10, fill: ink3 }); t.textContent = `k=${k}`; bars.append(t);
    }
    const note = svgEl("text", { x: bx0, y: 84, "font-size": 10, fill: ink3 }); note.textContent = "energy per wave, square-root scale so the small high-frequency bars stay visible"; bars.append(note);

    let kept = 0;
    for (let k = 1; k <= K; k++) kept += D.energy[k - 1];
    const pE = D.total > 0 ? (100 * kept) / D.total : 100;
    let worst = 0;
    for (let i = 0; i < NP; i++) worst = Math.max(worst, Math.abs(D.f[i] - rec[i]));
    sKept.textContent = `${K + 1} / ${KMAX + 1}`;
    sEnergy.textContent = `${pE.toFixed(1)}%`;
    sErr.textContent = worst.toFixed(2);

    if (kind === "bump") {
      verdict.className = "verdict";
      verdict.textContent = pE > 99.5 && worst < 0.03
        ? "Done with only a few waves. A smooth shape keeps almost all its content in the slow ones, so a low cutoff costs it almost nothing."
        : "Keep going. The energy climbs fast, because a smooth shape keeps nearly all of it in the first handful of waves.";
    } else {
      verdict.className = "verdict warn";
      verdict.textContent = pE > 90 && worst > 0.15
        ? `${pE.toFixed(0)}% of the energy captured, and the corners are still wrong. The overshoot won't go away, it only gets narrower. High energy is not high fidelity when there's an edge.`
        : worst > 0.4
          ? "A few slow waves can't form a sharp corner. The best they can do is a rounded ramp."
          : `Closer, but the overshoot at each corner is still ${worst.toFixed(2)}. An instant jump needs every wave.`;
    }
  }
  update();
}
