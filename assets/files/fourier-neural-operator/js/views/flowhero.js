// Intro animation: dye carried by the flow the model predicts for a stirred fluid.
// The model returns vorticity at one time (T = 5). From that field we get the velocity with a stream function
// (omega = -laplacian(psi), u = (dpsi/dy, -dpsi/dx)) and let tracer particles drift along it. It is a snapshot of
// the flow, so the motion shows how that snapshot carries fluid, not how it evolved to get there.
import { paintField } from "../render.js";
import { fft2, ifft2 } from "../fourier.js";

const R = 64;
let state = null; // { key, vi, vj, bg }
let host = null, raf = 0, parts = [], ctxT = null, cvT = null, cvB = null, last = 0, visible = true, tint = "#c2410c";

function velocity(omega) {
  const F = fft2(omega, R);
  const re = new Float64Array(R * R), im = new Float64Array(R * R); // d psi / d(dim0), d psi / d(dim1)
  const re1 = new Float64Array(R * R), im1 = new Float64Array(R * R);
  for (let a = 0; a < R; a++) for (let b = 0; b < R; b++) {
    const ka = a < R / 2 ? a : a - R, kb = b < R / 2 ? b : b - R, k2 = ka * ka + kb * kb, i = a * R + b;
    if (!k2) continue;
    const pr = F.re[i] / k2, pi = F.im[i] / k2;           // psi_hat = omega_hat / k^2
    re[i] = -ka * pi; im[i] = ka * pr;                      // i*ka * psi_hat
    re1[i] = -kb * pi; im1[i] = kb * pr;                    // i*kb * psi_hat
  }
  const dA = ifft2({ re, im }, R).re, dB = ifft2({ re: re1, im: im1 }, R).re;
  // dim0 = x, dim1 = y: u_x = dpsi/dy = dB, u_y = -dpsi/dx = -dA. Canvas: x along dim1, y along dim0.
  const vj = new Float64Array(R * R), vi = new Float64Array(R * R);
  let ms = 0;
  for (let i = 0; i < R * R; i++) { vj[i] = -dA[i]; vi[i] = dB[i]; ms += vj[i] * vj[i] + vi[i] * vi[i]; }
  const s = 1 / Math.sqrt(ms / (R * R) || 1);
  for (let i = 0; i < R * R; i++) { vj[i] *= s; vi[i] *= s; }
  return { vi, vj };
}

function sample(f, y, x) { // bilinear, periodic; y = dim0, x = dim1
  const y0 = Math.floor(y), x0 = Math.floor(x), fy = y - y0, fx = x - x0;
  const a = ((y0 % R) + R) % R, b = ((y0 + 1) % R + R) % R, c = ((x0 % R) + R) % R, d = ((x0 + 1) % R + R) % R;
  return (f[a * R + c] * (1 - fx) + f[a * R + d] * fx) * (1 - fy) + (f[b * R + c] * (1 - fx) + f[b * R + d] * fx) * fy;
}

function layout() {
  const dpr = Math.min(2, window.devicePixelRatio || 1), w = host.clientWidth, h = host.clientHeight;
  for (const c of [cvB, cvT]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  ctxT = cvT.getContext("2d");
  const k = cvB.width / R; // grid cell -> device px; the field is drawn "cover", cropped vertically
  state.k = k; state.band = cvB.height / k; state.y0 = (R - state.band) / 2;
  const g = cvB.getContext("2d");
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.clearRect(0, 0, cvB.width, cvB.height);
  g.globalAlpha = 0.5; g.drawImage(state.bg, 0, -state.y0 * k, cvB.width, R * k); g.globalAlpha = 1;
  parts = Array.from({ length: Math.round(w * h / 230) }, () => spawn(true));
}
function spawn(first) {
  return { x: Math.random() * R, y: state.y0 + Math.random() * state.band, life: (first ? Math.random() : 1) * (90 + Math.random() * 140) };
}

function frame(now) {
  raf = requestAnimationFrame(frame);
  if (!visible || !state || document.hidden) { last = now; return; }
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  const k = state.k, speed = 3.2; // grid cells per second at rms speed
  ctxT.globalCompositeOperation = "destination-out"; ctxT.fillStyle = "rgba(0,0,0,0.11)"; ctxT.fillRect(0, 0, cvT.width, cvT.height);
  ctxT.globalCompositeOperation = "source-over"; ctxT.strokeStyle = tint; ctxT.lineWidth = 1.15 * Math.max(1, cvT.width / 700);
  ctxT.lineCap = "round"; ctxT.beginPath();
  for (let n = 0; n < parts.length; n++) {
    const p = parts[n];
    const u = sample(state.vj, p.y, p.x), v = sample(state.vi, p.y, p.x);
    const nx = p.x + u * speed * dt, ny = p.y + v * speed * dt;
    ctxT.moveTo(p.x * k, (p.y - state.y0) * k); ctxT.lineTo(nx * k, (ny - state.y0) * k);
    p.x = ((nx % R) + R) % R; p.y = ny; p.life -= 1;
    if (p.life <= 0 || p.y < state.y0 - 1 || p.y > state.y0 + state.band + 1) parts[n] = spawn(false);
  }
  ctxT.stroke();
}

export function mountFlowHero(app, el) {
  const s = app.sampleOf?.("ns");
  const pred = s?.io?.[64]?.prediction;
  if (!pred || !el) return;
  const key = `${s.entry?.index ?? app.ds?.ns?.sampleIdx ?? 0}`;
  if (state?.key === key && host === el) return;
  host = el;
  const bg = document.createElement("canvas"); paintField(bg, pred, R);
  state = { key, ...velocity(pred), bg };
  if (!cvB) {
    cvB = document.createElement("canvas"); cvT = document.createElement("canvas");
    el.prepend(cvB, cvT);
    new ResizeObserver(() => state && layout()).observe(el);
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(el);
    const re = () => { tint = getComputedStyle(document.documentElement).getPropertyValue("--accent-text").trim() || tint; if (state) layout(); };
    new MutationObserver(re).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", re);
    tint = getComputedStyle(document.documentElement).getPropertyValue("--accent-text").trim() || tint;
  }
  layout();
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { for (let i = 0; i < 60; i++) frame(i * 16); cancelAnimationFrame(raf); return; }
  if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); }
}
