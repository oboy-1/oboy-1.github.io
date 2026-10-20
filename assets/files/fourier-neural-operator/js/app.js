// App shell: story state, the scroll driver, live inference, and the reader's own controls.
// Scrolling sets the model state; clicking refines it. A reader who only scrolls sees every step.
import { loadManifest, loadSample, loadWeights } from "./data.js";
import { setColormaps, setProblem, el, paintInput, isNS } from "./render.js";
import { fft2, ifft2 } from "./fourier.js";
import { renderStage } from "./views/stage.js";
import { forward, layerStats, spectra, channelCorrelations } from "./fno.js";
import { STEPS, notes } from "./story.js";

// Keys the reader owns. A step may propose a value, but never overrides a deliberate click.
const STICKY = new Set(["sampleIdx", "criterion", "showAll", "res", "layer", "channel"]);

// Two datasets, one page. Steps own `dataset`; everything dataset-specific (manifest, model,
// sample cache, chosen sample, drawing) is swapped in and out of the app object on a switch.
const DATASETS = { ns: "data-ns", darcy: "data" };
const PER_DATASET = ["manifest", "model", "sample", "cache", "drawing"];

const app = {
  manifest: null,
  model: null,
  sample: null,
  cache: new Map(),
  drawing: null, // Float64Array(64*64) of 0/1 once the reader starts drawing
  touched: new Set(),
  stepIndex: 0,
  lastStep: -1,
  ds: { ns: {}, darcy: {} },
  state: {
    dataset: "ns", sampleIdx: 0, criterion: "corr_target", showAll: false, res: 64,
    view: "flow", flowMode: "plain", layer: 2, channel: null, node: null, focus: null, draw: false,
  },

  /** A dataset's current sample, whether it is the active one or parked. */
  sampleOf(key) { return key === this.state.dataset ? this.sample : this.ds[key].sample; },

  /** Park the active dataset's objects and bring another one in. */
  switchDataset(next) {
    const cur = this.ds[this.state.dataset];
    for (const k of PER_DATASET) cur[k] = this[k];
    cur.sampleIdx = this.state.sampleIdx;
    const nx = this.ds[next];
    for (const k of PER_DATASET) this[k] = nx[k];
    this.state.sampleIdx = nx.sampleIdx;
    setProblem(this.manifest.problem);
  },

  step() { return STEPS[this.stepIndex]; },

  /** Reader-driven change. Marks the keys as theirs so later steps leave them alone. */
  async set(patch) {
    for (const k of Object.keys(patch)) if (STICKY.has(k)) this.touched.add(k);
    await this.apply(patch);
  },

  /** Story-driven change: sticky keys apply only while the reader hasn't set them. */
  async applyStep(i) {
    this.stepIndex = i;
    const patch = {};
    for (const [k, v] of Object.entries(STEPS[i].state)) {
      if (STICKY.has(k) && this.touched.has(k)) continue;
      patch[k] = v;
    }
    await this.apply(patch);
  },

  async apply(patch) {
    if (patch.dataset && patch.dataset !== this.state.dataset) {
      const { dataset, ...rest } = patch;
      this.switchDataset(dataset);
      this.state = { ...this.state, dataset };
      patch = rest;
    }
    const prev = this.state;
    this.state = { ...prev, ...patch };
    const s = this.state;
    if (patch.sampleIdx != null && patch.sampleIdx !== this.sample?.entry.test_index) await this.selectSample(patch.sampleIdx);
    if (s.draw && !this.drawing) this.drawing = Float64Array.from(this.sample.io[64].input);
    // Rerun when anything the live pass depends on differs from the last run of *this* dataset.
    const L = this.sample.live;
    const stale = !L || L.res !== s.res || L.draw !== s.draw || patch.redraw;
    if (this.model && stale) this.runLive();
    this.render();
  },

  /** Jump the story to a named step, optionally with the reader's own choices attached. */
  goToStep(id, patch = {}) {
    const i = STEPS.findIndex((s) => s.id === id);
    if (i < 0) return;
    if (Object.keys(patch).length) this.set(patch);
    const sec = document.getElementById(`step-${STEPS[i].id}`);
    if (sec) sec.scrollIntoView({ behavior: motionOK() ? "smooth" : "auto", block: "start" });
    else this.applyStep(i);
  },

  async selectSample(idx) {
    if (!this.cache.has(idx)) this.cache.set(idx, await loadSample(this.manifest, idx));
    const s = this.cache.get(idx);
    this.sample = { ...s, error: Object.fromEntries(Object.entries(s.entry.error).map(([r, e]) => [r, e.rel_l2])), live: null };
  },

  /** Run the JS forward pass for the current input and grid size. */
  runLive() {
    const { res, draw } = this.state;
    const s = this.sample;
    let input, target = null;
    if (draw && this.drawing) {
      input = isNS() ? resampleSmooth(this.drawing, 64, res) : resampleNearest(this.drawing, 64, res);
    } else {
      input = Float64Array.from(s.io[res].input);
      target = s.io[res].target;
    }
    const t0 = performance.now();
    const run = forward(this.model, input, res, { yMean: this.manifest.model.y_mean, yStd: this.manifest.model.y_std });
    const corr = channelCorrelations(run, input, target);
    let err = null;
    if (target) {
      let a = 0, b = 0;
      for (let i = 0; i < res * res; i++) { a += (run.prediction[i] - target[i]) ** 2; b += target[i] ** 2; }
      err = Math.sqrt(a / b);
    }
    let statsMemo = null, spectraMemo = null;
    s.live = {
      ...run, input, target, r_in: corr.r_in, r_u: corr.r_u, error: err, ms: performance.now() - t0, draw: !!draw,
      get layerStats() { return (statsMemo ??= layerStats(run)); },
      get spectra() { return (spectraMemo ??= spectra(run)); },
    };
  },

  hoverChannel(c, evt, text) {
    document.querySelectorAll(".tile.hl").forEach((t) => t.classList.remove("hl"));
    const tip = document.getElementById("tip");
    if (c == null) { tip.hidden = true; return; }
    document.querySelectorAll(`.tile[data-ch="${c}"]`).forEach((t) => t.classList.add("hl"));
    tip.innerHTML = text();
    tip.hidden = false;
    this.moveTip(evt);
  },

  moveTip(evt) {
    const tip = document.getElementById("tip");
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  },

  render() {
    const s = this.sample;
    s.res = this.state.res;
    s.fieldRes = s.live ? s.live.res : 64;
    s.fieldNote = !s.live && this.state.res !== 64
      ? `Loading the model to recompute hidden channels at ${this.state.res}×${this.state.res}. Until then they show the precomputed 64×64 fields.`
      : null;
    renderControls();
    const stage = document.getElementById("stage");
    renderStage(stage, this);
    if (this.stepIndex !== this.lastStep) {
      this.lastStep = this.stepIndex;
      if (motionOK()) { stage.classList.remove("swap"); void stage.offsetWidth; stage.classList.add("swap"); }
    }
    renderProse(this.stepIndex);
    renderDots(this.stepIndex);
    document.getElementById("tip").hidden = true;
  },
};

const motionOK = () => !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function resampleNearest(src, from, to) {
  const out = new Float64Array(to * to);
  for (let i = 0; i < to; i++) for (let j = 0; j < to; j++) {
    out[i * to + j] = src[Math.floor((i * from) / to) * from + Math.floor((j * from) / to)];
  }
  return out;
}

/**
 * Resample a smooth periodic field (a painted forcing). Down: point subsample, exactly how the
 * 32 and 64 test grids were made from 128. Up: Fourier interpolation, which adds no frequencies
 * the painting didn't have; nearest-neighbor would add staircase edges, i.e. fake fine detail.
 */
export function resampleSmooth(src, from, to) {
  if (to === from) return Float64Array.from(src);
  if (to < from) {
    const step = from / to, out = new Float64Array(to * to);
    for (let i = 0; i < to; i++) for (let j = 0; j < to; j++) out[i * to + j] = src[i * step * from + j * step];
    return out;
  }
  const F = fft2(src, from);
  const N = to * to, h = from / 2, scale = (to * to) / (from * from);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let x = 0; x < from; x++) {
    const kx = x < h ? x : x - from;
    if (Math.abs(kx) === h) continue; // drop the Nyquist row/column rather than guess its split
    for (let y = 0; y < from; y++) {
      const ky = y < h ? y : y - from;
      if (Math.abs(ky) === h) continue;
      const d = ((kx + to) % to) * to + ((ky + to) % to);
      re[d] = F.re[x * from + y] * scale; im[d] = F.im[x * from + y] * scale;
    }
  }
  return ifft2({ re, im }, to).re;
}

// ---------- reader controls, pinned with the viz ----------
function renderControls() {
  const nav = document.getElementById("samples");
  nav.replaceChildren();
  for (const s of app.manifest.samples) {
    const cv = el("canvas", { width: 64, height: 64 });
    const cached = app.cache.get(s.test_index);
    const m = app.manifest, cache = app.cache;
    if (cached) paintInput(cv, cached.io[64].input, 64);
    else loadSample(m, s.test_index).then((d) => {
      cache.set(s.test_index, d);
      // Paint with the dataset this thumbnail belongs to, even if the page has switched since.
      if (app.manifest === m) paintInput(cv, d.io[64].input, 64);
    });
    const label = s.note ? el("span", { class: "flag" }, `#${s.test_index} worst`) : el("span", {}, `#${s.test_index}`);
    const pressed = !app.state.draw && app.state.sampleIdx === s.test_index;
    const b = el("button", { class: "sample-btn", "aria-pressed": String(pressed), title: `Test sample ${s.test_index}, error ${(s.error["64"].rel_l2 * 100).toFixed(1)}% at 64×64${s.note ? ` (${s.note})` : ""}` }, cv, label);
    b.addEventListener("click", () => app.set({ sampleIdx: s.test_index, draw: false }));
    nav.append(b);
  }
  const dcv = el("canvas", { width: 64, height: 64 });
  if (app.drawing) paintInput(dcv, app.drawing, 64);
  const drawBtn = el("button", { class: "sample-btn draw-btn", "aria-pressed": String(app.state.draw), disabled: !app.model, title: app.model ? (isNS() ? "Paint your own forcing" : "Draw your own permeability map") : "Loading the model…" },
    app.drawing ? dcv : el("span", { class: "pencil", "aria-hidden": "true" }, "✎"), el("span", {}, "draw"));
  drawBtn.addEventListener("click", () => app.goToStep("draw", { draw: true }));
  nav.append(drawBtn);

  const badge = document.getElementById("dataset-badge");
  if (badge) {
    badge.textContent = isNS() ? "Navier–Stokes · forcing → swirl" : "Darcy · rock → pressure";
    badge.dataset.ds = app.state.dataset;
  }

  const box = document.getElementById("res");
  box.replaceChildren(el("span", { class: "res-label" }, "grid"));
  for (const r of app.manifest.resolutions) {
    const b = el("button", { "aria-pressed": String(app.state.res === r), title: `Run the same weights on a ${r}×${r} grid` }, `${r}`);
    b.addEventListener("click", () => app.set({ res: r }));
    box.append(b);
  }
}

// ---------- the story column ----------
function buildStory() {
  const story = document.getElementById("story");
  story.replaceChildren();
  placeProgress(); // rides with the article on desktop, with the pinned panel when narrow
  STEPS.forEach((st, i) => {
    story.append(el("section", { class: "step", id: `step-${st.id}`, "data-i": String(i) },
      el("div", { class: "step-inner" },
        el("span", { class: "eyebrow" }, `${String(i + 1).padStart(2, "0")} · ${st.eyebrow}`),
        el("h2", {}, st.title),
        el("div", { class: "step-prose" }))));
  });
}

/** Keep the progress rail out of the prose's way: in the article when wide, in the panel when narrow. */
const NARROW = window.matchMedia("(max-width: 1080px)");
function placeProgress() {
  const nav = document.querySelector(".progress");
  const home = NARROW.matches ? document.querySelector(".viz-bar") : document.getElementById("story");
  if (nav.parentElement === home) return;
  if (NARROW.matches) home.append(nav); else home.prepend(nav);
}
NARROW.addEventListener("change", placeProgress);

function renderProse(i) {
  const host = document.querySelector(`#step-${STEPS[i].id} .step-prose`);
  if (host) host.replaceChildren(...STEPS[i].prose(app).filter(Boolean));
}

function renderDots(active) {
  const rail = document.getElementById("dots");
  if (rail.children.length !== STEPS.length) {
    rail.replaceChildren(...STEPS.map((st, i) => {
      const b = el("button", { class: "dot", title: `${i + 1}. ${st.title}`, "aria-label": `Step ${i + 1}: ${st.title}` });
      b.addEventListener("click", () => app.goToStep(st.id));
      return b;
    }));
  }
  [...rail.children].forEach((d, i) => d.setAttribute("aria-current", String(i === active)));
  document.getElementById("progress-now").textContent = `${active + 1} / ${STEPS.length}`;
}

/** The driver: the active step is the last one whose top has crossed the trigger line. */
function watchScroll() {
  const secs = STEPS.map((st) => document.getElementById(`step-${st.id}`));
  let queued = false;
  const pick = () => {
    queued = false;
    const line = innerHeight * 0.45;
    let i = 0;
    for (let k = 0; k < secs.length; k++) if (secs[k].getBoundingClientRect().top <= line) i = k;
    secs.forEach((s, k) => s.classList.toggle("on", k === i));
    if (i !== app.stepIndex) app.applyStep(i);
  };
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(pick); } }, { passive: true });
  addEventListener("resize", () => { if (!queued) { queued = true; requestAnimationFrame(pick); } });
  pick();
}

async function boot() {
  const stage = document.getElementById("stage");
  stage.replaceChildren(el("p", { class: "loading" }, "Loading model data…"));
  try {
    // Load both datasets' manifests and first samples; the active one is whatever step 1 owns.
    for (const [key, dir] of Object.entries(DATASETS)) {
      const m = await loadManifest(dir);
      app.manifest = m; app.cache = new Map(); app.drawing = null; app.model = null;
      setProblem(m.problem);
      await app.selectSample(m.samples[0].test_index);
      app.ds[key] = { manifest: m, model: null, sample: app.sample, cache: app.cache, drawing: null, sampleIdx: m.samples[0].test_index };
    }
    for (const k of PER_DATASET) app[k] = app.ds[app.state.dataset][k];
    app.state.sampleIdx = app.ds[app.state.dataset].sampleIdx;
    setProblem(app.manifest.problem);
    setColormaps(app.manifest.colormaps);
    buildStory();
    await app.applyStep(0); // the first step owns the opening state, not the defaults
    watchScroll();
    window.__fno = app; // for the verification harness
    document.documentElement.dataset.ready = "1";
    // Weights for both models; the parked dataset gets its model when it is next swapped in.
    for (const key of Object.keys(DATASETS)) {
      const w = await loadWeights(app.ds[key].manifest);
      if (key === app.state.dataset) app.model = w; else app.ds[key].model = w;
      app.ds[key].model = w;
    }
    app.runLive();
    app.render();
    document.documentElement.dataset.live = "1";
    document.getElementById("notes").replaceChildren(...notes(app));
  } catch (e) {
    stage.replaceChildren(el("p", { class: "error" }, `Could not load the model data: ${e.message}. Serve the web/ folder over HTTP (for example: python3 -m http.server).`));
    console.error(e);
  }
}

export default app;
boot();
