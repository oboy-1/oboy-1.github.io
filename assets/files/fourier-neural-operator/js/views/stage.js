// One step in, one viz out. Each branch renders real data for exactly one story state.
import { el, M, paintField, paintBinary, paintInput, outScale, isNS, words, maxAbs, fmt, pct } from "../render.js";
import { spatialStd } from "../nn.js";
import { splitAtCutoff, keptMask, fft2 } from "../fourier.js";
import { renderFlow } from "./flow.js";
import { renderLayer, layerHeader, layerFields, layerStats, fieldTile } from "./layer.js";
import { renderDraw } from "./draw.js";
import { storedStage, multiplyStage, limitsStage } from "./learned.js";
import { buildUpPanel } from "./buildup.js";
import { getArch, onThemeChange, setArchSpectrum } from "./archscene.js";
import { mountFlowHero } from "./flowhero.js";
import { mountWaveSum } from "../widgets/wavesum.js";
import { truncateRealPanel, truncateWidgetPanel, rPanel, heatTile, waveIntroPanel } from "./drill.js";
import { convWaveStage, convSumStage, convCostStage, convWideStage, WAVE_PARTS, SUM_PARTS } from "./convolution.js";

const maskCache = new Map();
const mask = (res) => {
  if (!maskCache.has(res)) maskCache.set(res, keptMask(res));
  return maskCache.get(res);
};

// The 3D scene is the map; the 2D panels are the detail. mode: "tall" = the 3D alone, "strip" = 3D above the 2D,
// "ref" = a slim live strip that keeps the camera on the part under discussion while the 2D takes over.
const ARCH = {
  overview: { v: ["all", "all"], mode: "tall" },
  lift: { v: ["lift", "lift"], mode: "strip" },
  split: { v: ["layerS", ""], mode: "strip" },
  truncate: { v: ["spec12", ""], mode: "ref", label: "Spectral path · keep the slow waves" },
  multiply: { v: ["spec23", ""], mode: "ref", label: "Spectral path · × R" },
  widget: { v: ["spec12", ""], mode: "ref", label: "Spectral path · keep the slow waves" },
  bridge: { v: ["local", ""], mode: "ref", label: "Local path" },
  pixels: { v: ["spec34", ""], mode: "ref", size: "mid", label: "Spectral path · inverse FFT" },
  paths: { v: ["layer", ""], mode: "ref", size: "mid", label: "One layer, both paths" },
  block: { v: ["block", ""], mode: "strip", label: "The rest of the layer" },
  flow: { v: ["allR", "all"], mode: "ref", label: "The whole network" },
  project: { v: ["out", ""], mode: "ref", label: "Projection Q" },
};
// Clicking a node in the 2D layer diagram flies the 3D camera to the same part.
const NODE_VIEW = { fft: "spec12", truncate: "spec12", r: "spec23", conv: "local" };

let curStep = null, userBig = false, curCfg = null, themeHooked = false;

function place3d(viz, step, app) {
  const cfg = ARCH[step.id];
  curCfg = cfg ?? null;
  if (!cfg) { curStep = step.id; return null; }
  if (!themeHooked) { themeHooked = true; onThemeChange(() => app.render()); }
  const arch = getArch(), host = arch.el;
  const changed = curStep !== step.id;
  if (changed) userBig = false;
  curStep = step.id;
  if (!host.querySelector(".arch3d-toggle")) {
    const ui = host.querySelector(".arch3d-ui");
    const t = document.createElement("button"); t.type = "button"; t.className = "arch3d-toggle";
    t.addEventListener("click", () => {
      userBig = !userBig;
      host.dataset.size = userBig ? "strip" : (curCfg?.size ?? "ref");
      t.textContent = userBig ? "Shrink ▴" : "Enlarge ▾";
    });
    const part = document.createElement("div"); part.className = "arch3d-part";
    ui.append(t, part);
  }
  const t = host.querySelector(".arch3d-toggle");
  t.hidden = cfg.mode !== "ref"; t.textContent = userBig ? "Shrink ▴" : "Enlarge ▾";
  host.querySelector(".arch3d-part").textContent = cfg.label ?? "";
  const want = cfg.mode === "ref" ? (userBig ? "strip" : (cfg.size ?? "ref")) : cfg.mode;
  if (!host.dataset.size) host.dataset.size = "strip";
  if (changed && viz.parentElement) viz.parentElement.scrollTop = 0;
  viz.append(host);
  // the size is set one frame after the element is back in the page, so the change animates
  requestAnimationFrame(() => requestAnimationFrame(() => { host.dataset.size = want; }));
  let view = cfg.v[0];
  if (step.id === "split" && NODE_VIEW[app.state.node]) view = NODE_VIEW[app.state.node];
  arch.setView(view, cfg.v[1]);
  return cfg;
}

// The 3D spectrum is the real thing: FFT of one layer-1 input channel and of what the spectral path makes of it.
let specKey = "";
function feedSpectrum(app) {
  const s = app.sample, f = s?.fields;
  if (!f) return;
  const c = (app.state.layer === 1 ? app.state.channel : null) ?? layerStats(app, 1).default_channel;
  const key = `${app.state.dataset}|${app.state.sampleIdx}|${c}`;
  if (key === specKey) return;
  const inF = f["lifted"]?.[c], outF = f["layer1.spectral"]?.[c];
  if (!inF || !outF) return;
  const R = 64, CELLS = 16, B = R / CELLS;
  const pool = (field) => {
    const { re, im } = fft2(field, R);
    const out = new Float64Array(CELLS * CELLS);
    for (let a = 0; a < CELLS; a++) for (let b = 0; b < CELLS; b++) {
      let e = 0;
      for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
        const kx = (B * (a - CELLS / 2) + i + R) % R, ky = (B * (b - CELLS / 2) + j + R) % R;
        const k = kx * R + ky; e += re[k] * re[k] + im[k] * im[k];
      }
      out[a * CELLS + b] = Math.sqrt(e);
    }
    return out;
  };
  const inAmp = pool(inF), outAmp = pool(outF);
  let vmax = 0; for (const v of inAmp) vmax = Math.max(vmax, v);
  specKey = key;
  setArchSpectrum({ inAmp, outAmp, vmax, label: `real data · layer 1 · channel ${c}` });
}

// With no 3D role on the current slide the scene goes back up into the title block, where it just turns slowly.
function parkHero() {
  const slot = document.getElementById("hero3d");
  if (!slot) return;
  const arch = getArch(), host = arch.el;
  if (host.parentElement !== slot) { host.dataset.size = "hero"; slot.append(host); }
  arch.setView("hero");
}

// The sample / grid-size bar only appears where the slide actually uses a chosen sample or grid.
const NO_BAR = new Set(["operator", "recipe", "recipe2d", "energy", "overview", "multiply", "widget", "block", "limits", "resolution", "convcost", "convwide", ...WAVE_PARTS, ...SUM_PARTS]);
// Animated panels replay when their slide is entered, not on every re-render within it.
let prevStepId = null;

export function renderStage(viz, app) {
  const step = app.step();
  const fresh = step.id !== prevStepId;
  prevStepId = step.id;
  viz.replaceChildren();
  viz.dataset.step = step.id;
  viz.parentElement?.classList.toggle("nobar", NO_BAR.has(step.id));
  try {
    feedSpectrum(app);
    mountFlowHero(app, document.getElementById("heroFlow"));
    const cfg = place3d(viz, step, app);
    if (cfg?.mode === "tall") return;
    if (!cfg) parkHero();
  } catch (e) { console.warn("3D scene unavailable", e); }
  switch (step.id) {
    case "problem": return problemStage(viz, app);
    case "operator": return operatorStage(viz, app);
    case "invariance": return invarianceStage(viz, app);
    case "flow": return flowStage(viz, app, { counts: true });
    case "split": {
      // renderLayer clears whatever it is handed, so it gets its own host and the 3D strip above it stays.
      const host = el("div", { class: "layer-host" });
      viz.append(host);
      return renderLayer(host, app);
    }
    case "convwave": case "convkernel": case "convone": case "convslide": case "convout": return convWaveStage(viz, fresh, step.id);
    case "convsum": case "convsplit": case "convchange": case "convadd": return convSumStage(viz, fresh, step.id);
    case "convcost": return convCostStage(viz, fresh);
    case "convwide": return convWideStage(viz, app);
    case "pixels": return pixelsStage(viz, app);
    case "overview": return flowStage(viz, app, {});
    case "project": return projectStage(viz, app);
    case "lift": return liftStage(viz, app);
    case "recipe": return waveSumStage(viz, "recipe", "Waves, added up", { tabs: ["mix", "build"] });
    case "recipe2d": return waveSumStage(viz, "recipe2d", "The same idea in two dimensions", { tabs: ["surface"], cutoff: 8 });
    case "energy": return waveSumStage(viz, "energy", "Energy, measured on the recipe", { tabs: ["mix"], showEnergy: true });
    case "waves": return wavesStage(viz, app);
    case "paths": return pathsStage(viz, app);
    case "truncate": return truncateStage(viz, app);
    case "widget": return widgetStage(viz, app);
    case "resolution": return resolutionStage(viz, app);
    case "bridge": return bridgeStage(viz, app);
    case "stillfails": return stillFailsStage(viz, app);
    case "draw": return flowStage(viz, app, { draw: true });
    case "stored": return storedStage(viz, app);
    case "multiply": return multiplyStage(viz, app);
    case "limits": return limitsStage(viz, app);
    case "block": return blockStage(viz, app);
    default: return flowStage(viz, app, {});
  }
}

// ---------- act 0: solver vs model, same rock ----------
function problemStage(viz, app) {
  const { sample, state } = app;
  const res = state.res;
  const live = sample.live;
  const io = sample.io[res];
  const pred = live && !state.draw ? live.prediction : io.prediction;
  // The two output fields share one scale, or the comparison means nothing.
  const scale = outScale(io.target, pred), vmax = scale.vmax, w = words();
  const rockCv = el("canvas"), trueCv = el("canvas"), predCv = el("canvas");
  paintInput(rockCv, io.input, res);
  paintField(trueCv, io.target, res, scale);
  paintField(predCv, pred, res, scale);

  const card = (kind, eyebrow, title, sub, cv, foot) => el("figure", { class: `prob-card ${kind}` },
    el("figcaption", { class: "prob-cap" },
      el("span", { class: "prob-eyebrow" }, eyebrow),
      el("span", { class: "cap-t" }, title),
      el("span", { class: "cap-s" }, sub)),
    el("div", { class: "tile" }, cv),
    el("div", { class: "prob-foot" }, foot));

  viz.append(el("div", { class: "panel-body" },
    el("h4", { class: "drill-sub" }, isNS() ? `One stirred fluid, ${res}×${res}, and the two ways to find out how it swirls`
      : `One rock, ${res}×${res}, and the two ways to get its pressure field`),
    el("div", { class: "prob-row" },
      isNS() ? card("given", "Given", "The forcing", "how the fluid is pushed everywhere. It starts at rest.", rockCv, "the only input")
        : card("given", "Given", "The rock", "two materials, light is loose", rockCv, "the only input"),
      el("span", { class: "prob-split", "aria-hidden": "true" }, "→"),
      isNS() ? card("solver", "Way 1 · simulate", "Simulated", "a classical solver on a 1024×1024 grid, stepped forward to T = 5", trueCv,
        `computed offline, then sampled to ${res}×${res}`)
        : card("solver", "Way 1 · simulate", "Solved", `${(res * res).toLocaleString("en-US")} cells, one equation each`, trueCv,
          "computed offline, before this page existed"),
      card("model", "Way 2 · predict", "Predicted", "one pass through the network", predCv,
        live && !state.draw ? `${Math.round(live.ms)} ms, live in this tab${live.error != null ? ` · ${pct(live.error, 1)} off the simulation` : ""}` : "from the exported run")),
    el("p", { class: "cap-s" }, isNS()
      ? `Both swirl fields (vorticity: red spins one way, blue the other) on one colour scale, ±${fmt(vmax, 2)}. A test sample the model never trained on.`
      : `Both pressure fields on one colour scale, 0 to ${fmt(vmax, 2)}. The rock is a test sample the model never trained on.`)));
}

// ---------- act 0.25: a function in, a function out ----------
// The same forcing sampled on three grids: an ordinary network fits exactly one of them, an operator takes all of them.
function operatorStage(viz, app) {
  const m = app.manifest, sample = app.sample;
  const trained = m.model.trained_resolution;
  const card = (r, ok, badge) => {
    const cv = el("canvas");
    paintInput(cv, sample.io[r].input, r);
    return el("figure", { class: `op-card ${ok ? "ok" : "no"}` },
      el("div", { class: "tile" }, cv),
      el("figcaption", {},
        el("span", { class: "cap-t" }, `${r} × ${r}`),
        el("span", { class: "cap-s" }, `${(r * r).toLocaleString("en-US")} numbers`),
        el("span", { class: "op-badge" }, badge)));
  };
  const row = (title, sub, cards) => el("div", { class: "op-row" },
    el("div", { class: "op-head" }, el("span", { class: "op-title" }, ...title), el("span", { class: "cap-s" }, sub)),
    el("div", { class: "op-cards" }, ...cards));
  viz.append(el("div", { class: "panel-body" },
    el("h4", { class: "drill-sub" }, `One ${words().input}, sampled on three grids`),
    row([M("f:\\ \\mathbb{R}^n \\to \\mathbb{R}^m"), " — an ordinary network"], `n is baked into the first layer: it takes exactly ${(trained * trained).toLocaleString("en-US")} numbers and nothing else`,
      m.resolutions.map((r) => card(r, r === trained, r === trained ? "fits" : "doesn't fit"))),
    row([M("\\mathcal{G}:\\ a(x) \\mapsto u(x)"), " — an operator"], "the input is a function, so any grid that samples it is fine",
      m.resolutions.map((r) => card(r, true, "accepted"))),
    el("p", { class: "cap-s" }, `The same ${words().input} from one sample, evaluated on ${m.resolutions.map((r) => `${r}×${r}`).join(", ")} grids. The operator's output is a function too, so you can ask for it on whatever grid you like.`)));
}

// ---------- act 0.5: the hook ----------
function invarianceStage(viz, app) {
  const { manifest: m, sample } = app;
  const scale = outScale(...m.resolutions.map((r) => sample.io[r].prediction));
  const row = el("div", { class: "inv-row" });
  for (const r of m.resolutions) {
    const cv = el("canvas");
    paintField(cv, sample.io[r].prediction, r, scale);
    const trained = r === m.model.trained_resolution;
    row.append(el("figure", { class: `inv-card${trained ? " trained" : ""}` },
      el("div", { class: "tile" }, cv),
      el("figcaption", {},
        el("span", { class: "cap-t" }, `${r} × ${r}`),
        el("span", { class: "cap-s" }, `${(r * r).toLocaleString("en-US")} cells`),
        el("span", { class: "inv-err mono" }, `test error ${pct(m.stats.test_error[String(r)].mean, 3)}`),
        el("span", { class: `inv-tag${trained ? " on" : ""}` }, trained ? "trained on this" : "never seen this"))));
  }
  viz.append(el("div", { class: "panel-body" },
    el("div", { class: "inv-head" },
      el("h4", { class: "drill-sub" }, "The same weights, three different grids"),
      el("span", { class: "inv-badge mono" }, `${m.model.params.toLocaleString("en-US")} weights · unchanged`)),
    row,
    el("p", { class: "cap-s" }, `Predicted ${words().output} for one sample at each grid size, on one colour scale. Test error is the mean over ${m.stats.test_error["64"].n} unseen samples. Nothing about the model changes between them — no retraining, no resizing, no interpolation of weights.`)));
}

// ---------- steps 1, 2, 9 ----------
// ---------- lifting: a(x), x, y in; 32 channels out ----------
function liftStage(viz, app) {
  const { sample, manifest, state } = app;
  const live = sample.live;
  const res = sample.res;
  const io = live ? { input: live.input } : sample.io[res];
  const w = words();
  const ramp = (axis) => {
    const f = new Float64Array(res * res);
    for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) f[i * res + j] = (axis === "x" ? i : j) / (res - 1);
    return f;
  };
  const tile = (cv, label, sub) => el("figure", { class: "ftile" }, el("div", { class: "tile" }, cv),
    el("figcaption", {}, el("span", { class: "cap-t" }, label), sub ? el("span", { class: "cap-s" }, sub) : null));
  const aCv = el("canvas"), xCv = el("canvas"), yCv = el("canvas");
  paintInput(aCv, io.input, res);
  const seq = { mode: "sequential", vmax: 1, vmin: 0 };
  paintField(xCv, ramp("x"), res, seq);
  paintField(yCv, ramp("y"), res, seq);

  const fields = live ? live.lifted : sample.fields.lifted;
  const order = manifest.rankings.lifted[state.criterion].order.slice(0, 8);
  const outs = order.map((c) => {
    const cv = el("canvas");
    paintField(cv, fields[c], sample.fieldRes ?? 64);
    return tile(cv, `channel ${c}`);
  });
  viz.append(el("div", { class: "panel-body lift" },
    el("h4", { class: "drill-sub" }, "The lift: three input channels in, 32 out"),
    el("div", { class: "lift-row" },
      el("div", { class: "lift-in" }, tile(aCv, "a(x)", w.input), tile(xCv, "x", "position"), tile(yCv, "y", "position")),
      el("div", { class: "lift-op" }, el("span", { class: "mono" }, "P"), el("span", { class: "cap-s" }, "1\u00d71 conv")),
      el("div", { class: "lift-out" }, ...outs)),
    el("p", { class: "cap-s" }, `Left: what goes in, ${res}\u00d7${res}. Right: 8 of the 32 channels that come out, picked by the same ranking as the full network diagram. Each output channel is a different mix of the three inputs at every pixel.`)));
}

// ---------- projection: layer 4's channels in, one prediction out, the solver's answer beside it ----------
function projectStage(viz, app) {
  const { sample, manifest, state } = app;
  const live = sample.live, res = sample.res;
  const io = live ? { input: live.input, prediction: live.prediction, target: live.target } : sample.io[res];
  const fields = live ? live.layers[3].out : sample.fields["layer4.out"];
  const order = manifest.rankings.layer4[state.criterion].order.slice(0, 8);
  const tile = (cv, label, sub) => el("figure", { class: "ftile" }, el("div", { class: "tile" }, cv),
    el("figcaption", {}, el("span", { class: "cap-t" }, label), sub ? el("span", { class: "cap-s" }, sub) : null));
  const outs = order.map((c) => { const cv = el("canvas"); paintField(cv, fields[c], sample.fieldRes ?? 64); return tile(cv, `channel ${c}`); });
  const scale = outScale(io.prediction, ...(io.target ? [io.target] : []));
  const pCv = el("canvas"), tCv = el("canvas");
  paintField(pCv, io.prediction, res, scale);
  if (io.target) paintField(tCv, io.target, res, scale);
  const err = live ? live.error : sample.error?.[res];
  viz.append(el("div", { class: "panel-body lift" },
    el("h4", { class: "drill-sub" }, "The projection: 32 channels in, one out"),
    el("div", { class: "lift-row" },
      el("div", { class: "lift-out" }, ...outs),
      el("div", { class: "lift-op" }, el("span", { class: "mono" }, "Q"), el("span", { class: "cap-s" }, "1\u00d71 conv")),
      el("div", { class: "lift-in" }, tile(pCv, "prediction", err != null ? `error ${pct(err, 1)}` : "no solver run"),
        io.target ? tile(tCv, "solver's answer", "same scale") : null)),
    el("p", { class: "cap-s" }, "Left: 8 of layer 4's 32 channels, picked by the same ranking as the full network view. Right: what the last 1\u00d71 convolution makes of them, next to the answer a classical solver computed.")));
}

function flowStage(viz, app, { counts = false, draw = false }) {
  if (draw) viz.append(renderDraw(app));
  const host = el("div", { class: "flow-host" });
  viz.append(host);
  renderFlow(host, app, { counts });
}

// ---------- steps "recipe", "recipe2d", "energy": the wave-sum experiments ----------
function waveSumStage(viz, id, heading, opts) {
  const host = el("div", { class: "wavesum-host" });
  viz.append(el("div", { class: "panel-body" }, el("h4", { class: "drill-sub" }, heading), host));
  mountWaveSum(host, { id, ...opts });
}

// ---------- step "waves": ground wave + energy before paths/truncate/spectrum need them ----------
function wavesStage(viz, app) {
  viz.append(el("div", { class: "panel-body" },
    el("h4", { class: "drill-sub" }, "Any image, rebuilt from waves like these"),
    ...waveIntroPanel(app)));
}

// ---------- step 4: the centrepiece ----------
function pathsStage(viz, app) {
  const l = app.state.layer;
  const stats = layerStats(app, l);
  const c = app.state.channel ?? stats.default_channel;
  const F = layerFields(app, l, c);
  const res = F.res;
  // One scale across both paths and their sum. Normalising each tile on its own would erase the point.
  const shared = Math.max(maxAbs(F.spectral), maxAbs(F.local), maxAbs(F.sum));

  const m = mask(res);
  const sp = splitAtCutoff(F.spectral, res, m);
  const lc = splitAtCutoff(F.local, res, m);
  const hiMax = Math.max(maxAbs(sp.high), maxAbs(lc.high)) || 1;

  viz.append(...layerHeader(app));
  viz.append(el("div", { class: "paths-grid" },
    el("div", { class: "paths-row" },
      fieldTile(F.spectral, res, shared, { big: true, label: "Spectral path", sub: `std ${fmt(spatialStd(F.spectral), 2)}`, path: "spectral" }),
      fieldTile(F.local, res, shared, { big: true, label: "Local path", sub: `std ${fmt(spatialStd(F.local), 2)}`, path: "local" }),
      el("span", { class: "plus-chip", "aria-hidden": "true" }, "+"),
      fieldTile(F.sum, res, shared, { big: true, label: "Sum", sub: "what the layer carries on" })),
    el("div", { class: "scale-note" }, el("span", { class: "grad", "aria-hidden": "true" }),
      el("span", {}, `All three on one colour scale: ±${fmt(shared, 2)}. Red positive, blue negative.`)),
    el("h4", { class: "drill-sub" }, "The same two fields, keeping only what lies above the cutoff"),
    el("div", { class: "paths-row" },
      fieldTile(sp.high, res, hiMax, { label: "Spectral, above cutoff", sub: "nothing to draw", path: "spectral" }),
      fieldTile(lc.high, res, hiMax, { label: "Local, above cutoff", sub: `shared scale ±${fmt(hiMax, 3)}`, path: "local" })),
    readout(app, l, stats)));
}

function readout(app, l, stats) {
  const s = stats.outside_cutoff_share_spectral;
  return el("div", { class: "readout wide" },
    el("div", { class: "ro-head" }, "Energy above the cutoff",
      el("span", {}, `all 32 channels of layer ${l}, both paths, this sample`)),
    bar("Local path", stats.outside_cutoff_share_local, "local", "all of it"),
    bar("Spectral path", s, "spectral", s === 0 ? "exactly 0" : `${s.toExponential(1)} — rounding error, not signal`),
    el("p", { class: "ro-foot" }, `Truncation keeps ${pct(stats.truncation_kept_excl_mean, 1)} of this layer input's energy, excluding the mean. The missing ${pct(1 - stats.truncation_kept_excl_mean, 1)} is the sharp part, and only the local path still has it.`));
}

function bar(label, share, path, note) {
  return el("div", { class: `ro-bar p-${path}` },
    el("span", { class: "ro-lab" }, label),
    el("span", { class: "ro-track" }, el("span", { class: "ro-fill", style: `width:${Math.max(0, Math.min(1, share)) * 100}%` })),
    el("span", { class: "ro-val mono" }, share < 0.0005 ? "0.0%" : pct(share, 1)),
    el("span", { class: "ro-note" }, note));
}

// ---------- the spectral path, first node: every wave of the real input, the block we keep, the cut ----------
function truncateStage(viz, app) {
  const l = app.state.layer;
  viz.append(...layerHeader(app, { picker: false }));
  viz.append(el("div", { class: "panel-body drill" }, ...rPanel(app, { l, truncOnly: true })));
  viz.append(el("div", { class: "panel-body" }, ...truncateRealPanel(app, { l })));
  viz.append(el("div", { class: "panel-body" }, buildUpPanel(app, { l })));
}

// ---------- back to pixels: the waves after × R, then the grid again ----------
function pixelsStage(viz, app) {
  const l = app.state.layer;
  const stats = layerStats(app, l);
  const c = app.state.channel ?? stats.default_channel;
  const F = layerFields(app, l, c);
  viz.append(...layerHeader(app));
  viz.append(el("div", { class: "panel-body" },
    el("div", { class: "drill-row" },
      heatTile(app, { l }),
      el("span", { class: "op-chip" }, "inverse FFT"),
      fieldTile(F.spectral, F.res, maxAbs(F.spectral), { big: true, label: `Spectral path output, channel ${c}`, sub: `back on the ${F.res}×${F.res} grid`, path: "spectral" })),
    el("p", { class: "cap-s" }, "Left: how much energy each wave holds after the multiply, with everything outside the block still discarded. Right: one channel of what comes back out, a grid of pixels again.")));
}

// ---------- step 6 ----------
function widgetStage(viz, app) {
  viz.append(el("div", { class: "panel-body" },
    el("h4", { class: "drill-sub" }, "One slice, rebuilt from waves"),
    ...truncateWidgetPanel()));
}

// ---------- resolution, checked: both datasets side by side ----------
/** One row of three grids for one dataset, painted explicitly (not through the page-wide problem). */
function resRow(app, key) {
  const m = app.ds[key].manifest, sample = app.sampleOf(key), ns = key === "ns";
  const te = m.stats.test_error;
  const preds = m.resolutions.map((r) => sample.io[r].prediction);
  const vmax = Math.max(...preds.map(maxAbs)) || 1;
  const row = el("div", { class: "res-row" });
  m.resolutions.forEach((r, i) => {
    const io = sample.io[r];
    const inCv = el("canvas"), predCv = el("canvas");
    if (ns) paintField(inCv, io.input, r, { vmax: maxAbs(io.input) }); else paintBinary(inCv, io.input, r);
    paintField(predCv, preds[i], r, ns ? { vmax } : { mode: "sequential", vmin: 0, vmax });
    row.append(el("figure", { class: `res-card${r === m.model.trained_resolution ? " on" : ""}` },
      el("figcaption", { class: "res-cap" }, el("span", { class: "cap-t" }, `${r} × ${r}`),
        el("span", { class: "cap-s" }, r === m.model.trained_resolution ? "trained here" : "never trained here")),
      el("div", { class: "res-pair" }, el("div", { class: "tile" }, inCv), el("div", { class: "tile" }, predCv)),
      el("div", { class: "res-nums" },
        el("div", {}, el("span", { class: "k" }, `test set, ${te[String(r)].n}`), el("span", { class: "v mono" }, pct(te[String(r)].mean, 3))))));
  });
  return el("div", { class: "res-block" },
    el("h4", { class: "drill-sub" }, ns ? "Fluid (Navier–Stokes): forcing → swirl" : "Rock (Darcy): rock map → pressure"), row);
}

function resolutionStage(viz, app) {
  const ns = app.ds.ns.manifest;
  const cons = ns.analysis?.ns?.resolution_consistency;
  viz.append(el("div", { class: "panel-body" },
    resRow(app, "ns"),
    resRow(app, "darcy"),
    cons ? el("p", { class: "cap-s" }, `Fluid: the 64×64 prediction matches the 128×128 one sampled at every 2nd point to within ${fmt(cons.max_abs_pred128_sub2_minus_pred64, 4)}, on a field that peaks at ${fmt(cons.pred_max_abs, 2)} (200 test samples). Each pair: input left, prediction right; predictions share one scale per row. Exported from PyTorch.`) : null));
}

// ---------- bridge: from the smooth fluid to the sharp rock ----------
function bridgeStage(viz, app) {
  const an = app.ds.ns.manifest.analysis;
  const mask = keptMask(64);
  const card = (key) => {
    const ns = key === "ns", input = app.sampleOf(key).io[64].input;
    const hi = splitAtCutoff(input, 64, mask).high;
    const a = el("canvas"), b = el("canvas");
    if (ns) paintField(a, input, 64, { vmax: maxAbs(input) }); else paintBinary(a, input, 64);
    paintField(b, hi, 64, { vmax: maxAbs(hi) || 1 });
    const share = an?.[key]?.energy_above_cutoff_64?.input;
    return el("figure", { class: `bridge-card ${key}` },
      el("figcaption", {}, el("span", { class: "cap-t" }, ns ? "the fluid's forcing" : "a rock map"),
        el("span", { class: "cap-s" }, ns ? "smooth everywhere" : "two materials, sharp edges")),
      el("div", { class: "bridge-pair" },
        el("div", { class: "tile" }, a),
        el("div", { class: "tile" }, b)),
      el("div", { class: "bridge-num" },
        el("span", { class: "v mono" }, share != null ? pct(share, 2) : "—"),
        el("span", { class: "cap-s" }, "of its energy is above the cutoff")));
  };
  viz.append(el("div", { class: "panel-body" },
    el("h4", { class: "drill-sub" }, "Left: the input. Right: only the part above the cutoff, each on its own scale."),
    el("div", { class: "bridge-row" }, card("ns"), card("darcy")),
    el("p", { class: "cap-s" }, "Shares are pooled over 200 test samples per problem. From here until the resolution check, the page runs the rock model: same architecture, same cutoff, trained separately.")));
}

// ---------- where it still fails: the fluid's mistakes, in waves ----------
function stillFailsStage(viz, app) {
  const m = app.ds.ns.manifest, an = m.analysis;
  const s = app.sampleOf("ns"), io = s.io[64];
  const pred = s.live && s.live.res === 64 && !s.live.draw ? s.live.prediction : io.prediction;
  const err = Float64Array.from(pred, (v, i) => v - io.target[i]);
  const errCv = el("canvas"), truthCv = el("canvas");
  paintField(truthCv, io.target, 64, { vmax: maxAbs(io.target) });
  paintField(errCv, err, 64, { vmax: maxAbs(err) || 1 });
  const E = an.ns.energy_above_cutoff_64;
  const bars = [["forcing (input)", E.input], ["true swirl", E.truth], ["predicted swirl", E.prediction], ["the error", E.error]];
  const top = Math.max(...bars.map(([, v]) => v));
  const A = (k) => an[k].ablation_64;
  viz.append(el("div", { class: "panel-body" },
    el("div", { class: "fail-top" },
      el("figure", { class: "ftile fail-tile" }, el("div", { class: "tile" }, truthCv),
        el("figcaption", {}, el("span", { class: "cap-t" }, "true swirl"), el("span", { class: "cap-s" }, `sample #${s.entry.test_index}, 64×64`))),
      el("figure", { class: "ftile fail-tile" }, el("div", { class: "tile" }, errCv),
        el("figcaption", {}, el("span", { class: "cap-t" }, "prediction − truth"), el("span", { class: "cap-s" }, `own scale, ±${fmt(maxAbs(err), 2)}`)))),
    el("h4", { class: "drill-sub" }, "Share of each field's energy that sits above the cutoff (fluid, 200 test samples)"),
    el("div", { class: "fail-bars" }, ...bars.map(([label, v]) => el("div", { class: `fail-bar${label === "the error" ? " hot" : ""}` },
      el("span", { class: "fb-lab" }, label),
      el("span", { class: "fb-track" }, el("span", { class: "fb-fill", style: `width:${Math.max(0.6, (v / top) * 100)}%` })),
      el("span", { class: "fb-val mono" }, pct(v, 2))))),
    el("h4", { class: "drill-sub" }, "Switch off the local path's fine detail, nothing retrained"),
    el("table", { class: "mini" },
      el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", {}, "fluid"), el("th", {}, "rock"))),
      el("tbody", {},
        el("tr", {}, el("td", {}, "error as trained"), el("td", { class: "num" }, pct(A("ns").baseline.mean, 2)), el("td", { class: "num" }, pct(A("darcy").baseline.mean, 2))),
        el("tr", { class: "on" }, el("td", {}, "local path's above-cutoff part removed"), el("td", { class: "num" }, pct(A("ns").local_above_cutoff_removed.mean, 2)), el("td", { class: "num" }, pct(A("darcy").local_above_cutoff_removed.mean, 2))),
        el("tr", {}, el("td", {}, "whole local path removed"), el("td", { class: "num" }, pct(A("ns").local_removed.mean, 1)), el("td", { class: "num" }, pct(A("darcy").local_removed.mean, 1))))),
    el("p", { class: "cap-s" }, `Over ${A("ns").n} test samples per problem at 64×64. "Removed" means that part of the local path's output is replaced by nothing in all four layers, in the trained PyTorch model.`)));
}

// ---------- the whole layer as implemented: the two paths plus everything around them ----------
function blockStage(viz, app) {
  const m = app.manifest, t = m.weights.tensors, l = app.state.layer;
  const shape = (k) => t[k]?.shape.join("×") ?? "";
  const box = (cls, label, sub) => el("div", { class: `bk ${cls}` }, el("span", { class: "bk-l" }, label), sub ? el("span", { class: "bk-s mono" }, sub) : null);
  const arrow = (txt) => el("div", { class: "bk-arrow", "aria-hidden": "true" }, txt);
  viz.append(...layerHeader(app, { picker: false }));
  viz.append(el("div", { class: "panel-body block" },
    el("h4", { class: "drill-sub" }, `One whole layer, as implemented (layer ${l} shapes)`),
    el("div", { class: "bk-flow" },
      box("io", "v", "the layer's 32 input channels"),
      arrow("↓"),
      el("div", { class: "bk-block" },
        el("div", { class: "bk-split" },
          box("spectral", "Spectral path", `FFT → keep ${m.stats.n_kept_modes} waves → × R ${shape(`L${l - 1}.spec.w`)} → FFT⁻¹`),
          box("local", "Local path", `1×1 conv ${shape(`L${l - 1}.skip.w`)}`)),
        arrow("↓ add"),
        box("act", "GELU", "elementwise"),
        arrow("↓"),
        box("unit delta", "Channel MLP", `${shape(`L${l - 1}.mlp0.w`)} → GELU → ${shape(`L${l - 1}.mlp1.w`)}`),
        arrow("↓ add the layer's own input back in"),
        box("unit delta", "Gated skip  g ⊙ v", `one learned number per channel, ${shape(`L${l - 1}.gate.w`)}`),
        arrow("↓"),
        box("act", l < 4 ? "GELU" : "no GELU on layer 4", l < 4 ? "elementwise" : "its output goes straight to the projection")),
      arrow("↓"),
      box("io", `layer ${l} output`, "32 channels, same grid")),
    el("p", { class: "code mono" }, m.model.block),
    el("p", { class: "cap-s" }, `Shapes read from the exported weight index. Switch layers with the tabs above: every layer has this structure and only layer 4 drops the last GELU. Verified against the library: ${m.checks.length} assertions in the export, including that replaying this block math by hand reproduces the library's own output.`)));
}
