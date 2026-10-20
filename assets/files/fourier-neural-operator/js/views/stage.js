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
        live && !state.draw ? `${Math.round(live.ms)} ms, live in this tab${live.error != null ? ` · ${pct(live.error, 1)} off the simulation` : ""}` : app.model ? "from the exported run" : "the exported run, while the model loads")),
    el("p", { class: "cap-s" }, isNS()
      ? `Both swirl fields (vorticity: red spins one way, blue the other) on one color scale, ±${fmt(vmax, 2)}. A test sample the model never trained on.`
      : `Both pressure fields on one color scale, 0 to ${fmt(vmax, 2)}. The rock is a test sample the model never trained on.`)));
}

// ---------- act 0.25: a function in, a function out ----------
// One grid switch drives both rows: the operator runs on whatever grid it's given, while an ordinary
// network has exactly n input slots, drawn as a bar that the grid's numbers either fill, leave half empty, or spill past.
let opRes = null;
function operatorStage(viz, app) {
  const m = app.manifest, sample = app.sample, w = words();
  const trained = m.model.trained_resolution, n = trained * trained, most = Math.max(...m.resolutions) ** 2;
  opRes ??= trained;
  const scale = outScale(...m.resolutions.map((r) => sample.io[r].prediction));
  const body = el("div", { class: "op2" });
  const num = (v) => v.toLocaleString("en-US");

  function draw() {
    const r = opRes, count = r * r;
    const inCv = el("canvas"), outCv = el("canvas");
    paintInput(inCv, sample.io[r].input, r);
    paintField(outCv, sample.io[r].prediction, r, scale);
    const fill = Math.min(count, n), spill = Math.max(0, count - n);
    const verdict = count === n ? "fits exactly"
      : count < n ? `${num(n - count)} slots left empty, so it can't run`
      : `${num(spill)} numbers with nowhere to go, so it can't run`;
    body.replaceChildren(
      el("div", { class: "op2-ctl" }, el("span", { class: "cv-lab" }, "grid"),
        el("div", { class: "seg" }, ...m.resolutions.map((g) => el("button", { type: "button", "aria-pressed": String(g === r), onclick: () => { opRes = g; draw(); } }, `${g} × ${g}`)))),
      el("div", { class: "op2-row" },
        el("div", { class: "op-head" }, el("span", { class: "op-title" }, M("\\mathcal{G}:\\ a(x) \\mapsto u(x)"), " — an operator"),
          el("span", { class: "cap-s" }, `the same ${num(m.model.params)} weights on every grid`)),
        el("div", { class: "op2-flow" },
          el("figure", { class: "op2-card" }, el("div", { class: "tile" }, inCv),
            el("figcaption", {}, el("span", { class: "cap-t" }, `${w.input}, ${r} × ${r}`), el("span", { class: "cap-s" }, `${num(count)} numbers in`))),
          el("span", { class: "op2-arrow", "aria-hidden": "true" }, "→ 𝒢 →"),
          el("figure", { class: "op2-card" }, el("div", { class: "tile" }, outCv),
            el("figcaption", {}, el("span", { class: "cap-t" }, `${w.output}, ${r} × ${r}`), el("span", { class: "cap-s" }, `${num(count)} numbers out${r === trained ? " · trained on this grid" : " · never trained on this grid"}`))))),
      el("div", { class: "op2-row" },
        el("div", { class: "op-head" }, el("span", { class: "op-title" }, M("f:\\ \\mathbb{R}^n \\to \\mathbb{R}^m"), " — an ordinary network"),
          el("span", { class: "cap-s" }, `its first layer has exactly n = ${num(n)} input slots, one per number`)),
        el("div", { class: "op2-bar", role: "img", "aria-label": `${num(count)} numbers against ${num(n)} slots: ${verdict}` },
          el("div", { class: "op2-slots", style: `width:${(100 * n) / most}%` }),
          el("div", { class: "op2-fill", style: `width:${(100 * fill) / most}%` }),
          spill ? el("div", { class: "op2-spill", style: `left:${(100 * n) / most}%; width:${(100 * spill) / most}%` }) : null),
        el("div", { class: "op2-legend cap-s" },
          el("span", {}, `${num(count)} numbers from this grid · ${num(n)} slots`),
          el("strong", { class: count === n ? "ok" : "no" }, verdict))),
      el("p", { class: "cap-s" }, `The ${w.input} and the model's prediction for one test sample, on one color scale. The bar is drawn to scale.`));
  }
  draw();
  viz.append(el("div", { class: "panel-body" }, el("h4", { class: "drill-sub" }, `One ${w.input}, any grid`), body));
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
    el("p", { class: "cap-s" }, `Predicted ${words().output} for one sample at each grid size, on one color scale. Test error is the mean over ${m.stats.test_error["64"].n} unseen samples. Nothing about the model changes between them — no retraining, no resizing, no interpolation of weights.`),
    isNS() ? el("p", { class: "cap-s" }, `Why the three errors nearly match: the 32×32 and 64×64 grids are every 4th and every 2nd point of the same 128×128 data, and the fluid is smooth. The model only uses waves up to ${Math.max(...m.kept_modes_centered.map(([a]) => a)) - m.resolutions[1] / 2} in each direction, and even 32×32 can hold waves up to 16, so all three grids give it the same waves to work with.`) : null));
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

// ---------- step 4: the centerpiece ----------
function pathsStage(viz, app) {
  const l = app.state.layer;
  const stats = layerStats(app, l);
  const c = app.state.channel ?? stats.default_channel;
  const F = layerFields(app, l, c);
  const res = F.res;
  // One scale across both paths and their sum. Normalizing each tile on its own would erase the point.
  const shared = Math.max(maxAbs(F.spectral), maxAbs(F.local), maxAbs(F.sum));

  const m = mask(res);
  const sp = splitAtCutoff(F.spectral, res, m);
  const lc = splitAtCutoff(F.local, res, m);
  const hiMax = Math.max(maxAbs(sp.high), maxAbs(lc.high)) || 1;

  const sS = spatialStd(F.spectral), sL = spatialStd(F.local);
  const bigger = sS >= sL
    ? `The spectral path is the bigger signal here: its values spread ${fmt(sS / sL, 1)}× as wide as the local path's (standard deviation ${fmt(sS, 2)} against ${fmt(sL, 2)}).`
    : `In this channel the local path is actually the bigger signal (standard deviation ${fmt(sL, 2)} against ${fmt(sS, 2)}). Try a few other channels above.`;
  viz.append(...layerHeader(app));
  viz.append(el("div", { class: "paths-grid" },
    el("h4", { class: "drill-sub" }, "1 · Both paths, side by side"),
    el("p", { class: "lead-in" }, `Here's channel ${c} of layer ${l} coming out of each path, all on one color scale. ${bigger}`),
    el("div", { class: "paths-row" },
      fieldTile(F.spectral, res, shared, { big: true, label: "Spectral path", sub: `std ${fmt(spatialStd(F.spectral), 2)}`, path: "spectral" }),
      fieldTile(F.local, res, shared, { big: true, label: "Local path", sub: `std ${fmt(spatialStd(F.local), 2)}`, path: "local" }),
      el("span", { class: "plus-chip", "aria-hidden": "true" }, "+"),
      fieldTile(F.sum, res, shared, { big: true, label: "Sum", sub: "what the layer carries on" })),
    el("div", { class: "scale-note" }, el("span", { class: "grad", "aria-hidden": "true" }),
      el("span", {}, `All three on one color scale: ±${fmt(shared, 2)}. Red positive, blue negative.`)),
    el("h4", { class: "drill-sub" }, "2 · Only what's above the cutoff"),
    el("p", { class: "lead-in" }, "Now strip both of them down to just the fast waves, everything above the cutoff. That's the sharp detail. The spectral tile comes out blank, because the cutoff already removed all of it. The local tile is what's left of the detail, and it sits along the edges of the rock."),
    el("div", { class: "paths-row" },
      fieldTile(sp.high, res, hiMax, { label: "Spectral, above cutoff", sub: "nothing to draw", path: "spectral" }),
      fieldTile(lc.high, res, hiMax, { label: "Local, above cutoff", sub: `shared scale ±${fmt(hiMax, 3)}`, path: "local" })),
    el("p", { class: "cap-s" }, `Across all 32 channels, ${pct(1 - stats.truncation_kept_excl_mean, 1)} of the energy coming into layer ${l} sits above the cutoff (not counting the flat average). That's the sharp part, and of the two paths, only the local one passes it on.`)));
}

// ---------- the spectral path, first node: every wave of the real input, the block we keep, the cut ----------
function truncateStage(viz, app) {
  const l = app.state.layer;
  viz.append(...layerHeader(app, { picker: false }));
  // In the order the text goes: the waves, proof they are the input, the cut, what it drops, where it hurts.
  const [title, lead, grid, lost] = rPanel(app, { l, truncOnly: true });
  viz.append(el("div", { class: "panel-body drill" }, title, lead, grid));
  viz.append(el("div", { class: "panel-body" }, buildUpPanel(app, { l })));
  viz.append(el("div", { class: "panel-body" }, ...truncateRealPanel(app, { l, part: "real" })));
  viz.append(el("div", { class: "panel-body drill" }, lost));
  viz.append(el("div", { class: "panel-body" }, ...truncateRealPanel(app, { l, part: "rock" })));
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
  const bars = [["the forcing (input)", E.input], ["the true swirl", E.truth], ["the model's prediction", E.prediction], ["the model's mistake", E.error]];
  const A = (k) => an[k].ablation_64;
  viz.append(el("div", { class: "panel-body" },
    el("h4", { class: "drill-sub" }, "1 · Where the mistakes are"),
    el("p", { class: "lead-in" }, `Here's the true swirl for sample #${s.entry.test_index}, next to the model's mistake: its prediction minus the truth. The mistake is small, and it sits along the thin streaks.`),
    el("div", { class: "fail-top" },
      el("figure", { class: "ftile fail-tile" }, el("div", { class: "tile" }, truthCv),
        el("figcaption", {}, el("span", { class: "cap-t" }, "true swirl"), el("span", { class: "cap-s" }, `sample #${s.entry.test_index}, 64×64`))),
      el("figure", { class: "ftile fail-tile" }, el("div", { class: "tile" }, errCv),
        el("figcaption", {}, el("span", { class: "cap-t" }, "prediction − truth"), el("span", { class: "cap-s" }, `own scale, ±${fmt(maxAbs(err), 2)}`)))),
    el("h4", { class: "drill-sub" }, "2 · How much of each is fine detail"),
    el("p", { class: "lead-in" }, "Each bar asks the same question: how much of this picture is fine detail, meaning the fast waves the spectral path throws away? The forcing, the true swirl and the model's prediction are all smooth, with well under 1% fine detail each. The mistake is the odd one out: almost half of it is fine detail. So the sharp stuff makes up a big share of what the model gets wrong, even though the swirl itself has almost none."),
    el("div", { class: "fail-bars" },
      el("div", { class: "fail-bar fail-head" }, el("span", {}, ""), el("span", { class: "fb-scale" }, el("span", {}, "0%"), el("span", {}, "50%"), el("span", {}, "100%")), el("span", { class: "fb-val" }, "fine detail")),
      ...bars.map(([label, v]) => el("div", { class: `fail-bar${label === "the model's mistake" ? " hot" : ""}` },
      el("span", { class: "fb-lab" }, label),
      el("span", { class: "fb-track" }, el("span", { class: "fb-fill", style: `width:${Math.max(0.6, v * 100)}%` })),
      el("span", { class: "fb-val mono" }, pct(v, 2))))),
    el("p", { class: "cap-s" }, `Each number is the share of that picture's energy sitting in the fast waves, the ones above the cutoff. 0% would be perfectly smooth. 100% would be all fine detail. Averaged over ${A("ns").n} test samples of the fluid.`),
    el("h4", { class: "drill-sub" }, "3 · Does the local path's fast detail help?"),
    el("p", { class: "lead-in" }, `To find out, we change the trained model without retraining it, and run the ${A("ns").n} test samples of each problem again.`),
    el("table", { class: "mini" },
      el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", {}, "fluid"), el("th", {}, "rock"))),
      el("tbody", {},
        el("tr", {}, el("td", {}, "the model as trained"), el("td", { class: "num" }, pct(A("ns").baseline.mean, 2)), el("td", { class: "num" }, pct(A("darcy").baseline.mean, 2))),
        el("tr", { class: "on" }, el("td", {}, "local path smoothed: in every layer, its waves above the cutoff are deleted"), el("td", { class: "num" }, pct(A("ns").local_above_cutoff_removed.mean, 2)), el("td", { class: "num" }, pct(A("darcy").local_above_cutoff_removed.mean, 2))),
        el("tr", {}, el("td", {}, "local path switched off entirely"), el("td", { class: "num" }, pct(A("ns").local_removed.mean, 1)), el("td", { class: "num" }, pct(A("darcy").local_removed.mean, 1))))),
    el("p", { class: "cap-s" }, `Error, averaged over ${A("ns").n} test samples per problem at 64×64, in the trained PyTorch model. Smoothing the local path leaves neither of the two paths able to pass on sharp detail. It costs the fluid about a tenth more error and the rock more than a third.`),
    el("p", { class: "cap-s" }, "Error is the size of the prediction's mistake relative to the size of the true field, so it can go past 100%. Predicting all zeros would score exactly 100%. With the local path switched off, the fluid model still draws roughly the right swirl, but about 2.5× too strong, so its mistake ends up bigger than the swirl itself.")));
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
