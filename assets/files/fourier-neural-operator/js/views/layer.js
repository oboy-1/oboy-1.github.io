// Zoom level 2: one Fourier layer, split into its spectral and local paths.
import { el, paintField, maxAbs, fmt, pct } from "../render.js";
import { geluArray, spatialStd } from "../nn.js";

const NODES = {
  fft: { label: "FFT", path: "spectral", title: "Fourier transform" },
  truncate: { label: "|k| ≤ 8", path: "spectral", title: "Truncation" },
  r: { label: "× R", path: "spectral", title: "Learned mode weights" },
  ifft: { label: "FFT⁻¹", path: "spectral", title: "Back to space", drill: "fft" },
  conv: { label: "1×1 conv W", path: "local", title: "Local path" },
};

/** Fields for one layer and channel: from the live forward pass if present, else precomputed atlases. */
export function layerFields(app, l, c) {
  const s = app.sample;
  const live = s.live;
  const spectral = live ? live.layers[l - 1].spectral[c] : s.fields[`layer${l}.spectral`][c];
  const local = live ? live.layers[l - 1].local[c] : s.fields[`layer${l}.local`][c];
  const out = live ? live.layers[l - 1].out[c] : s.fields[`layer${l}.out`][c];
  const sum = new Float64Array(spectral.length);
  for (let i = 0; i < sum.length; i++) sum[i] = spectral[i] + local[i];
  const act = l < 4 ? geluArray(sum) : sum;
  return { spectral, local, sum, act, out, res: live ? live.res : 64 };
}

export function layerStats(app, l) {
  // Live stats (computed in the browser from float64 fields) take precedence over the manifest.
  return app.sample.live?.layerStats?.[l - 1] ?? app.sample.entry.layers[l - 1];
}

export function fieldTile(field, res, vmax, { big = false, label, sub, path } = {}) {
  const cv = el("canvas");
  paintField(cv, field, res, { vmax });
  return el("figure", { class: `ftile${big ? " big" : ""}${path ? ` p-${path}` : ""}` },
    el("div", { class: "tile" }, cv),
    el("figcaption", {}, el("span", { class: "cap-t" }, label), sub ? el("span", { class: "cap-s" }, sub) : null));
}

// A labeled step in the diagram. Not clickable: the slides after this one explain each step.
function node(app, key) {
  const n = NODES[key];
  return el("span", { class: `node static p-${n.path}`, title: n.title }, n.label);
}

function arrowLine(kind) {
  return el("span", { class: `wire ${kind}`, "aria-hidden": "true" });
}

/** Layer tabs and the 32-channel picker. Shared by every step that sits inside a layer. */
export function layerHeader(app, { picker: withPicker = true } = {}) {
  const { sample } = app;
  const l = app.state.layer;
  const stats = layerStats(app, l);
  const c = app.state.channel ?? stats.default_channel;
  const res = sample.live ? sample.live.res : 64;
  const tabs = el("div", { class: "tabs", role: "tablist", "aria-label": "Layer" },
    [1, 2, 3, 4].map((k) => {
      const b = el("button", { role: "tab", "aria-selected": String(k === l) }, `Layer ${k}`);
      b.addEventListener("click", () => app.set({ layer: k, channel: null }));
      return b;
    }));
  const out = [el("div", { class: "layer-top" },
    el("div", { class: "crumbs" }, el("span", { class: "here" }, `Layer ${l} of 4`), el("span", { class: "hint" }, "pick any layer")), tabs)];
  if (!withPicker) return out;

  const pick = el("div", { class: "chpick", role: "listbox", "aria-label": "Output channel" });
  const outFields = sample.live ? sample.live.layers[l - 1].out : sample.fields[`layer${l}.out`];
  for (let k = 0; k < 32; k++) {
    const cv = el("canvas");
    paintField(cv, outFields[k], res);
    const b = el("button", { class: "chip", role: "option", "aria-selected": String(k === c), title: `channel ${k}` }, cv, el("span", {}, k));
    b.addEventListener("click", () => app.set({ channel: k }));
    pick.append(b);
  }
  out.push(el("div", { class: "chpick-wrap" },
    el("div", { class: "chpick-head" },
      el("span", {}, "Output channel"),
      el("span", { class: "mono" }, `ch${String(c).padStart(2, "0")}`),
      c === stats.default_channel ? el("span", { class: "hint" }, "largest combined spatial std in this layer") : null),
    pick));
  return out;
}

export function renderLayer(root, app) {
  const { state, manifest, sample } = app;
  const l = state.layer;
  const stats = layerStats(app, l);
  const c = state.channel ?? stats.default_channel;
  const F = layerFields(app, l, c);
  const res = F.res;
  const shared = Math.max(maxAbs(F.spectral), maxAbs(F.local), maxAbs(F.sum));

  root.replaceChildren();
  root.append(...layerHeader(app));

  // the two-path diagram
  const prevKey = l === 1 ? "lifted" : `layer${l - 1}.out`;
  const inFields = sample.live ? (l === 1 ? sample.live.lifted : sample.live.layers[l - 2].out) : sample.fields[prevKey];
  const vStack = el("div", { class: "vstack" },
    [0, 1, 2].map((k) => { const cv = el("canvas"); paintField(cv, inFields[[c, (c + 11) % 32, (c + 23) % 32][k]], res); return el("div", { class: "tile" }, cv); }));

  const diagram = el("div", { class: "twopath" },
    el("div", { class: "tp-in" }, vStack, el("div", { class: "cap-t" }, "v"), el("div", { class: "cap-s" }, `32 channels in, from ${l === 1 ? "P" : `layer ${l - 1}`}`)),
    el("div", { class: "tp-split", "aria-hidden": "true" }),
    el("div", { class: "tp-row spectral" },
      node(app, "fft"), arrowLine("spectral"), node(app, "truncate"), arrowLine("spectral"), node(app, "r"), arrowLine("spectral"), node(app, "ifft"), arrowLine("spectral grow")),
    el("div", { class: "tp-row local" }, node(app, "conv"), arrowLine("local grow")),
    el("div", { class: "tp-out spectral" }, fieldTile(F.spectral, res, shared, { big: true, label: "Spectral path", sub: `std ${fmt(spatialStd(F.spectral), 2)}`, path: "spectral" })),
    el("div", { class: "tp-out local" }, fieldTile(F.local, res, shared, { big: true, label: "Local path", sub: `std ${fmt(spatialStd(F.local), 2)}`, path: "local" })),
    el("div", { class: "tp-merge", "aria-hidden": "true" }, el("span", { class: "plus" }, "+")),
    el("div", { class: "tp-sum" }, fieldTile(F.sum, res, shared, { big: true, label: "Sum", sub: "before the nonlinearity" })),
  );


  const outMax = maxAbs(F.out);
  const after = el("div", { class: "after" },
    el("span", { class: "after-lab" }, "then"),
    fieldTile(F.sum, res, shared, { label: "sum" }),
    el("span", { class: "op-chip" }, l < 4 ? "GELU" : "no GELU in layer 4"),
    fieldTile(F.act, res, shared, { label: l < 4 ? "after GELU" : "unchanged" }),
    el("span", { class: "op-chip wide" }, "channel MLP + gated skip of v", l < 4 ? " → GELU" : ""),
    fieldTile(F.out, res, outMax, { label: `layer ${l} output`, sub: `own scale ±${fmt(outMax, 2)}` }));

  const legend = el("div", { class: "scale-note" },
    el("span", { class: "grad", "aria-hidden": "true" }),
    el("span", {}, `Spectral, local, sum and after-GELU share one color scale: ±${fmt(shared, 2)}. Red positive, blue negative.`));

  // the central readout
  const sShare = stats.outside_cutoff_share_spectral, lShare = stats.outside_cutoff_share_local;
  const readout = el("div", { class: "readout" },
    el("div", { class: "ro-head" }, "Energy above the cutoff", el("span", {}, `modes outside the ${manifest.stats.n_kept_modes} kept, sum of both paths, all 32 channels`)),
    bar("Spectral path", sShare, "spectral", sShare === 0 ? "exactly 0" : `${sShare.toExponential(1)} of it: rounding error`),
    bar("Local path", lShare, "local", "all of it"),
    el("p", { class: "ro-foot" }, `Truncation keeps ${pct(stats.truncation_kept_excl_mean, 1)} of this layer input's energy (excluding the mean). The remaining ${pct(1 - stats.truncation_kept_excl_mean, 1)} is fine detail and edges that only the local path can pass on.`),
  );

  root.append(el("div", { class: "layer-body" }, diagram, readout), legend, after);
  if (sample.live?.note) root.append(el("p", { class: "fcap", style: "text-align:left" }, sample.live.note));
}

function bar(label, share, path, note) {
  const shown = share < 0.0005 ? "0.0%" : pct(share, 1);
  return el("div", { class: `ro-bar p-${path}` },
    el("span", { class: "ro-lab" }, label),
    el("span", { class: "ro-track" }, el("span", { class: "ro-fill", style: `width:${Math.max(0, Math.min(1, share)) * 100}%` })),
    el("span", { class: "ro-val mono" }, shown),
    el("span", { class: "ro-note" }, note));
}
