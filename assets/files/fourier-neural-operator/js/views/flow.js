// Zoom level 1: a(x) -> P -> Layer 1..4 -> Q -> u_hat, with u pinned at the right.
import { el, paintField, paintInput, outScale, isNS, words, fmt, pct } from "../render.js";
import { resembleCounts, THRESH } from "../story.js";

const STAGES = [
  { key: "lifted", field: "lifted", title: "Lifted", sub: "P(a), 32 ch", layer: null },
  { key: "layer1", field: "layer1.out", title: "Layer 1", sub: "output", layer: 1 },
  { key: "layer2", field: "layer2.out", title: "Layer 2", sub: "output", layer: 2 },
  { key: "layer3", field: "layer3.out", title: "Layer 3", sub: "output", layer: 3 },
  { key: "layer4", field: "layer4.out", title: "Layer 4", sub: "output", layer: 4 },
];

const CRITERIA = [
  ["corr_target", "|corr| with target u"],
  ["corr_input", "|corr| with input a"],
  ["variance", "spatial variance"],
];

function arrow() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 22 10");
  svg.setAttribute("class", "arrow");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = '<path d="M1 5h18M15 1.5 19.5 5 15 8.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>';
  return svg;
}

function op(label, mark) {
  return el("div", { class: `op${label ? " named" : ""}${mark ? " mark" : ""}`, "aria-hidden": "true" },
    label ? el("span", {}, label) : null, arrow(), mark ? el("span", { class: "op-mark" }, mark) : null);
}

export function renderFlow(root, app, opts = {}) {
  const { manifest, sample, state } = app;
  const counts = opts.counts ? resembleCounts(manifest) : null;
  const res = sample.res;
  root.replaceChildren();

  const crit = el("select", { id: "crit", "aria-label": "Rank channels by" },
    CRITERIA.map(([v, label]) => el("option", { value: v, selected: v === state.criterion }, label)));
  crit.addEventListener("change", () => app.set({ criterion: crit.value }));
  const allBtn = el("button", { class: "btn", id: "show-all", "aria-pressed": String(state.showAll) },
    state.showAll ? "Show top 8" : "Show all 32");
  allBtn.addEventListener("click", () => app.set({ showAll: !state.showAll }));
  root.append(el("div", { class: "toolbar" },
    el("label", {}, "Rank channels by", crit),
    allBtn,
    el("span", { class: "mono", style: "color:var(--ink-3)" },
      state.showAll ? "all 32 channels, ranked row by row" : "top 8 of 32 channels per column")));

  const flow = el("div", { class: "flow", role: "list" });
  const live = sample.live;
  const io = live
    ? { input: live.input, prediction: live.prediction, target: live.target }
    : sample.io[res];

  // input
  const w = words();
  const inCv = el("canvas", { "aria-label": `Input ${w.input}` });
  paintInput(inCv, io.input, res);
  flow.append(el("div", { class: "fcol io-tile", role: "listitem" },
    el("div", { class: "fhead" }, `Input ${w.inputSym}(x)`, el("span", { class: "sub" }, `${w.input}, ${res}×${res}`)),
    el("div", { class: "tile" }, inCv),
    el("div", { class: "fcap" }, live ? `model re-run live at ${res}×${res}` : isNS() ? "red and blue push opposite ways" : "two rock types")));
  flow.append(op("P"));

  const nTiles = state.showAll ? 32 : 8;
  const cols = state.showAll ? 4 : 1;
  STAGES.forEach((st, si) => {
    const order = manifest.rankings[st.key][state.criterion].order.slice(0, nTiles);
    const live = sample.live;
    const fields = live ? (st.layer ? live.layers[st.layer - 1].out : live.lifted) : sample.fields[st.field];
    const stats = live ? { r_in: live.r_in[st.key], r_u: live.r_u?.[st.key] } : sample.entry.channel_stats[st.key];
    const tiles = el("div", { class: `tiles${state.showAll ? " all" : ""}`, style: `--cols:${cols}` });
    for (const c of order) {
      const cv = el("canvas");
      const m = paintField(cv, fields[c], sample.fieldRes ?? 64);
      const t = el("div", { class: "tile", dataset: { ch: c, stage: st.key, max: m } }, cv,
        state.showAll ? null : el("span", { class: "idx" }, c));
      t.addEventListener("pointerenter", (e) => app.hoverChannel(c, e, () => tipText(app, st, c, m, stats)));
      t.addEventListener("pointermove", (e) => app.moveTip(e));
      t.addEventListener("pointerleave", () => app.hoverChannel(null));
      tiles.append(t);
    }
    const head = el("div", { class: "fhead" }, st.title, el("span", { class: "sub" }, st.sub),
      counts ? el("span", { class: "count-badge", title: `${counts[si]} of 32 channels reach |r| >= ${THRESH} with the true ${words().output}` }, `${counts[si]} / 32`) : null);
    let col;
    if (st.layer) {
      col = el("div", { class: "fcol layer-col", role: "listitem", tabindex: "0", "aria-label": `Open layer ${st.layer}` },
        head, tiles, el("span", { class: "open" }, "open layer →"));
      const open = () => app.goToStep("split", { layer: st.layer, channel: null, node: null });
      col.addEventListener("click", open);
      col.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    } else {
      col = el("div", { class: "fcol", role: "listitem" }, head, tiles);
    }
    flow.append(col);
    flow.append(counts && si === 2 ? op(null, "handoff") : op(si === STAGES.length - 1 ? "Q" : null));
  });

  // prediction + target share one sequential scale
  const scale = outScale(io.prediction, ...(io.target ? [io.target] : []));
  const vmax = scale.vmax;
  const predCv = el("canvas"), tgtCv = el("canvas");
  paintField(predCv, io.prediction, res, scale);
  if (io.target) paintField(tgtCv, io.target, res, scale);
  const err = live ? live.error : sample.error?.[res];
  flow.append(el("div", { class: "fcol io-tile", role: "listitem" },
    el("div", { class: "fhead" }, isNS() ? "Prediction ω̂(x)" : "Prediction û(x)", el("span", { class: "sub" }, w.output)),
    el("div", { class: "tile" }, predCv),
    el("div", { class: "fcap" }, err != null ? `rel. L2 error ${pct(err, 1)}` : "no ground truth")));
  flow.append(el("div", { class: "op", "aria-hidden": "true" }, el("span", {}, "vs")));
  flow.append(el("div", { class: "fcol io-tile", role: "listitem" },
    el("div", { class: "fhead" }, isNS() ? "True ω(x)" : "True u(x)", el("span", { class: "sub" }, "solver")),
    io.target ? el("div", { class: "tile" }, tgtCv) : el("div", { class: "tile empty" }, el("span", {}, "no solver run for a drawn map")),
    el("div", { class: "fcap" }, isNS() ? `same scale, ±${fmt(vmax, 2)}` : `same scale, 0–${fmt(vmax, 2)}`)));

  root.append(el("div", { class: "flow-scroll" }, flow));
  if (sample.fieldNote) root.append(el("p", { class: "fcap", style: "text-align:left" }, sample.fieldNote));
}

function tipText(app, st, c, m, stats) {
  const s = app.manifest.rankings[st.key][app.state.criterion].score[c];
  const lines = [`<b>${st.title} · ch${String(c).padStart(2, "0")}</b>`, `color scale ±${fmt(m, 3)}`];
  lines.push(`r with input a  ${fmt(stats.r_in[c], 2)}`);
  lines.push(stats.r_u ? `r with target u ${fmt(stats.r_u[c], 2)}` : "r with target u  — (no ground truth)");
  lines.push(`rank score ${fmt(s, 3)} <span style="opacity:.7">(mean over 4 samples)</span>`);
  return lines.join("<br>");
}
