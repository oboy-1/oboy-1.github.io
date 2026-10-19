// The whole network as an animated 3D scene, in the style of the CNN explainers: grid slabs going in and out.
// One Fourier layer is drawn once and labelled "× 4". The heart is the spectral path, shown as four stations
// (FFT, keep the slow waves, scale + shift each wave, inverse FFT); the lift and the local path are 1×1 convolutions
// and are drawn as a window sliding across the grid. Slabs are neutral grids: real data lives in the 2D panels.
import * as THREE from "../../vendor/three/three.module.min.js";

const SP = 0xea580c;

// ---- light / dark: the scene is built for one theme and rebuilt when the page switches
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t === "dark" || (t !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
};
const LABEL_DARK = { "#25252b": "#e8e8ec", "#c2410c": "#fb923c", "#52525b": "#a9a9b3", "#475467": "#cbd5e1", "#667085": "#9aa4b5", "#334155": "#cbd5e1" };
function pal() {
  const dark = isDark();
  return dark ? {
    dark, lo: 0x9fb3cf, edge: 0x9ca3af, frame: 0x667085, mute: 0x5a524c, needle: 0xe5e7eb, qbox: 0x4a2e22, plus: 0x3f4654,
    neutral: 0x9ca3af, ground: 0x0d0f13, groundOp: 0.5,
    slabFill: "rgba(40,44,54,0.92)", slabLine: "#4b5563", slabEdge: "#9ca3af",
    cellFill: "#2b221c", cellLine: "#7a4a2e", cellA: "#fdba74", cellB: "#cbd5e1",
    label: (c) => LABEL_DARK[c] ?? c,
  } : {
    dark, lo: 0x3b4a5e, edge: 0x6b7280, frame: 0x98a2b3, mute: 0xcfc3bb, needle: 0x1f2937, qbox: 0xffe3d1, plus: 0xffffff,
    neutral: 0x667085, ground: 0xe8ebf0, groundOp: 0.35,
    slabFill: "rgba(250,251,253,0.9)", slabLine: "#a3a9b3", slabEdge: "#6b7280",
    cellFill: "#fff8f2", cellLine: "#e6b28f", cellA: "#7c2d12", cellB: "#52525b",
    label: (c) => c,
  };
}
let PAL = pal();
const N = 16, KEEP = 4;                    // spectrum cells per side; kept block per side (1/4, like 16 of 64)
const DECK_N = 32, GAP = 0.1, SLAB = 3.2, G = 8; // 32 channel sheets; slab grid is G×G
const FY = 5.6;                            // height of the spectral path
const X_IN = -12, X_L0 = 0, X_L1 = 40.6;     // input slab, deck before the layer, deck after it
const XP = 21;                             // first add: spectral + local
const CYCLE = 13;                          // seconds for one pass of the spectral path

// camera presets: target, distance, yaw, pitch
const P = (t, d, yaw = -0.5, pitch = 0.36) => ({ t, d, yaw, pitch });
const VIEWS = {
  all: P([19.5, 2, 0], 69, -0.38, 0.36),
  allR: P([19.5, 2.2, 0], 44, -0.4, 0.3),
  hero: P([17, 1.5, 0], 70, -0.55, 0.3),
  lift: P([-5.2, 0.6, 0], 19.5, -0.6, 0.3),
  wide: P([12, 2, 0], 36, -0.4, 0.4),
  layer: P([12.5, 2, 0], 34, -0.45, 0.42),
  block: P([28, -0.9, 0], 24, -0.18, 0.2),
  layerS: P([13.5, 2.2, 0], 37, -0.42, 0.34),
  layerR: P([11.5, 2.3, 0], 35, -0.4, 0.3),
  spectral: P([10.5, 4.6, 0], 25, -0.35, 0.55),
  spec12: P([8, 4.6, 0], 14.5, -0.3, 0.55),
  spec23: P([13, 5.4, 0], 15.5, -0.3, 0.55),
  spec34: P([17.6, 4.9, 0], 16.5, -0.4, 0.34),
  local: P([12, -4.4, 0], 17, -0.45, 0.4),
  out: P([49, -0.4, 0], 18, -0.5, 0.3),
};
const CHIPS = [["Whole network", "all"], ["Lift", "lift"], ["One layer", "layer"], ["Spectral path", "spectral"], ["Local path", "local"], ["MLP & skip", "block"], ["Output", "out"]];
// which stretch of the spectral timeline each focus replays
const WINDOWS = { spec12: [0, 0.42], spec23: [0.36, 0.74], spec34: [0.68, 1], spectral: [0, 1] };

const hash = (a, b, s = 0) => { const v = Math.sin(a * 127.1 + b * 311.7 + s * 74.7) * 43758.5453; return v - Math.floor(v); };
const ease = (t) => t * t * (3 - 2 * t);
const seg = (p, a, b) => ease(Math.min(1, Math.max(0, (p - a) / (b - a))));

function makeLabel(text, size = 1, color = "#25252b") {
  const c = document.createElement("canvas");
  const g0 = c.getContext("2d");
  g0.font = "700 50px 'Source Sans 3', sans-serif";
  const w = Math.max(640, Math.ceil(g0.measureText(text).width) + 40);
  c.width = w; c.height = 112;
  const g = c.getContext("2d");
  g.font = "700 50px 'Source Sans 3', sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = PAL.label(color); g.fillText(text, w / 2, 56);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthTest: false }));
  s.scale.set(6.4 * size * 1.3 * (w / 640), 1.12 * size * 1.3, 1);
  s.userData.base = s.scale.clone();
  s.renderOrder = 20;
  return s;
}

function gridTex(n = G, fill = PAL.slabFill, line = PAL.slabLine) {
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = fill; g.fillRect(0, 0, 256, 256);
  g.strokeStyle = line; g.lineWidth = 2;
  for (let i = 0; i <= n; i++) { const p = (i / n) * 256; g.beginPath(); g.moveTo(p, 0); g.lineTo(p, 256); g.moveTo(0, p); g.lineTo(256, p); g.stroke(); }
  g.lineWidth = 7; g.strokeStyle = PAL.slabEdge; g.strokeRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

function valueTex(rows, size = KEEP) {
  const px = 160, c = document.createElement("canvas"); c.width = c.height = px * size;
  const g = c.getContext("2d");
  g.fillStyle = PAL.cellFill; g.fillRect(0, 0, c.width, c.height);
  g.textAlign = "center"; g.textBaseline = "middle";
  for (let a = 0; a < size; a++) for (let b = 0; b < size; b++) {
    const v = rows[a][b];
    g.strokeStyle = PAL.cellLine; g.lineWidth = 3; g.strokeRect(a * px + 1.5, b * px + 1.5, px - 3, px - 3);
    g.fillStyle = PAL.cellA; g.font = "700 44px 'Source Sans 3', sans-serif"; g.fillText(v[0], a * px + px / 2, b * px + px * 0.5);
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

function plainGridTex() {
  const px = 16, c = document.createElement("canvas"); c.width = c.height = px * N;
  const g = c.getContext("2d");
  g.fillStyle = PAL.dark ? "#22262e" : "#fbfbfc"; g.fillRect(0, 0, c.width, c.height);
  g.strokeStyle = PAL.dark ? "#3f4551" : "#c4c8cf"; g.lineWidth = 1.5;
  for (let i = 0; i <= N; i++) { g.beginPath(); g.moveTo(i * px, 0); g.lineTo(i * px, c.height); g.moveTo(0, i * px); g.lineTo(c.width, i * px); g.stroke(); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

let inst = null;
let lastSpec = null;
export function setArchSpectrum(spec) {
  lastSpec = spec;
  inst?.setSpectrum(spec);
}
export function getArch() {
  if (inst && inst.dark !== isDark()) { inst.dispose(); inst = null; }
  return inst ?? (inst = build());
}
/** Calls cb when the page switches between light and dark, so the caller can redraw. */
export function onThemeChange(cb) {
  new MutationObserver(cb).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", cb);
}

function build() {
  PAL = pal();
  const LO = PAL.lo;
  const host = document.createElement("div");
  host.className = "arch3d";
  if (PAL.dark) host.dataset.dark = "1";
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2.5, window.devicePixelRatio || 1));
  host.append(renderer.domElement);
  const scene = new THREE.Scene();
  const root = new THREE.Group(); scene.add(root);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xdfe3ea, 0.95));
  const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(-10, 24, 16); scene.add(sun);
  const camera = new THREE.PerspectiveCamera(34, 2, 0.1, 400);

  const add = (...o) => o.forEach((x) => root.add(x));
  // Labels are real HTML text placed over the canvas each frame, so they stay sharp at any zoom.
  const labels = [];
  const lblLayer = document.createElement("div"); lblLayer.className = "arch3d-labels"; host.append(lblLayer);
  const put = (text, x, y, z = 0, size = 0.62, color = "#25252b", group = null) => {
    const d = document.createElement("div"); d.className = `arch3d-lbl${text === "+" ? " bare" : ""}`;
    d.textContent = text; d.style.color = PAL.label(color); lblLayer.append(d);
    const L = { d, pos: new THREE.Vector3(x, y, z), size, group, fs: 0, op: -1 };
    labels.push(L); return L;
  };
  const _v = new THREE.Vector3();
  function placeLabels(act) {
    const w = host.clientWidth, h = host.clientHeight;
    for (const L of labels) {
      _v.copy(L.pos).project(camera);
      const dist = camera.position.distanceTo(L.pos);
      const off = _v.z > 1 || Math.abs(_v.x) > 1.25 || Math.abs(_v.y) > 1.25;
      if (off) { if (L.shown !== false) { L.d.style.display = "none"; L.shown = false; } continue; }
      if (L.shown === false) { L.d.style.display = ""; L.shown = true; }
      const fs = Math.round(L.size * 21 * Math.min(1.3, Math.max(0.8, 26 / dist)) * 4) / 4;
      if (fs !== L.fs) { L.d.style.fontSize = `${fs}px`; L.fs = fs; }
      L.d.style.transform = `translate(${((_v.x * 0.5 + 0.5) * w).toFixed(1)}px, ${((-_v.y * 0.5 + 0.5) * h).toFixed(1)}px) translate(-50%, -50%)`;
      let op = L.group ? (L.group === act ? 1 : 0.5) : 1;
      if (L.size <= 0.5) op *= host.dataset.size === "ref" || host.dataset.size === "mid" ? 0 : Math.min(1, Math.max(0, (34 - dist) / 8)); // fine print only when zoomed in and large
      if (op !== L.op) { L.d.style.opacity = op; L.op = op; }
    }
  }
  const slabTex = gridTex();
  const slab = (size, x, y, z = 0, opacity = 1) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ map: slabTex, side: THREE.DoubleSide, transparent: true, opacity, depthWrite: false }));
    m.rotation.y = -Math.PI / 2; m.position.set(x, y, z);
    return m;
  };
  const deck = (x, y = 0) => {
    const g = new THREE.Group(); g.position.set(x, y, 0);
    for (let j = 0; j < DECK_N; j++) g.add(slab(SLAB, j * GAP - (DECK_N - 1) * GAP / 2, 0, 0, 0.88));
    g.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry((DECK_N - 1) * GAP + 0.04, SLAB, SLAB)), new THREE.LineBasicMaterial({ color: PAL.edge })));
    root.add(g); return g;
  };

  // ---- tubes with travelling dots: how data flows
  const flows = [];
  function path(points, color, { dots = 3, speed = 0.12 } = {}) {
    const vs = points.map((p) => new THREE.Vector3(...p));
    const curve = new THREE.CurvePath();
    for (let i = 0; i < vs.length - 1; i++) curve.add(new THREE.LineCurve3(vs[i], vs[i + 1]));
    add(new THREE.Mesh(new THREE.TubeGeometry(curve, vs.length * 8, 0.055, 6, false), new THREE.MeshBasicMaterial({ color })));
    const a = vs[vs.length - 2], b = vs[vs.length - 1];
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.55, 14), new THREE.MeshBasicMaterial({ color }));
    cone.position.copy(b); cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize()); add(cone);
    const list = [];
    for (let k = 0; k < dots; k++) {
      const d = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 8), new THREE.MeshBasicMaterial({ color }));
      add(d); list.push({ d, off: k / dots });
    }
    flows.push({ curve, list, speed });
  }

  // ================= input -> lift (a 1×1 window sliding over the grid) -> deck
  const XL = X_IN;
  [0, 1, 2].forEach((k) => add(slab(3.6, XL + k * 0.5, 0, 0, 0.92)));
  add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1.04, 3.6, 3.6)), new THREE.LineBasicMaterial({ color: PAL.edge })).translateX(XL + 0.5));
  put("input: a(x), x, y", XL + 0.5, -2.8, 0, 0.62);
  put("lift P: a 1×1 window slides over the grid", X_L0 - 6.2, 3.1, 0, 0.55, "#c2410c");
  put("32 channels", X_L0, -2.8, 0, 0.62);
  deck(X_L0);
  const win1 = new THREE.Mesh(new THREE.BoxGeometry(1.3, 3.6 / G, 3.6 / G), new THREE.MeshBasicMaterial({ color: SP, transparent: true, opacity: 0.95 }));
  add(win1);
  const outCol = new THREE.Mesh(new THREE.BoxGeometry((DECK_N - 1) * GAP + 0.12, SLAB / G, SLAB / G), new THREE.MeshBasicMaterial({ color: SP, transparent: true, opacity: 0.55, depthWrite: false }));
  add(outCol);
  const beam = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(1, 0, 0)]), new THREE.LineBasicMaterial({ color: SP, transparent: true, opacity: 0.6 }));
  add(beam);

  // ================= the Fourier layer (drawn once, "× 4")
  const fr = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(X_L1 - X_L0 - 2.4, 15.4)), new THREE.LineDashedMaterial({ color: PAL.frame, dashSize: 0.5, gapSize: 0.35 }));
  fr.computeLineDistances(); fr.position.set((X_L0 + X_L1) / 2 - 0.4, 2.5, -4.2); add(fr);
  put("Fourier layer  × 4", (X_L0 + X_L1) / 2 - 0.4, 11.2, -4.2, 0.9, "#475467");
  put("same structure four times, each with its own learned weights", (X_L0 + X_L1) / 2 - 0.4, 10.1, -4.2, 0.48, "#667085");

  // spectrum data
  const cells = [];
  for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
    const kx = a - N / 2, ky = b - N / 2, r = Math.hypot(kx + 0.5, ky + 0.5);
    const amp = 3.0 * Math.exp(-Math.pow(r / 2.4, 1.15)) * (0.65 + 0.35 * hash(a, b, 1)) + 0.05;
    const kept = Math.abs(kx + 0.5) <= KEEP / 2 && Math.abs(ky + 0.5) <= KEEP / 2;
    cells.push({ a, b, amp, kept, ph: (hash(a, b, 6) - 0.5) * 2 * Math.PI, R: 0.45 + 1.25 * hash(a, b, 10), th: (hash(a, b, 4) - 0.5) * 2.2 });
  }
  const keptCells = cells.filter((c) => c.kept).sort((p, q) => p.a - q.a || p.b - q.b);

  const s1x = X_L0 + 5, s1 = 3.8, c1 = s1 / N;
  const s2x = X_L0 + 10.5, s3x = X_L0 + 15.2, ssz = 3.2, scs = ssz / KEEP, s4x = X_L0 + 19.3;
  const barMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, metalness: 0.02 });
  const dummy = new THREE.Object3D();
  const colSp = new THREE.Color(SP), colMute = new THREE.Color(PAL.mute);

  // station 1: every wave
  const fl = new THREE.Mesh(new THREE.PlaneGeometry(s1, s1), new THREE.MeshBasicMaterial({ map: plainGridTex() }));
  fl.rotation.x = -Math.PI / 2; fl.position.set(s1x, FY, 0); add(fl);
  const bars1 = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), barMat, N * N);
  bars1.instanceMatrix.setUsage(THREE.DynamicDrawUsage); add(bars1);
  const kw = KEEP * c1, kcx = s1x - s1 / 2 + (N / 2) * c1, kcz = -s1 / 2 + (N / 2) * c1;
  const foot = new THREE.Mesh(new THREE.PlaneGeometry(kw, kw), new THREE.MeshBasicMaterial({ color: SP, transparent: true, opacity: 0.3, side: THREE.DoubleSide }));
  foot.rotation.x = -Math.PI / 2; foot.position.set(kcx, FY + 0.03, kcz); add(foot);
  const kframe = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(kw, 3.3, kw)), new THREE.LineBasicMaterial({ color: SP }));
  kframe.position.set(kcx, FY + 1.65, kcz); add(kframe);
  put("1  FFT: the image as waves", s1x, FY + 4.5, 0, 0.58, "#c2410c", 1);
  put("each cell pools 4×4 nearby waves · bar = strength, square-root scaled", s1x, FY - 0.05, s1 / 2 + 1.35, 0.42, "#52525b", 1);

  // axes for station 1: two floor directions = how fast the wave wiggles, height = how strong
  const axMat = new THREE.LineBasicMaterial({ color: PAL.frame });
  const axis = (from, to) => {
    const A = new THREE.Vector3(...from), B = new THREE.Vector3(...to);
    add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([A, B]), axMat));
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.3, 10), new THREE.MeshBasicMaterial({ color: PAL.frame }));
    cone.position.copy(B); cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize()); add(cone);
  };
  const zf = s1 / 2 + 0.25, xl = s1x - s1 / 2 - 0.25;
  axis([kcx, FY, zf], [s1x + s1 / 2 + 0.2, FY, zf]);
  axis([xl, FY, kcz], [xl, FY, s1 / 2 + 0.2]);
  axis([xl, FY, -s1 / 2], [xl, FY + 3.4, -s1 / 2]);
  put("faster across →", s1x + s1 / 2 - 0.4, FY - 0.05, zf + 0.5, 0.4, "#52525b", 1);
  put("faster down →", xl - 0.5, FY + 0.35, s1 / 2 - 0.2, 0.4, "#52525b", 1);
  put("stronger", xl - 0.1, FY + 3.7, -s1 / 2, 0.42, "#52525b", 1);
  put("slowest", kcx, FY - 0.05, zf + 0.5, 0.38, "#52525b", 1);

  // stations 2 and 3: the kept block as a table of numbers, with bars and phase needles
  const mkStation = (x, fn) => {
    const rows = Array.from({ length: KEEP }, () => Array(KEEP));
    keptCells.forEach((c, k) => { rows[Math.floor(k / KEEP)][k % KEEP] = fn(c); });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(ssz, ssz), new THREE.MeshBasicMaterial({ map: valueTex(rows), transparent: true }));
    m.rotation.x = -Math.PI / 2; m.position.set(x, FY, 0); add(m);
    const ol = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(ssz, ssz)), new THREE.LineBasicMaterial({ color: SP }));
    ol.position.set(x, FY + 0.01, 0); ol.rotation.x = -Math.PI / 2; add(ol);
    const bars = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), barMat, KEEP * KEEP);
    bars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    add(bars);
    return { x, m, ol, bars };
  };
  const st2 = mkStation(s2x, (c) => [`${(c.amp * 3.3).toFixed(1)}`, ""]);
  const st3 = mkStation(s3x, (c) => [`${(c.amp * c.R * 3.3).toFixed(1)}`, ""]);
  put("2  keep only the slow waves", s2x, FY + 2.7, 0, 0.55, "#c2410c", 2);
  put("3  × R: a learned weight for each wave", s3x + 0.3, FY + 3.7, 0, 0.55, "#c2410c", 3);
  put("each cell: how strong that wave is", s2x, FY - 0.05, ssz / 2 + 0.7, 0.44, "#52525b", 2);
  put("R: learned, the same for every image", s3x, FY - 0.05, ssz / 2 + 0.7, 0.44, "#52525b", 3);
  const zl = (zs) => { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(kcx + kw / 2, FY, kcz + zs * kw / 2), new THREE.Vector3(s2x - ssz / 2, FY, zs * ssz / 2)]), new THREE.LineDashedMaterial({ color: SP, dashSize: 0.25, gapSize: 0.18 })); l.computeLineDistances(); return l; };
  const z1 = zl(1), z2 = zl(-1); add(z1, z2);

  // station 4: a grid again
  const s4 = slab(2.6, s4x, FY - 1.3, 0, 0.95); add(s4);
  const sheen = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 0.5), new THREE.MeshBasicMaterial({ color: SP, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false }));
  sheen.rotation.y = -Math.PI / 2; add(sheen);
  put("4  inverse FFT: a grid again", s4x + 0.6, FY - 3.4, 0, 0.55, "#c2410c", 4);

  // spectral flow
  path([[X_L0 + 1.9, 1.3, 0], [X_L0 + 1.9, FY, 0], [s1x - s1 / 2 - 0.5, FY, 0]], SP, { dots: 3 });
  path([[s2x + ssz / 2 + 0.2, FY, 0], [s3x - ssz / 2 - 0.5, FY, 0]], SP, { dots: 2 });
  path([[s3x + ssz / 2 + 0.2, FY, 0], [s4x - 0.9, FY, 0], [s4x - 0.9, FY - 1.3, 0], [s4x - 0.15, FY - 1.3, 0]], SP, { dots: 2 });
  path([[s4x, FY - 2.6, 0], [s4x, 0, 0], [XP - 0.6, 0, 0]], SP, { dots: 3 });

  // ---- local path: a window runs through all 32 channels at one pixel, writing one pixel of one sheet
  const lx = X_L0 + 11.6, ly = -4.9;
  deck(lx).position.y = ly;
  add(slab(SLAB, lx + 4.6, ly, 0, 0.95));
  put("local path: 1×1 conv, each pixel on its own", lx + 2.2, ly - 2.7, 0, 0.55, "#334155");
  put("mixes the 32 channels at that pixel", lx + 2.2, ly - 3.5, 0, 0.44, "#667085");
  const win2 = new THREE.Mesh(new THREE.BoxGeometry((DECK_N - 1) * GAP + 0.12, SLAB / G, SLAB / G), new THREE.MeshBasicMaterial({ color: LO, transparent: true, opacity: 0.85 }));
  win2.position.x = lx; add(win2);
  const lit = new THREE.Mesh(new THREE.BoxGeometry(0.5, SLAB / G, SLAB / G), new THREE.MeshBasicMaterial({ color: LO })); add(lit);
  const beam2 = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(1, 0, 0)]), new THREE.LineBasicMaterial({ color: LO, transparent: true, opacity: 0.6 })); add(beam2);
  path([[X_L0 + 1.9, -1.3, 0], [X_L0 + 1.9, ly, 0], [lx - 2.2, ly, 0]], LO, { dots: 3 });
  path([[lx + 6.6, ly, 0], [XP, ly, 0], [XP, -0.8, 0]], LO, { dots: 3 });

  // merge, then the rest of the layer: GELU, channel MLP, the gated skip coming back in, a second add, GELU
  const plusMesh = (x) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.55, 28, 18), new THREE.MeshStandardMaterial({ color: PAL.plus, roughness: 0.4 })); m.position.set(x, 0, 0); add(m); put("+", x, 0.02, 0, 0.9); return m; };
  const G1 = XP + 2.8, MA = XP + 5.7, MB = XP + 7.7, MC = XP + 9.7, P2 = XP + 12.9, G2 = XP + 15.7;
  plusMesh(XP);
  const gelu = (x, text) => {
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.62), new THREE.MeshStandardMaterial({ color: PAL.qbox, roughness: 0.5 }));
    m.position.set(x, 0, 0); m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: SP }))); add(m);
    put(text, x, 1.45, 0, 0.5, "#c2410c");
  };
  gelu(G1, "GELU");
  const mlpBox = (x, th) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(th, 2.2, 2.2), new THREE.MeshStandardMaterial({ color: PAL.qbox, roughness: 0.5, transparent: true, opacity: 0.95 }));
    m.position.set(x, 0, 0); m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: SP }))); add(m);
  };
  mlpBox(MA, 1.0); mlpBox(MB, 0.5); mlpBox(MC, 1.0);
  put("channel MLP: 32 → 16 → 32", MB, -2.0, 0, 0.5, "#c2410c");
  put("1×1 convs, each pixel on its own", MB, -2.8, 0, 0.5, "#667085");
  plusMesh(P2);
  put("add the input back", P2, 1.3, 0, 0.5, "#667085");
  gelu(G2, "GELU");
  put("(not on layer 4)", G2, -1.2, 0, 0.44, "#667085");
  path([[XP + 0.6, 0, 0], [G1 - 0.5, 0, 0]], PAL.neutral, { dots: 1 });
  path([[G1 + 0.55, 0, 0], [MA - 0.75, 0, 0]], PAL.neutral, { dots: 1 });
  path([[MA + 0.55, 0, 0], [MB - 0.4, 0, 0]], PAL.neutral, { dots: 1 });
  path([[MB + 0.3, 0, 0], [MC - 0.75, 0, 0]], PAL.neutral, { dots: 1 });
  path([[MC + 0.55, 0, 0], [P2 - 0.7, 0, 0]], PAL.neutral, { dots: 1 });
  path([[P2 + 0.7, 0, 0], [G2 - 0.55, 0, 0]], PAL.neutral, { dots: 1 });
  path([[G2 + 0.62, 0, 0], [X_L1 - 1.9, 0, 0]], PAL.neutral, { dots: 1 });
  // the skip: the layer's own input goes round the back, through the gate g, and comes in at the second add
  const gx = XP - 4.5;
  path([[X_L0 + 1.9, 0, 0], [X_L0 + 1.9, 0, -3.1], [P2, 0, -3.1], [P2, 0, -0.7]], PAL.neutral, { dots: 4, speed: 0.08 });
  const gate = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.0, 1.0), new THREE.MeshStandardMaterial({ color: PAL.qbox, roughness: 0.5 }));
  gate.position.set(gx, 0, -3.1); gate.add(new THREE.LineSegments(new THREE.EdgesGeometry(gate.geometry), new THREE.LineBasicMaterial({ color: SP }))); add(gate);
  put("g: one number per channel", gx, 1.3, -3.1, 0.5, "#c2410c");
  put("gated skip", gx, -1.25, -3.1, 0.5, "#667085");
  deck(X_L1);
  put("32 channels", X_L1, -2.8, 0, 0.62);
  const qx = X_L1 + 6.2;
  const qbox = new THREE.Mesh(new THREE.BoxGeometry(1.2, 2.2, 2.2), new THREE.MeshStandardMaterial({ color: PAL.qbox, roughness: 0.5 }));
  qbox.position.set(qx, 0, 0); qbox.add(new THREE.LineSegments(new THREE.EdgesGeometry(qbox.geometry), new THREE.LineBasicMaterial({ color: SP }))); add(qbox);
  put("Q: 1×1 conv, 32 → 1", qx, -2.5, 0, 0.58);
  path([[X_L1 + 2.0, 0, 0], [qx - 0.9, 0, 0]], PAL.neutral, { dots: 2 });
  const outX = qx + 6.6;
  path([[qx + 0.9, 0, 0], [outX - 0.5, 0, 0]], PAL.neutral, { dots: 2 });
  add(slab(3.6, outX, 0));
  put("output: u(x)", outX, 2.6, 0, 0.62);
  path([[XL + 2.2, 0, 0], [X_L0 - 2.4, 0, 0]], PAL.neutral, { dots: 2 });

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 60), new THREE.MeshBasicMaterial({ color: PAL.ground, transparent: true, opacity: PAL.groundOp }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(10, -6.6, 0); add(ground);

  // ---- animation
  let win = [0, 1], tt = 0, last = performance.now();

  function writeBars1(p) {
    for (let k = 0; k < cells.length; k++) {
      const c = cells[k];
      const sink = c.kept ? 0 : seg(p, 0.22, 0.34);
      const h = Math.max(0.001, c.amp * seg(p, 0, 0.16) * (1 - sink) * (1 - seg(p, 0.96, 1)));
      dummy.position.set(s1x - s1 / 2 + (c.a + 0.5) * c1, FY + h / 2, -s1 / 2 + (c.b + 0.5) * c1);
      dummy.scale.set(c1 * 0.62, h, c1 * 0.62); dummy.rotation.set(0, 0, 0); dummy.updateMatrix();
      bars1.setMatrixAt(k, dummy.matrix); bars1.setColorAt(k, c.kept ? colSp : colMute);
    }
    bars1.instanceMatrix.needsUpdate = true; if (bars1.instanceColor) bars1.instanceColor.needsUpdate = true;
  }
  function writeStation(st, mix, appear, fade) {
    keptCells.forEach((c, k) => {
      const a = Math.floor(k / KEEP), b = k % KEEP;
      const a1 = c.ampOut ?? c.amp * c.R;
      const h = Math.max(0.001, (c.amp + (a1 - c.amp) * mix) * appear * (1 - fade)) * 0.9;
      const px = st.x - ssz / 2 + (a + 0.5) * scs, pz = -ssz / 2 + (b + 0.5) * scs;
      dummy.position.set(px, FY + h / 2, pz); dummy.scale.set(scs * 0.5, h, scs * 0.5); dummy.rotation.set(0, 0, 0); dummy.updateMatrix();
      st.bars.setMatrixAt(k, dummy.matrix); st.bars.setColorAt(k, colSp);
    });
    st.bars.instanceMatrix.needsUpdate = true; if (st.bars.instanceColor) st.bars.instanceColor.needsUpdate = true;
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (!host.isConnected) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now; tt += dt;
    const pr = renderer.getPixelRatio();
    if (renderer.domElement.width !== Math.round(host.clientWidth * pr) || renderer.domElement.height !== Math.round(host.clientHeight * pr)) resize();

    const p = win[0] + ((tt % CYCLE) / CYCLE) * (win[1] - win[0]);
    writeBars1(p);
    const fade = seg(p, 0.96, 1);
    writeStation(st2, 0, seg(p, 0.3, 0.42), fade);
    writeStation(st3, seg(p, 0.5, 0.68), seg(p, 0.42, 0.5), fade);
    st2.m.material.opacity = seg(p, 0.28, 0.4) * (1 - fade); st3.m.material.opacity = seg(p, 0.42, 0.5) * (1 - fade);
    z1.visible = z2.visible = p > 0.28 && p < 0.98;
    s4.material.opacity = 0.3 + 0.65 * seg(p, 0.7, 0.8);
    sheen.visible = p > 0.72 && p < 0.92;
    sheen.position.set(s4x + 0.02, FY - 1.3 + (seg(p, 0.72, 0.92) - 0.5) * 2.2, 0);
    const act = p < 0.2 ? 1 : p < 0.42 ? 2 : p < 0.74 ? 3 : 4;

    // sliding 1×1 windows, raster order over the grid
    const scan = (rate) => { const n = Math.floor(tt * rate) % (G * G); return [Math.floor(n / G), n % G]; };
    const [r1, q1] = scan(5);
    const cell = 3.6 / G, c2 = SLAB / G;
    win1.position.set(XL + 0.5, (G / 2 - r1 - 0.5) * cell, (q1 + 0.5 - G / 2) * cell);
    outCol.position.set(X_L0, (G / 2 - r1 - 0.5) * c2, (q1 + 0.5 - G / 2) * c2);
    const bp = beam.geometry.attributes.position;
    bp.setXYZ(0, XL + 1.2, (G / 2 - r1 - 0.5) * cell, (q1 + 0.5 - G / 2) * cell);
    bp.setXYZ(1, X_L0 - (DECK_N - 1) * GAP / 2, (G / 2 - r1 - 0.5) * c2, (q1 + 0.5 - G / 2) * c2); bp.needsUpdate = true;
    const [r2, q2] = scan(3.4);
    const yy = ly + (G / 2 - r2 - 0.5) * c2, zz = (q2 + 0.5 - G / 2) * c2;
    win2.position.set(lx, yy, zz); lit.position.set(lx + 4.6, yy, zz);
    const b2 = beam2.geometry.attributes.position;
    b2.setXYZ(0, lx + (DECK_N - 1) * GAP / 2 + 0.1, yy, zz); b2.setXYZ(1, lx + 4.6, yy, zz); b2.needsUpdate = true;

    for (const f of flows) for (const d of f.list) d.d.position.copy(f.curve.getPoint((tt * f.speed + d.off) % 1));

    if (host.dataset.size === "hero" && !drag) camT.yaw = -0.55 + Math.sin(tt * 0.22) * 0.28;
    for (const k of ["d", "yaw", "pitch"]) cam[k] += (camT[k] - cam[k]) * 0.09;
    cam.t.lerp(camT.t, 0.09);
    const cp = Math.cos(cam.pitch);
    camera.position.set(cam.t.x + cam.d * Math.sin(cam.yaw) * cp, cam.t.y + cam.d * Math.sin(cam.pitch), cam.t.z + cam.d * Math.cos(cam.yaw) * cp);
    camera.lookAt(cam.t);
    camera.updateMatrixWorld();
    placeLabels(act);
    renderer.render(scene, camera);
  }

  // ---- camera + controls
  const cam = { t: new THREE.Vector3(...VIEWS.all.t), d: VIEWS.all.d, yaw: VIEWS.all.yaw, pitch: VIEWS.all.pitch };
  const camT = { t: cam.t.clone(), d: cam.d, yaw: cam.yaw, pitch: cam.pitch };
  function go(name) { const v = VIEWS[name] ?? VIEWS.all; camT.t.set(...v.t); camT.d = v.d; camT.yaw = v.yaw; camT.pitch = v.pitch; }
  function resize() {
    const w = host.clientWidth || 600, h = host.clientHeight || 300;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  let drag = null;
  const cv = renderer.domElement;
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 || e.button === 1 }; cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; });
  cv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const s = camT.d * 0.0016;
      camT.t.addScaledVector(new THREE.Vector3(Math.cos(camT.yaw), 0, -Math.sin(camT.yaw)), -dx * s); camT.t.y += dy * s;
      cam.t.copy(camT.t);
    } else {
      camT.yaw -= dx * 0.006; camT.pitch = Math.min(1.3, Math.max(-0.2, camT.pitch + dy * 0.005));
      cam.yaw = camT.yaw; cam.pitch = camT.pitch;
    }
  });
  const up = () => { drag = null; cv.style.cursor = "grab"; };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  cv.addEventListener("wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) return; // plain scroll keeps scrolling the story
    e.preventDefault(); camT.d = Math.min(80, Math.max(6, camT.d * (1 + e.deltaY * 0.002)));
  }, { passive: false });

  // overlay: view chips, zoom buttons, hint
  const ui = document.createElement("div"); ui.className = "arch3d-ui";
  const chips = document.createElement("div"); chips.className = "arch3d-chips";
  const chipEls = CHIPS.map(([text, name]) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = text; b.dataset.view = name;
    b.addEventListener("click", () => { go(name); setActive(name); win = [0, 1]; });
    chips.append(b); return b;
  });
  const zoom = document.createElement("div"); zoom.className = "arch3d-zoom";
  for (const [t, f] of [["+", 0.8], ["−", 1.25]]) {
    const b = document.createElement("button"); b.type = "button"; b.textContent = t; b.setAttribute("aria-label", t === "+" ? "Zoom in" : "Zoom out");
    b.addEventListener("click", () => { camT.d = Math.min(80, Math.max(6, camT.d * f)); });
    zoom.append(b);
  }
  const tip = document.createElement("div"); tip.className = "arch3d-tip"; tip.textContent = "drag to rotate · shift-drag to pan · +/− or ⌘/ctrl-scroll to zoom";
  ui.append(chips, zoom, tip); host.append(ui);
  function setActive(name) { chipEls.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === name))); }

  // ---- real numbers: replace the illustrative spectrum with the actual FFT of one channel
  const headLabel = put("", s1x, FY + 5.3, 0, 0.5, "#52525b", 1);
  function relabel(L, text) { L.d.textContent = text; L.d.style.visibility = text ? "" : "hidden"; }
  function setSpectrum(spec) {
    if (!spec) return;
    const top = 3.0;
    for (const c of cells) {
      const i = c.a * N + c.b;
      c.amp = top * Math.sqrt(spec.inAmp[i] / spec.vmax) + 0.02;
      c.ampOut = top * Math.sqrt(spec.outAmp[i] / spec.vmax) + 0.02;
    }
    const swap = (st, fn) => {
      const rows = Array.from({ length: KEEP }, () => Array(KEEP));
      keptCells.forEach((c, k) => { rows[Math.floor(k / KEEP)][k % KEEP] = fn(c); });
      st.m.material.map?.dispose(); st.m.material.map = valueTex(rows); st.m.material.needsUpdate = true;
    };
    const rel = (arr, c) => (arr[c.a * N + c.b] / spec.vmax).toFixed(2);
    swap(st2, (c) => [rel(spec.inAmp, c), ""]);
    swap(st3, (c) => [rel(spec.outAmp, c), ""]);
    relabel(headLabel, spec.label || "");
  }
  if (lastSpec) setSpectrum(lastSpec);

  let raf = requestAnimationFrame(frame);
  let lastView = null;
  return {
    el: host,
    dark: PAL.dark,
    setSpectrum,
    dispose() { cancelAnimationFrame(raf); renderer.dispose(); host.remove(); },
    setView(view) {
      if (view === lastView) return;
      lastView = view;
      go(view);
      setActive(chipEls.some((b) => b.dataset.view === view) ? view : view.startsWith("spec") ? "spectral" : "all");
      win = WINDOWS[view] ?? [0, 1];
      tt = 0;
    },
  };
}
