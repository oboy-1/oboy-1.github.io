// Loads the static bundle written by export_web.py. Decodes atlases into Float32Arrays.
// Encodings are documented in manifest.encoding; every number shown in the UI comes from here.

// Two bundles: data/ (Darcy) and data-ns/ (Navier-Stokes). Each manifest remembers its folder.
const BASE = new URL("../data/", import.meta.url);
const baseOf = (m) => (m?.__base ? new URL(m.__base) : BASE);
const TILE = 64;
// Binary files are .bin locally; a host that refuses that extension can set
// document.documentElement.dataset.binExt (e.g. ".wasm") before this module loads.
const BIN = (typeof document !== "undefined" && document.documentElement.dataset.binExt) || ".bin";

export async function loadManifest(dir = "data") {
  const base = new URL(`../${dir}/`, import.meta.url);
  const res = await fetch(new URL("manifest.json", base));
  if (!res.ok) throw new Error(`${dir}/manifest.json: HTTP ${res.status}`);
  const m = await res.json();
  m.__base = base.href;
  return m;
}

async function fetchBytes(path, base = BASE) {
  const res = await fetch(new URL(path, base));
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

async function decodeGray(path, base) {
  const blob = new Blob([await fetchBytes(path, base)], { type: "image/png" });
  // colorSpaceConversion "none" keeps the 8-bit codes exactly as written.
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  return { data: ctx.getImageData(0, 0, bmp.width, bmp.height).data, width: bmp.width };
}

/** Atlas -> array of 32 Float32Array(64*64), in physical activation units. */
export async function loadAtlas(sampleIdx, stage, tileMax, base = BASE) {
  const { data, width } = await decodeGray(`samples/${sampleIdx}/atlas_${stage}.png`, base);
  return tileMax.map((m, c) => {
    const r = Math.floor(c / 8), k = c % 8;
    const f = new Float32Array(TILE * TILE);
    for (let y = 0; y < TILE; y++) {
      for (let x = 0; x < TILE; x++) {
        const q = data[((r * TILE + y) * width + (k * TILE + x)) * 4];
        f[y * TILE + x] = ((q - 128) / 127) * m;
      }
    }
    return f;
  });
}

/** io_r{res}.bin -> { input, target, prediction } Float32Array(res*res), physical units. */
export async function loadIO(sampleIdx, res, base = BASE) {
  const buf = new Float32Array(await fetchBytes(`samples/${sampleIdx}/io_r${res}${BIN}`, base));
  const n = res * res;
  return { res, input: buf.subarray(0, n), target: buf.subarray(n, 2 * n), prediction: buf.subarray(2 * n, 3 * n) };
}

/** spectra.bin -> spectra[layer][kind] Float32Array(64*64), log10 energy, (0,0) at [32,32]. */
export async function loadSpectra(sampleIdx, base = BASE) {
  const buf = new Float32Array(await fetchBytes(`samples/${sampleIdx}/spectra${BIN}`, base));
  const n = TILE * TILE;
  return [0, 1, 2, 3].map((l) => [0, 1, 2].map((k) => buf.subarray((l * 3 + k) * n, (l * 3 + k + 1) * n)));
}

export async function loadWeights(manifest) {
  return decodeWeights(manifest, await fetchBytes(manifest.weights.file.replace(/\.bin$/, BIN), baseOf(manifest)));
}

/** Decode weights.bin (ArrayBuffer) into { name: { shape, complex, data: Float64Array } }. */
export function decodeWeights(manifest, raw) {
  const flat = manifest.weights.dtype === "float16" ? float16ToFloat64(new Uint16Array(raw)) : Float64Array.from(new Float32Array(raw));
  const out = {};
  for (const [name, t] of Object.entries(manifest.weights.tensors)) {
    const size = t.shape.reduce((a, b) => a * b, 1) * (t.complex ? 2 : 1);
    out[name] = { shape: t.shape, complex: t.complex, data: flat.subarray(t.offset, t.offset + size) };
  }
  return out;
}

function float16ToFloat64(u16) {
  const out = new Float64Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x03ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

/** Everything the views need for one sample at 64x64, from precomputed assets. */
export async function loadSample(manifest, sampleIdx) {
  const entry = manifest.samples.find((s) => s.test_index === sampleIdx);
  const base = baseOf(manifest);
  const stages = ["lifted"];
  for (let l = 1; l <= 4; l++) stages.push(`layer${l}.spectral`, `layer${l}.local`, `layer${l}.out`);
  const fields = {};
  await Promise.all(stages.map(async (st) => { fields[st] = await loadAtlas(sampleIdx, st, entry.tile_max[st], base); }));
  const io = {};
  await Promise.all(manifest.resolutions.map(async (r) => { io[r] = await loadIO(sampleIdx, r, base); }));
  const spectra = await loadSpectra(sampleIdx, base);
  return { entry, fields, io, spectra, res: 64, source: "precomputed" };
}
