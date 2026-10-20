// 2D FFT helpers on top of fft.js, following fno_reference.py line for line.
// Layout: fields are row-major Float64Array(res*res); x = row index (dim 0), y = column (dim 1).
// Spectra are stored uncentered (index k mod res), like numpy.fft.fft2.
import FFT from "../vendor/fft.mjs";

const plans = new Map();
function plan(n) {
  if (!plans.has(n)) plans.set(n, { fft: new FFT(n), a: null, b: null });
  const p = plans.get(n);
  if (!p.a) { p.a = p.fft.createComplexArray(); p.b = p.fft.createComplexArray(); }
  return p;
}

/** Unnormalized 2D FFT (numpy.fft.fft2). Returns { re, im }. */
export function fft2(field, res, inverse = false) {
  const { fft, a, b } = plan(res);
  const re = new Float64Array(res * res), im = new Float64Array(res * res);
  const srcIm = field.im;
  const srcRe = field.re ?? field;
  // rows
  for (let x = 0; x < res; x++) {
    for (let y = 0; y < res; y++) { a[2 * y] = srcRe[x * res + y]; a[2 * y + 1] = srcIm ? srcIm[x * res + y] : 0; }
    inverse ? fft.inverseTransform(b, a) : fft.transform(b, a);
    for (let y = 0; y < res; y++) { re[x * res + y] = b[2 * y]; im[x * res + y] = b[2 * y + 1]; }
  }
  // columns
  for (let y = 0; y < res; y++) {
    for (let x = 0; x < res; x++) { a[2 * x] = re[x * res + y]; a[2 * x + 1] = im[x * res + y]; }
    inverse ? fft.inverseTransform(b, a) : fft.transform(b, a);
    for (let x = 0; x < res; x++) { re[x * res + y] = b[2 * x]; im[x * res + y] = b[2 * x + 1]; }
  }
  return { re, im };
}

/** numpy.fft.ifft2 (includes the 1/(res*res) factor). */
export function ifft2(spec, res) {
  return fft2(spec, res, true);
}

export const MODES = { kxLo: -8, kxHi: 7, kyLo: 0, kyHi: 8 }; // n_modes = (16, 16) -> 16 x 9 rfft block

/**
 * The library's truncation + inverse, given an rfft-block writer.
 * write(kx, ky) must return [re, im] of the output coefficient (norm="forward" units).
 * Implements: fill kept block, Hermitian fill F[-kx,-ky] = conj(F[kx,ky]) for 1 <= ky < res/2,
 * real(ifft2) * res^2.
 */
export function synthesize(res, write) {
  const N = res * res;
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let kx = MODES.kxLo; kx <= MODES.kxHi; kx++) {
    const r = ((kx % res) + res) % res;
    for (let ky = MODES.kyLo; ky <= MODES.kyHi; ky++) {
      const [vr, vi] = write(kx, ky);
      re[r * res + ky] = vr; im[r * res + ky] = vi;
    }
  }
  for (let ky = 1; ky < res / 2; ky++) {
    for (let r = 0; r < res; r++) {
      const src = r * res + ky;
      const dst = ((res - r) % res) * res + (res - ky);
      re[dst] = re[src]; im[dst] = -im[src];
    }
  }
  const out = ifft2({ re, im }, res);
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) y[i] = out.re[i] * N;
  return y;
}

/** Truncation alone (identity weights): what the spectral path keeps of one field. */
export function truncate(field, res) {
  const N = res * res;
  const F = fft2(field, res);
  return synthesize(res, (kx, ky) => {
    const i = (((kx % res) + res) % res) * res + ky;
    return [F.re[i] / N, F.im[i] / N];
  });
}

/** Centered energy |F/N|^2 as Float64Array(res*res), (0,0) at [res/2, res/2]. */
export function centeredEnergy(fields, res) {
  const N = res * res, h = res / 2;
  const E = new Float64Array(N);
  for (const f of fields) {
    const F = fft2(f, res);
    for (let x = 0; x < res; x++) {
      for (let y = 0; y < res; y++) {
        const i = x * res + y;
        const e = (F.re[i] ** 2 + F.im[i] ** 2) / (N * N);
        E[((x + h) % res) * res + ((y + h) % res)] += e;
      }
    }
  }
  return E;
}

/** Centered boolean mask of modes that survive truncation at this resolution (empirical, as in export_web.py). */
export function keptMask(res) {
  const N = res * res;
  const noise = new Float64Array(N);
  let s = 12345;
  for (let i = 0; i < N; i++) { s = (s * 1103515245 + 12345) % 2147483648; noise[i] = s / 2147483648 - 0.5; }
  const E = centeredEnergy([truncate(noise, res)], res);
  let mx = 0;
  for (const e of E) mx = Math.max(mx, e);
  return E.map((e) => (e > mx * 1e-18 ? 1 : 0));
}

/** Split a field into its part at kept modes and its part above the cutoff (exact partition in Fourier space). */
export function splitAtCutoff(field, res, maskCentered) {
  const N = res * res, h = res / 2;
  const F = fft2(field, res);
  const lo = { re: new Float64Array(N), im: new Float64Array(N) };
  const hi = { re: new Float64Array(N), im: new Float64Array(N) };
  for (let x = 0; x < res; x++) {
    for (let y = 0; y < res; y++) {
      const i = x * res + y;
      const keep = maskCentered[((x + h) % res) * res + ((y + h) % res)];
      const dst = keep ? lo : hi;
      dst.re[i] = F.re[i]; dst.im[i] = F.im[i];
    }
  }
  const a = ifft2(lo, res), b = ifft2(hi, res);
  return { low: a.re, high: b.re };
}
