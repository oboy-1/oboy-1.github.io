// Numerics shared by the views and the forward pass.

const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);

/** erf to ~1e-14: Taylor series for |x| < 3, Lentz continued fraction for erfc beyond. */
export function erf(x) {
  const ax = Math.abs(x);
  if (ax < 3) {
    let term = x, sum = x;
    const x2 = x * x;
    for (let n = 1; n < 200; n++) {
      term *= -x2 / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    return TWO_OVER_SQRT_PI * sum;
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
  const tiny = 1e-300;
  let f = ax, C = ax, D = 0;
  for (let n = 1; n < 500; n++) {
    const a = n / 2;
    D = ax + a * D; D = Math.abs(D) < tiny ? tiny : D; D = 1 / D;
    C = ax + a / C; C = Math.abs(C) < tiny ? tiny : C;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  const erfc = Math.exp(-ax * ax) / Math.sqrt(Math.PI) / f;
  return x > 0 ? 1 - erfc : erfc - 1;
}

/** Exact GELU, as torch.nn.functional.gelu (approximate="none"). */
export function gelu(v) {
  return 0.5 * v * (1 + erf(v / Math.SQRT2));
}

export function geluArray(src, out = new Float64Array(src.length)) {
  for (let i = 0; i < src.length; i++) out[i] = gelu(src[i]);
  return out;
}

export function pearson(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  const d = Math.sqrt(saa * sbb);
  return d > 0 ? sab / d : 0;
}

export function spatialStd(f) {
  let m = 0;
  for (let i = 0; i < f.length; i++) m += f[i];
  m /= f.length;
  let s = 0;
  for (let i = 0; i < f.length; i++) s += (f[i] - m) ** 2;
  return Math.sqrt(s / f.length);
}
