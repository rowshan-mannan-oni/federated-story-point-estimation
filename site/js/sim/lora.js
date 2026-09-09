/* ==========================================================================
   lora.js — the merging experiment, run in the browser.

   This is real arithmetic, not an animation of one. Every time the reader
   moves a control, the page builds a fresh set of client patches, merges them
   both ways, and measures the gap.

   It runs at a smaller size than the model does — a 64-wide square instead of
   768 — because the reader is dragging a control and the answer has to come
   back immediately. The size does not change the argument, and the page says
   so: the same experiment at the real 768 is run by the extraction script and
   quoted alongside.

   The identity being demonstrated:

       avg(Bi·Ai) − avg(B)·avg(A)  =  Σ pi (Bi − B̄)(Ai − Ā)

   i.e. the error from averaging the two halves separately is exactly the
   weighted cross-covariance of how far each client drifted. Freeze A and every
   (Ai − Ā) is zero, so the error is not small — it is gone.
   ========================================================================== */

/** A small, seeded generator, so a given setting always gives the same result. */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Normal-ish noise from two uniforms (Box–Muller). */
function gaussian(rng) {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function matrix(rows, cols, scale, rng) {
  const out = new Float64Array(rows * cols);
  for (let i = 0; i < out.length; i++) out[i] = gaussian(rng) * scale;
  return { rows, cols, data: out };
}

function addNoise(base, scale, rng) {
  const out = new Float64Array(base.data.length);
  for (let i = 0; i < out.length; i++) out[i] = base.data[i] + gaussian(rng) * scale;
  return { rows: base.rows, cols: base.cols, data: out };
}

function multiply(a, b) {
  const out = new Float64Array(a.rows * b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < a.cols; k++) {
      const av = a.data[i * a.cols + k];
      if (av === 0) continue;
      for (let j = 0; j < b.cols; j++) out[i * b.cols + j] += av * b.data[k * b.cols + j];
    }
  }
  return { rows: a.rows, cols: b.cols, data: out };
}

function weightedSum(mats, weights) {
  const out = new Float64Array(mats[0].data.length);
  mats.forEach((m, i) => {
    const w = weights[i];
    for (let j = 0; j < out.length; j++) out[j] += w * m.data[j];
  });
  return { rows: mats[0].rows, cols: mats[0].cols, data: out };
}

function subtract(a, b) {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] - b.data[i];
  return { rows: a.rows, cols: a.cols, data: out };
}

function norm(m) {
  let total = 0;
  for (let i = 0; i < m.data.length; i++) total += m.data[i] * m.data[i];
  return Math.sqrt(total);
}

function maxAbs(m) {
  let peak = 0;
  for (let i = 0; i < m.data.length; i++) peak = Math.max(peak, Math.abs(m.data[i]));
  return peak;
}

/**
 * Merge a set of client patches both ways and measure the difference.
 *
 * @param {object} options
 *   spread   how differently the clients' A halves drifted (0 = identical)
 *   frozen   true = one shared A nobody trains (what the thesis does)
 *   clients  how many projects
 *   width    size of the square the patch stands in for
 *   rank     width of the strips
 * @returns {{error:number, covResidual:number, truth:object, naive:object, gap:object}}
 */
export function mergeExperiment({
  spread = 0.02, frozen = false, clients = 18, width = 64, rank = 4, seed = 7,
} = {}) {
  const rng = makeRng(seed);

  // Weight each project by how much data it has, as the real server does.
  const rawWeights = Array.from({ length: clients }, () => 0.5 + rng());
  const totalWeight = rawWeights.reduce((a, b) => a + b, 0);
  const weights = rawWeights.map((w) => w / totalWeight);

  const sharedA = matrix(rank, width, 0.02, rng);
  const bs = Array.from({ length: clients }, () => matrix(width, rank, 0.02, rng));
  const as = frozen
    ? bs.map(() => sharedA)
    : bs.map(() => addNoise(sharedA, spread, rng));

  // What the update actually is: average the finished patches.
  const truth = weightedSum(bs.map((b, i) => multiply(b, as[i])), weights);

  // What separate averaging gives: average each half, then multiply.
  const bBar = weightedSum(bs, weights);
  const aBar = weightedSum(as, weights);
  const naive = multiply(bBar, aBar);

  const gap = subtract(truth, naive);
  const error = norm(truth) === 0 ? 0 : norm(gap) / norm(truth);

  // The closed form: the gap should BE the cross-covariance of the deviations.
  const covariance = weightedSum(
    bs.map((b, i) => multiply(subtract(b, bBar), subtract(as[i], aBar))),
    weights,
  );
  const covResidual = maxAbs(subtract(gap, covariance));

  return { error, covResidual, truth, naive, gap, bBar, aBar };
}

/** A coarse picture of a matrix, for drawing. Returns rows of -1..1 values. */
export function thumbnail(m, size = 12) {
  const out = [];
  const peak = maxAbs(m) || 1;
  const rowStep = Math.max(1, Math.floor(m.rows / size));
  const colStep = Math.max(1, Math.floor(m.cols / size));
  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) {
      // Average the block this cell stands for, so the picture is the matrix
      // rather than a scattering of individual samples from it.
      let sum = 0;
      let seen = 0;
      for (let i = r * rowStep; i < Math.min((r + 1) * rowStep, m.rows); i++) {
        for (let j = c * colStep; j < Math.min((c + 1) * colStep, m.cols); j++) {
          sum += m.data[i * m.cols + j];
          seen += 1;
        }
      }
      row.push(seen ? (sum / seen) / peak : 0);
    }
    out.push(row);
  }
  return out;
}
