/* ==========================================================================
   format.js — turning numbers into something a reader can take in.

   Used by every control and every readout, so the site says "1.01 MB" and
   "252,484" the same way everywhere. Formatting lives here and nowhere else.
   ========================================================================== */

/** 1234567.8 -> "1,234,568" (or "1,234,567.80" with digits: 2) */
export function num(value, { digits = 0, group = true } = {}) {
  if (!Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
    useGrouping: group,
  });
}

/** 0.2019 -> "20.2%" ; pass fraction: false if the value is already 0–100 */
export function pct(value, { digits = 1, fraction = true } = {}) {
  if (!Number.isFinite(value)) return "—";
  return `${num(fraction ? value * 100 : value, { digits })}%`;
}

/** 1009936 -> "1.01 MB". Decimal units, because that is how the run reports. */
export function bytes(value, { digits = 2 } = {}) {
  if (!Number.isFinite(value)) return "—";
  const units = ["B", "kB", "MB", "GB", "TB"];
  let v = Math.abs(value);
  let i = 0;
  while (v >= 1000 && i < units.length - 1) { v /= 1000; i += 1; }
  const sign = value < 0 ? "-" : "";
  return `${sign}${num(v, { digits: i === 0 ? 0 : digits })} ${units[i]}`;
}

/** 495.26 -> "495×" — for "this is N times smaller" comparisons. */
export function times(value, { digits = 0 } = {}) {
  if (!Number.isFinite(value)) return "—";
  return `${num(value, { digits })}×`;
}

/** How many decimal places a step implies: 0.01 -> 2, 5 -> 0. */
export function decimalsFor(step) {
  if (!Number.isFinite(step) || step <= 0) return 0;
  const text = String(step);
  const dot = text.indexOf(".");
  return dot === -1 ? 0 : text.length - dot - 1;
}

/**
 * Snap a value onto the step grid and keep it inside the range.
 * Rounding through the step count avoids the 0.1 + 0.2 problem.
 */
export function quantize(value, min, max, step) {
  if (!Number.isFinite(value)) return min;
  const steps = Math.round((value - min) / step);
  const snapped = min + steps * step;
  const fixed = Number(snapped.toFixed(decimalsFor(step) + 2));
  return Math.min(Math.max(fixed, min), max);
}

/** Where `value` sits in [min, max], as 0–1. */
export function fractionOf(value, min, max) {
  if (max === min) return 0;
  return Math.min(Math.max((value - min) / (max - min), 0), 1);
}
