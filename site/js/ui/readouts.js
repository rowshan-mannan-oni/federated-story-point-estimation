/* ==========================================================================
   readouts.js — the things that show a number back.

   Same contract as the controls: { el, set(value) }. None of them accept
   HTML, only numbers and plain strings, so a stop can never inject markup
   through a label.

   Numbers count up to their new value rather than snapping, because a value
   that MOVES tells you which way it went. That is switched off when the
   reader has asked for less motion.
   ========================================================================== */

import { num, fractionOf } from "./format.js";

function make(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function svg(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function motionIsReduced() {
  return document.documentElement.dataset.motion === "reduced"
    || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Count from one number to another, then stop. Returns a cancel function. */
function countTo(from, to, ms, onFrame) {
  if (motionIsReduced() || !Number.isFinite(from) || ms <= 0) {
    onFrame(to);
    return () => {};
  }
  const started = performance.now();
  let raf = 0;
  const tick = (now) => {
    const t = Math.min((now - started) / ms, 1);
    const eased = 1 - (1 - t) ** 3;             // fast, then settles
    onFrame(from + (to - from) * eased);
    if (t < 1) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}

/* ==========================================================================
   READOUT — a number in a lit well, with a label and a unit.
   ========================================================================== */
export function createReadout({
  label, value = 0, unit = "", format, size = "md", caption = "",
} = {}) {
  const el = make("div", `part part-readout is-${size}`);
  const show = format ?? ((v) => num(v, { digits: 0 }));

  const labelEl = make("span", "part-label", label);
  const well = make("div", "readout-well mat-recess");
  const valueEl = make("span", "readout-value");
  const unitEl = unit ? make("span", "readout-unit", unit) : null;

  well.append(valueEl);
  if (unitEl) well.append(unitEl);
  el.append(labelEl, well);
  if (caption) el.append(make("p", "part-hint", caption));

  let current = value;
  let cancel = () => {};
  valueEl.textContent = show(current);

  return {
    el,
    get: () => current,
    set(next) {
      cancel();
      const from = current;
      current = next;
      cancel = countTo(from, next, 420, (v) => { valueEl.textContent = show(v); });
    },
  };
}

/* ==========================================================================
   METER — a horizontal bar. Optionally with a marker: a reference value the
   bar is being judged against (a baseline, a target, a previous result).
   ========================================================================== */
export function createMeter({
  label, min = 0, max = 1, value = 0, marker = null, markerLabel = "",
  format, unit = "",
} = {}) {
  const el = make("div", "part part-meter");
  const show = format ?? ((v) => num(v, { digits: 2 }));

  const head = make("div", "part-head");
  const labelEl = make("span", "part-label", label);
  const valueEl = make("span", "part-value");
  head.append(labelEl, valueEl);

  const bar = make("div", "meter-bar mat-recess");
  const fill = make("i", "meter-fill");
  bar.append(fill);

  bar.setAttribute("role", "meter");
  bar.setAttribute("aria-valuemin", String(min));
  bar.setAttribute("aria-valuemax", String(max));
  bar.setAttribute("aria-label", label);

  if (marker != null) {
    const pin = make("i", "meter-marker");
    pin.style.setProperty("--f", String(fractionOf(marker, min, max)));
    if (markerLabel) pin.title = markerLabel;
    bar.append(pin);
  }

  el.append(head, bar);

  let current = value;
  let cancel = () => {};

  function paint(v) {
    fill.style.setProperty("--f", String(fractionOf(v, min, max)));
    valueEl.textContent = `${show(v)}${unit ? ` ${unit}` : ""}`;
  }
  paint(current);
  bar.setAttribute("aria-valuenow", String(current));

  return {
    el,
    get: () => current,
    set(next) {
      cancel();
      const from = current;
      current = next;
      bar.setAttribute("aria-valuenow", String(next));
      cancel = countTo(from, next, 420, paint);
    },
  };
}

/* ==========================================================================
   GAUGE — a needle on an arc. For a single score you want to read at a
   glance, where the position on the scale matters more than the digits.
   ========================================================================== */
export function createGauge({
  label, min = 0, max = 1, value = 0, format, caption = "",
} = {}) {
  const el = make("div", "part part-gauge");
  const show = format ?? ((v) => num(v, { digits: 3 }));

  const A0 = -120;
  const A1 = 120;
  const R = 38;
  const CX = 50;
  const CY = 46;

  const point = (angle, radius) => {
    const rad = ((angle - 90) * Math.PI) / 180;
    return [CX + radius * Math.cos(rad), CY + radius * Math.sin(rad)];
  };
  const arc = (a0, a1, radius) => {
    const [x0, y0] = point(a0, radius);
    const [x1, y1] = point(a1, radius);
    const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
    return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${radius} ${radius} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  };

  const chart = svg("svg", { viewBox: "0 0 100 62", class: "gauge-svg", "aria-hidden": "true" });
  chart.append(svg("path", { d: arc(A0, A1, R), class: "gauge-track" }));

  // Ticks at every tenth of the scale.
  for (let i = 0; i <= 10; i += 1) {
    const a = A0 + ((A1 - A0) * i) / 10;
    const [x0, y0] = point(a, R - (i % 5 === 0 ? 7 : 4));
    const [x1, y1] = point(a, R - 1);
    chart.append(svg("line", {
      x1: x0.toFixed(2), y1: y0.toFixed(2), x2: x1.toFixed(2), y2: y1.toFixed(2),
      class: i % 5 === 0 ? "gauge-tick is-major" : "gauge-tick",
    }));
  }

  const needle = svg("line", { x1: CX, y1: CY, x2: CX, y2: CY - R + 5, class: "gauge-needle" });
  chart.append(needle);
  chart.append(svg("circle", { cx: CX, cy: CY, r: 3.4, class: "gauge-hub" }));

  const labelEl = make("span", "part-label", label);
  const valueEl = make("span", "gauge-value");

  el.append(chart, valueEl, labelEl);
  if (caption) el.append(make("p", "part-hint", caption));

  el.setAttribute("role", "img");

  let current = value;
  let cancel = () => {};

  function paint(v) {
    const angle = A0 + (A1 - A0) * fractionOf(v, min, max);
    needle.setAttribute("transform", `rotate(${angle.toFixed(2)} ${CX} ${CY})`);
    valueEl.textContent = show(v);
  }
  function describe(v) {
    el.setAttribute("aria-label", `${label}: ${show(v)} (scale ${show(min)} to ${show(max)})`);
  }
  paint(current);
  describe(current);

  return {
    el,
    get: () => current,
    set(next) {
      cancel();
      const from = current;
      current = next;
      describe(next);
      cancel = countTo(from, next, 520, paint);
    },
  };
}

/* ==========================================================================
   STAT PLATE — one number that matters, with a word about what it is.
   The workhorse of the results stops.
   ========================================================================== */
export function createStatPlate({ label, value = "—", caption = "", tone = "plain" } = {}) {
  const el = make("div", `part part-stat mat-sub is-${tone}`);
  const valueEl = make("strong", "stat-value", String(value));
  const labelEl = make("span", "stat-label", label);
  el.append(valueEl, labelEl);
  if (caption) el.append(make("span", "stat-caption", caption));

  return {
    el,
    get: () => valueEl.textContent,
    set(next) { valueEl.textContent = String(next); },
  };
}

/* ==========================================================================
   DRAWER — something you can open when you want the detail, and leave shut
   when you do not. Built on <details>, which already knows how to behave.
   ========================================================================== */
export function createDrawer({ label, summary = "", open = false } = {}) {
  const el = make("details", "part part-drawer mat-sub");
  el.open = open;

  const head = make("summary", "drawer-head");
  head.append(make("span", "drawer-title", label));
  if (summary) head.append(make("span", "drawer-summary", summary));
  head.append(make("i", "drawer-chevron"));

  const body = make("div", "drawer-body");
  el.append(head, body);

  return { el, body, open: () => { el.open = true; }, close: () => { el.open = false; } };
}
