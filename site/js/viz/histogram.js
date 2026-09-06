/* ==========================================================================
   histogram.js — counts as bars, with room for a second number underneath.

   Used where the shape of a distribution is the point: how long issues are,
   how errors are spread. Bars are drawn in the page's own ink so they work in
   both themes, and every bar states its own value in text as well as height,
   because a bar chart nobody can read the numbers off is decoration.
   ========================================================================== */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @param {object} options
 *   bars     [{ label, value, note, mark }]
 *   caption  a hidden description for screen readers
 *   noteLabel  what the small number under each bar means
 */
export function createHistogram({ bars = [], caption, noteLabel } = {}) {
  const wrap = el("figure", "hist");
  const peak = Math.max(...bars.map((b) => b.value), 1);

  if (caption) {
    wrap.setAttribute("role", "img");
    wrap.setAttribute("aria-label",
      `${caption}. ` + bars.map((b) => `${b.label}: ${b.value}`).join(", "));
  }

  const plot = el("div", "hist-plot");
  bars.forEach((bar) => {
    const column = el("div", "hist-col");
    if (bar.mark) column.dataset.mark = "true";

    const value = el("span", "hist-value", formatCount(bar.value));
    const stem = el("div", "hist-stem");
    const fill = el("span", "hist-fill");
    fill.style.setProperty("--h", String(bar.value / peak));
    stem.append(fill);

    column.append(value, stem, el("span", "hist-label", bar.label));
    if (bar.note != null) column.append(el("span", "hist-note", bar.note));
    plot.append(column);
  });
  wrap.append(plot);

  if (noteLabel) {
    wrap.append(el("figcaption", "hist-caption", noteLabel));
  }
  return { el: wrap };
}

function formatCount(value) {
  if (value >= 10000) return `${Math.round(value / 1000)}k`;
  return Number(value).toLocaleString();
}
