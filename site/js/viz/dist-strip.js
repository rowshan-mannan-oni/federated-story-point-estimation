/* ==========================================================================
   dist-strip.js — how a project spreads its work across the five numbers.

   One bar, five segments, widths proportional to how many issues got each
   value. Put two of these next to each other and the central problem of this
   thesis is visible in about a second: the same five numbers, used to mean
   entirely different things.

   Greyscale, like everything else, so the ramp from 1 to 8 is a ramp from
   light to dark. That works in both themes because the shades are mixed
   against the page's own ink and panel colours rather than hard-coded.
   ========================================================================== */

export const CLASSES = [1, 2, 3, 5, 8];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @param {object}  options
 *   title    the project or group this describes
 *   counts   { "1": n, "2": n, "3": n, "5": n, "8": n }
 *   note     a line under the bar
 *   compact  drop the in-bar numbers, for small multiples
 */
export function createDistStrip({ title, counts = {}, note, compact = false } = {}) {
  const values = CLASSES.map((c) => Number(counts[String(c)] ?? 0));
  const total = values.reduce((sum, v) => sum + v, 0);

  const wrap = el("div", "dist");

  if (title) {
    const head = el("div", "dist-head");
    head.append(el("span", "dist-title", title));
    if (note) head.append(el("span", "dist-note", note));
    wrap.append(head);
  }

  const bar = el("div", "dist-bar mat-recess");
  // One accessible description for the whole bar: a screen reader gets the
  // shares in words rather than five unlabelled boxes.
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label",
    total === 0
      ? `${title ?? "Distribution"}: no issues`
      : `${title ?? "Distribution"}: ` +
        CLASSES.map((c, i) => `${Math.round(100 * values[i] / total)}% are ${c}`).join(", "));

  CLASSES.forEach((cls, i) => {
    const share = total === 0 ? 0 : values[i] / total;
    const seg = el("span", "dist-seg");
    seg.dataset.cls = String(cls);
    seg.style.setProperty("--share", String(share));
    seg.title = `${cls} point${cls === 1 ? "" : "s"}: ${values[i]} issues (${Math.round(100 * share)}%)`;
    // Only label a segment wide enough to hold the label without squashing.
    if (!compact && share > 0.08) seg.append(el("span", "dist-seg-label", String(cls)));
    bar.append(seg);
  });

  wrap.append(bar);
  return { el: wrap, total };
}

/** The shared key, shown once above a set of strips. */
export function createDistLegend() {
  const legend = el("div", "dist-legend");
  legend.append(el("span", "label", "Story points"));
  CLASSES.forEach((cls) => {
    const item = el("span", "dist-key");
    const swatch = el("i", "dist-swatch");
    swatch.dataset.cls = String(cls);
    item.append(swatch, el("span", null, String(cls)));
    legend.append(item);
  });
  return legend;
}
