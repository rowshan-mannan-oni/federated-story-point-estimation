/* ==========================================================================
   spans.js — date ranges on a shared axis.

   Built for one job: showing that two ways of dividing the same data produce
   completely different pictures. Cut at random and every pile covers the whole
   history; cut by date and they follow one another. Nothing else makes that as
   obvious as putting the ranges on the same axis and looking at them.
   ========================================================================== */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function toTime(value) {
  const t = Date.parse(value);
  return Number.isFinite(t) ? t : null;
}

/**
 * @param {object} options
 *   rows   [{ label, first, last, n, tone }]  ISO dates
 *   note   a line under the chart
 */
export function createSpans({ rows = [], note } = {}) {
  const wrap = el("div", "spans");

  const times = rows.flatMap((r) => [toTime(r.first), toTime(r.last)]).filter(Boolean);
  if (!times.length) {
    wrap.append(el("p", "note", "No dates to show."));
    return { el: wrap };
  }
  const start = Math.min(...times);
  const end = Math.max(...times);
  const range = Math.max(end - start, 1);

  const axis = el("div", "spans-axis");
  axis.append(el("span", null, new Date(start).getFullYear()));
  axis.append(el("span", null, new Date(end).getFullYear()));
  wrap.append(axis);

  rows.forEach((row) => {
    const first = toTime(row.first);
    const last = toTime(row.last);

    const line = el("div", "span-row");
    line.append(el("span", "span-label", row.label));

    const track = el("div", "span-track mat-recess");
    if (first !== null && last !== null) {
      const bar = el("span", "span-bar");
      bar.dataset.tone = row.tone ?? "plain";
      bar.style.setProperty("--from", String((first - start) / range));
      bar.style.setProperty("--to", String((last - start) / range));
      bar.title = `${row.first} to ${row.last}`;
      track.append(bar);
    }
    line.append(track);

    const meta = el("span", "span-meta",
      first === null ? "—" : `${String(row.first).slice(0, 7)} → ${String(row.last).slice(0, 7)}`);
    line.append(meta);

    // The bar is decorative; the dates and counts are the real content, and
    // they are in the text, so a screen reader gets the same information.
    line.setAttribute("aria-label",
      `${row.label}: ${row.n} issues, ${row.first ?? "no dates"} to ${row.last ?? ""}`);

    wrap.append(line);
  });

  if (note) wrap.append(el("p", "spans-note", note));
  return { el: wrap };
}

/** Do two ranges share any time at all? */
export function overlaps(a, b) {
  const a1 = toTime(a.first), a2 = toTime(a.last);
  const b1 = toTime(b.first), b2 = toTime(b.last);
  if ([a1, a2, b1, b2].some((v) => v === null)) return false;
  return a1 <= b2 && b1 <= a2;
}
