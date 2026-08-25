/* ==========================================================================
   stop-parts.js — the pieces every stop is built from.

   The plan for this site is that each stop has the same shape, so a reader
   always knows where they are: what problem we just hit, the idea in one
   sentence, something to try, what happened, what it costs, and the question
   that leads to the next stop.

   Those pieces live here rather than being retyped in twenty-five files, so
   the shape cannot drift as the site grows.
   ========================================================================== */

import { STOPS, COUNT, areaLabel } from "../core/stops.js";

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * The card a stop lives on, with its heading already in place.
 * Returns the card so a stop can append to it.
 */
export function stopCard({ stop, index, title, standfirst }) {
  const card = el("article", "card mat-panel");

  const overline = el("p", "overline label");
  overline.append(el("span", null, areaLabel(stop.area)));
  overline.append(el("span", "bar"));
  overline.append(el("span", null, `Stop ${index + 1} of ${COUNT}`));
  card.append(overline);

  card.append(el("h1", null, title ?? stop.title));

  if (standfirst) {
    const lead = el("p", "lead");
    lead.innerHTML = standfirst;      // written by us, never by data
    card.append(lead);
  }
  return card;
}

/** A titled block within a stop. */
export function block(title, ...children) {
  const section = el("section", "stop-block");
  if (title) section.append(el("h2", "block-title", title));
  children.forEach((child) => child && section.append(child));
  return section;
}

/** A paragraph written as markup — for prose with facts and terms in it. */
export function prose(html, className = "note") {
  const p = el("p", className);
  p.innerHTML = html;
  return p;
}

/**
 * The handover at the foot of a stop: the question this stop leaves you with,
 * and the button that goes and answers it.
 */
export function handover({ question, next, cta }) {
  const wrap = el("div", "handover mat-sub");

  const text = el("p", "handover-question");
  text.innerHTML = question;
  wrap.append(text);

  const button = el("button", "ctrl handover-next");
  button.type = "button";
  button.append(el("span", "ctrl-label", cta ?? "Next stop"));
  const arrow = el("span", null, "→");
  arrow.setAttribute("aria-hidden", "true");
  button.append(arrow);
  button.addEventListener("click", () => next());
  wrap.append(button);

  return wrap;
}

/** A row of stat plates, kept on one line where there is room. */
export function statRow(...plates) {
  const row = el("div", "stat-row");
  plates.forEach((plate) => row.append(plate.el ?? plate));
  return row;
}

/** The seven parts of the path, as buttons that jump to each one. */
export function pathOverview(goTo) {
  const list = el("ul", "path-list");
  const seen = new Set();

  STOPS.forEach((stop, index) => {
    if (seen.has(stop.area)) return;
    seen.add(stop.area);

    const count = STOPS.filter((s) => s.area === stop.area).length;
    const item = el("li");
    const button = el("button", "path-row mat-sub");
    button.type = "button";
    button.append(el("span", "path-n", String(index + 1).padStart(2, "0")));
    const words = el("span", "path-words");
    words.append(el("span", "path-title", areaLabel(stop.area)));
    words.append(el("span", "path-count", `${count} stop${count === 1 ? "" : "s"}`));
    button.append(words);
    button.setAttribute("aria-label", `${areaLabel(stop.area)} — ${count} stops, starting at stop ${index + 1}`);
    button.addEventListener("click", () => goTo(index));
    item.append(button);
    list.append(item);
  });

  return list;
}
