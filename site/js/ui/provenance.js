/* ==========================================================================
   provenance.js — every number on this site can be asked where it came from.

   Numbers in a thesis explainer are claims. A reader who cannot check a claim
   has to take it on trust, and trust is exactly what an explainer has to earn.
   So a figure rendered through here is a button: hover it or tab to it and it
   says which file it came from and how it was worked out.

   It also refuses to fail silently. A fact that has not been generated shows
   a dash and says so, rather than a plausible-looking zero.
   ========================================================================== */

import * as data from "../core/data.js";
import * as tip from "./tip.js";
import { stampFor } from "./stamp.js";

const KIND_WORDS = {
  measured: "Measured from the data on this machine",
  run:      "Read from a finished training run",
  derived:  "Worked out from other figures",
  missing:  "Not available yet",
};

/**
 * A number with its provenance attached.
 *
 * @param {string} key     e.g. "corpus.rows_clean"
 * @param {object} options
 *   format(value, fact) -> string   override the display text
 *   label                            words read out after the number
 */
export function factEl(key, { format, label } = {}) {
  const entry = data.fact(key);

  const el = document.createElement("button");
  el.type = "button";
  el.className = "fact";
  el.dataset.kind = entry.kind;

  const shown = format ? format(entry.value, entry) : entry.text;
  el.append(Object.assign(document.createElement("span"), {
    className: "fact-value",
    textContent: shown,
  }));

  if (label) {
    el.append(Object.assign(document.createElement("span"), {
      className: "fact-label",
      textContent: ` ${label}`,
    }));
  }

  el.setAttribute("aria-label", `${shown}${label ? " " + label : ""} — where this came from`);
  tip.attach(el, () => buildTip(key, entry));
  return el;
}

/** Just the text of a fact, for places a button would be wrong (e.g. a title). */
export function factText(key, { format } = {}) {
  const entry = data.fact(key);
  return format ? format(entry.value, entry) : entry.text;
}

function buildTip(key, entry) {
  const box = document.createElement("div");
  box.className = "tip-body";

  const head = document.createElement("div");
  head.className = "tip-head";
  head.append(stampFor(entry));
  const kind = document.createElement("span");
  kind.className = "tip-kind";
  kind.textContent = KIND_WORDS[entry.kind] ?? entry.kind;
  head.append(kind);
  box.append(head);

  const how = document.createElement("p");
  how.className = "tip-how";
  how.textContent = entry.how;
  box.append(how);

  const source = document.createElement("p");
  source.className = "tip-source";
  source.append(Object.assign(document.createElement("span"), {
    className: "label", textContent: "Source",
  }));
  source.append(Object.assign(document.createElement("code"), {
    textContent: entry.source,
  }));
  box.append(source);

  const id = document.createElement("p");
  id.className = "tip-id";
  id.textContent = key;
  box.append(id);

  return box;
}

/**
 * Replace <span data-fact="corpus.rows_clean"></span> placeholders inside a
 * block of markup. Lets a stop be written as readable prose with the numbers
 * dropped in, instead of being assembled node by node.
 */
export function fillFacts(root) {
  root.querySelectorAll("[data-fact]").forEach((slot) => {
    const key = slot.dataset.fact;
    const options = {};
    if (slot.dataset.factLabel) options.label = slot.dataset.factLabel;
    slot.replaceWith(factEl(key, options));
  });
  return root;
}
