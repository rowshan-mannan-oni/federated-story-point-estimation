/* ==========================================================================
   glossary.js — a word a reader does not know should never stop them.

   Terms are marked in the prose and explained where they stand. The rule for
   the site is: explain a term the first time it appears in a stop, in one
   line, and mark it so the fuller definition is a hover or a tab away.

   Definitions live in site/data/glossary.json. They are words rather than
   measurements, so that file is written by hand — the only one in site/data/
   that is.
   ========================================================================== */

import * as data from "../core/data.js";
import * as tip from "./tip.js";

let terms = null;

export async function init() {
  const payload = await data.glossary();
  terms = payload?.terms ?? {};
  return Boolean(payload);
}

function lookup(key) {
  if (!terms) return null;
  const wanted = key.trim().toLowerCase();
  return terms[wanted] ?? null;
}

function buildTip(key, entry) {
  const box = document.createElement("div");
  box.className = "tip-body";

  const head = document.createElement("p");
  head.className = "tip-term";
  head.textContent = key;
  box.append(head);

  const short = document.createElement("p");
  short.className = "tip-short";
  short.textContent = entry.short;
  box.append(short);

  if (entry.more) {
    const more = document.createElement("p");
    more.className = "tip-more";
    more.textContent = entry.more;
    box.append(more);
  }
  return box;
}

/**
 * Turn <span data-term="lora">LoRA</span> into an explainable term.
 * An unknown term is left as plain text rather than becoming a dead control —
 * a marker with nothing behind it is worse than no marker.
 */
export function fillTerms(root) {
  root.querySelectorAll("[data-term]").forEach((slot) => {
    const key = slot.dataset.term;
    const entry = lookup(key);
    const words = slot.textContent || key;

    if (!entry) {
      slot.replaceWith(document.createTextNode(words));
      return;
    }

    const el = document.createElement("button");
    el.type = "button";
    el.className = "term";
    el.textContent = words;
    el.setAttribute("aria-label", `${words} — what this means`);
    tip.attach(el, () => buildTip(words, entry));
    slot.replaceWith(el);
  });
  return root;
}

export function allTerms() {
  return terms ? { ...terms } : {};
}
