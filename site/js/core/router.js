/* ==========================================================================
   router.js — which stop you are on, and how that ends up in the address bar.

   The address is the single source of truth. Everything that navigates — the
   Next button, a rail tick, the map, the arrow keys — does the same thing:
   set the hash. The router hears the change and tells everyone else.

   That means the browser's own Back and Forward buttons work for free, and
   any stop can be linked to directly: .../site/#/archive
   ========================================================================== */

import { STOPS, indexOf, clampIndex, COUNT } from "./stops.js";

const listeners = new Set();
let current = 0;
let started = false;

/** Read the address bar. Returns -1 when it names no stop we know. */
function indexFromHash() {
  const raw = location.hash.replace(/^#\/?/, "").trim();
  if (!raw) return -1;
  return indexOf(raw);
}

function notify(index, reason) {
  current = index;
  for (const fn of listeners) fn(index, STOPS[index], reason);
}

/** Called on every hash change, including the browser's Back button. */
function onHashChange() {
  const found = indexFromHash();

  if (found === -1) {
    // An address we don't recognise: go to the first stop and tidy the URL
    // without adding another history entry to step back through.
    history.replaceState(null, "", `#/${STOPS[0].id}`);
    notify(0, "correct");
    return;
  }

  notify(found, "hash");
}

export function start() {
  if (started) return current;
  started = true;

  const found = indexFromHash();
  const initial = found === -1 ? 0 : found;

  // Normalise the address on arrival (".../site/" becomes ".../site/#/front-door")
  // with replaceState, so the reader's first Back press leaves the site rather
  // than bouncing between two versions of the same stop.
  history.replaceState(null, "", `#/${STOPS[initial].id}`);

  window.addEventListener("hashchange", onHashChange);

  notify(initial, "initial");
  return initial;
}

/** Ask to move. This only sets the address; the change comes back through. */
export function goTo(index) {
  const target = clampIndex(index);
  if (target === current) return;
  location.hash = `#/${STOPS[target].id}`;
}

export function goToId(id) {
  const i = indexOf(id);
  if (i !== -1) goTo(i);
}

export function next() { goTo(current + 1); }
export function prev() { goTo(current - 1); }
export function first() { goTo(0); }
export function last()  { goTo(COUNT - 1); }

export function currentIndex() { return current; }

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
