/* ==========================================================================
   store.js — the reader's settings, and a small way to react to changes.

   Three settings, all remembered between visits:
     theme     "auto" | "light" | "dark"       the room in daylight or at night
     materials "rich" | "flat"                 textures on, or flattened
     motion    "auto" | "reduced"              movement on, or held still

   Each one becomes a data-attribute on <html>, which is all the CSS needs.
   Later steps add more keys here (which stop you are on, plain/detailed
   wording, sound). Nothing else in the site should touch localStorage.
   ========================================================================== */

const KEY = "fedsp.site.v1";

const DEFAULTS = {
  theme: "auto",
  materials: "rich",
  motion: "auto",
};

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS };
  } catch {
    // Private browsing, disabled storage, corrupted value: fall back quietly.
    return { ...DEFAULTS };
  }
}

let state = read();
const listeners = new Set();

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* not being able to remember a preference is not worth breaking over */
  }
}

/** Push the current settings onto <html> so CSS can act on them. */
export function apply() {
  const root = document.documentElement;

  if (state.theme === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", state.theme);

  root.setAttribute("data-materials", state.materials);
  root.setAttribute("data-motion", state.motion);
}

export function get(key) {
  return key === undefined ? { ...state } : state[key];
}

export function set(key, value) {
  if (state[key] === value) return value;
  state[key] = value;
  persist();
  apply();
  for (const fn of listeners) fn(key, value, { ...state });
  return value;
}

/** Step to the next value in a list — what the toolbar buttons use. */
export function cycle(key, values) {
  const now = state[key];
  const i = values.indexOf(now);
  return set(key, values[(i + 1) % values.length]);
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** True when the reader has asked for less movement, by setting or by OS. */
export function motionIsReduced() {
  if (state.motion === "reduced") return true;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Which theme is actually on screen right now, "auto" resolved. */
export function resolvedTheme() {
  if (state.theme !== "auto") return state.theme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
