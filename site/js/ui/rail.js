/* ==========================================================================
   rail.js — the progress rail along the bottom.

   One tick per stop, in order. It answers three questions at a glance:
   how far along am I, how much is left, and how much of it is actually built
   yet. Ticks are real buttons, so the rail is also the fastest way to jump.
   ========================================================================== */

import { STOPS, COUNT } from "../core/stops.js";

let railEl = null;
let ticks = [];

export function init({ el, onJump }) {
  railEl = el;
  railEl.innerHTML = "";

  ticks = STOPS.map((stop, i) => {
    const tick = document.createElement("button");
    tick.type = "button";
    tick.className = "tick";
    tick.dataset.index = String(i);
    tick.dataset.built = String(stop.built);
    // A tick is a 6px-wide button, so its accessible name has to carry
    // everything the eye gets from position and shading.
    tick.setAttribute(
      "aria-label",
      `Stop ${i + 1} of ${COUNT}: ${stop.title}${stop.built ? "" : " (not built yet)"}`
    );
    tick.title = `${i + 1}. ${stop.title}`;
    tick.addEventListener("click", () => onJump(i));
    railEl.append(tick);
    return tick;
  });
}

export function update(index) {
  ticks.forEach((tick, i) => {
    const isCurrent = i === index;
    tick.dataset.state = isCurrent ? "current" : i < index ? "past" : "ahead";
    if (isCurrent) tick.setAttribute("aria-current", "true");
    else tick.removeAttribute("aria-current");
  });
}
