/* ==========================================================================
   map.js — the whole path on one screen.

   The rail shows where you are; the map shows what the journey is. Stops are
   grouped by the part of the story they belong to, so the shape of the
   argument is visible before you have walked it.

   Built on the browser's own <dialog>, deliberately: it gives a proper modal
   with Escape-to-close and focus kept inside, none of which is worth
   re-implementing by hand.
   ========================================================================== */

import { byArea, STOPS } from "../core/stops.js";

let dialogEl = null;
let buttons = [];
let onPick = () => {};

export function init({ el, onSelect }) {
  dialogEl = el;
  onPick = onSelect;
  buttons = [];                       // safe to call init again

  const body = el.querySelector("[data-map-body]");
  body.innerHTML = "";

  for (const group of byArea()) {
    const section = document.createElement("section");
    section.className = "map-group";

    const heading = document.createElement("h3");
    heading.className = "label";
    heading.textContent = group.label;
    section.append(heading);

    const list = document.createElement("ul");
    list.className = "map-list";

    for (const stop of group.stops) {
      const i = STOPS.indexOf(stop);
      const item = document.createElement("li");

      const button = document.createElement("button");
      button.type = "button";
      button.className = "map-item mat-sub";
      button.dataset.index = String(i);
      button.dataset.built = String(stop.built);
      button.innerHTML =
        `<span class="n">${String(i + 1).padStart(2, "0")}</span>` +
        `<span class="t"></span>` +
        `<span class="b"></span>`;
      // Titles and blurbs are set as text, never as HTML, so content can
      // never smuggle markup into the page.
      button.querySelector(".t").textContent = stop.title;
      button.querySelector(".b").textContent = stop.built
        ? stop.blurb
        : `${stop.blurb} — arrives in build step ${stop.step}`;

      button.addEventListener("click", () => {
        close();
        onPick(i);
      });

      item.append(button);
      list.append(item);
      buttons.push(button);
    }

    section.append(list);
    body.append(section);
  }

  // A click on the backdrop lands on the dialog element itself.
  dialogEl.addEventListener("click", (event) => {
    if (event.target === dialogEl) close();
  });

  el.querySelector("[data-map-close]")?.addEventListener("click", close);
}

export function open(currentIndex) {
  if (!dialogEl || dialogEl.open) return;
  buttons.forEach((b, i) => {
    if (i === currentIndex) b.setAttribute("aria-current", "true");
    else b.removeAttribute("aria-current");
  });
  dialogEl.showModal();
  // Start on the stop you are actually at, not at the top of the list.
  buttons[currentIndex]?.focus({ preventScroll: true });
}

export function close() {
  if (dialogEl?.open) dialogEl.close();
}

export function toggle(currentIndex) {
  if (dialogEl?.open) close();
  else open(currentIndex);
}

export function isOpen() {
  return Boolean(dialogEl?.open);
}
