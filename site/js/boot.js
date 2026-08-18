/* ==========================================================================
   boot.js — power on the room, and wire the three toolbar switches.

   The boot sequence is short on purpose: it exists so the first moment feels
   like switching on a machine, then it gets out of the way. If the reader has
   asked for less movement, the room is simply on when they arrive.
   ========================================================================== */

import * as store from "./core/store.js";

store.apply();

/* --------------------------------------------------------------------------
   ES modules only load over http://. Opening the file directly gives a blank
   room and a console error, so say plainly what to do instead.
   -------------------------------------------------------------------------- */
if (location.protocol === "file:") {
  const warn = document.getElementById("file-warning");
  if (warn) warn.hidden = false;
}

/* --------------------------------------------------------------------------
   The toolbar.
   -------------------------------------------------------------------------- */
const LABELS = {
  theme: { auto: "Light: auto", light: "Light: day", dark: "Light: night" },
  materials: { rich: "Surfaces: real", flat: "Surfaces: flat" },
  motion: { auto: "Motion: on", reduced: "Motion: still" },
};

function paintControl(button) {
  const key = button.dataset.setting;
  const value = store.get(key);
  const options = Object.keys(LABELS[key]);
  // The first option is the resting state ("auto" / "real"), so the lamp is
  // lit only when the reader has actively chosen something else.
  const changed = value !== options[0];

  button.querySelector(".ctrl-label").textContent = LABELS[key][value];
  button.setAttribute("aria-pressed", String(changed));
  button.setAttribute("title", LABELS[key][value] + " — click to change");

  const lamp = button.querySelector(".lamp");
  if (lamp) lamp.dataset.on = String(changed);
}

document.querySelectorAll("[data-setting]").forEach((button) => {
  const key = button.dataset.setting;
  const options = Object.keys(LABELS[key]);
  paintControl(button);
  button.addEventListener("click", () => {
    store.cycle(key, options);
    paintControl(button);
  });
});

/* --------------------------------------------------------------------------
   Power on.
   -------------------------------------------------------------------------- */
const lamps = [...document.querySelectorAll(".boot-lamps .lamp")];

function lightUp() {
  document.body.dataset.boot = "on";
  // Once the room is up, the boot panel must not be reachable by keyboard.
  const boot = document.getElementById("boot");
  if (boot) window.setTimeout(() => { boot.hidden = true; }, 700);
}

if (store.motionIsReduced()) {
  lamps.forEach((l) => { l.dataset.on = "true"; });
  lightUp();
} else {
  // Lamps come up one at a time, the way real panel lights do.
  lamps.forEach((lamp, i) => {
    window.setTimeout(() => { lamp.dataset.on = "true"; }, 120 + i * 130);
  });
  window.setTimeout(lightUp, 120 + lamps.length * 130 + 220);
}

/* Let the reader skip straight in. */
document.getElementById("boot")?.addEventListener("click", lightUp);
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && document.body.dataset.boot !== "on") lightUp();
});
