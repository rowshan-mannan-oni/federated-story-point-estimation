/* ==========================================================================
   boot.js — powering the room on, and the three toolbar switches.

   Exports functions rather than running on import, so app.js decides the
   order things happen in. The power-on is short on purpose: it exists so the
   first moment feels like switching on a machine, then it gets out of the way.
   If the reader has asked for less movement, the room is simply on when they
   arrive.
   ========================================================================== */

import * as store from "./core/store.js";

const LABELS = {
  theme:     { auto: "Light: auto",    light: "Light: day",    dark: "Light: night" },
  materials: { rich: "Surfaces: real", flat:  "Surfaces: flat" },
  motion:    { auto: "Motion: on",     reduced: "Motion: still" },
};

/* --------------------------------------------------------------------------
   ES modules only load over http://. Opening the file straight from disk
   gives a blank page and a console error, so say plainly what to do instead.
   -------------------------------------------------------------------------- */
export function warnIfOpenedFromDisk() {
  if (location.protocol !== "file:") return false;
  const warn = document.getElementById("file-warning");
  if (warn) warn.hidden = false;
  return true;
}

function paintControl(button) {
  const key = button.dataset.setting;
  const value = store.get(key);
  const options = Object.keys(LABELS[key]);
  // The first option is the resting state ("auto" / "real"), so the lamp is
  // lit only when the reader has actively chosen something else.
  const changed = value !== options[0];

  button.querySelector(".ctrl-label").textContent = LABELS[key][value];
  button.setAttribute("aria-pressed", String(changed));
  button.setAttribute("title", `${LABELS[key][value]} — click to change`);

  const lamp = button.querySelector(".lamp");
  if (lamp) lamp.dataset.on = String(changed);
}

export function wireToolbar() {
  document.querySelectorAll("[data-setting]").forEach((button) => {
    const key = button.dataset.setting;
    const options = Object.keys(LABELS[key]);
    paintControl(button);
    button.addEventListener("click", () => {
      store.cycle(key, options);
      paintControl(button);
    });
  });
}

export function powerOn() {
  const boot = document.getElementById("boot");
  const lamps = [...document.querySelectorAll(".boot-lamps .lamp")];

  const finish = () => {
    document.body.dataset.boot = "on";
    // Once the room is up the panel must not be reachable by keyboard.
    window.setTimeout(() => { if (boot) boot.hidden = true; }, 520);
  };

  if (store.motionIsReduced()) {
    lamps.forEach((l) => { l.dataset.on = "true"; });
    finish();
  } else {
    lamps.forEach((lamp, i) => {
      window.setTimeout(() => { lamp.dataset.on = "true"; }, 110 + i * 120);
    });
    window.setTimeout(finish, 110 + lamps.length * 120 + 180);
  }

  boot?.addEventListener("click", finish);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.dataset.boot !== "on") finish();
  });
}
