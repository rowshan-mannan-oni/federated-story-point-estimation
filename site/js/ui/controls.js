/* ==========================================================================
   controls.js — the things a reader can touch.

   Every control follows the same contract, so a stop can swap one for another
   without rewriting anything:

       const part = createDial({ label, value, onInput });
       parent.append(part.el);
       part.get();        // read the value
       part.set(0.4);     // change it (does NOT fire onInput — no loops)

   Two rules hold across all of them:
     · Everything is operable from the keyboard, with the right ARIA role.
       A knob that only responds to dragging is a knob half the readers of
       this site cannot use.
     · A control never formats its own meaning. It reports a number; the stop
       decides what that number says.
   ========================================================================== */

import { quantize, fractionOf, decimalsFor, num } from "./format.js";

/* small helper: make an element with a class and optional text */
function make(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;

/* ==========================================================================
   SWITCH — on or off.
   ========================================================================== */
export function createSwitch({ label, checked = false, hint = "", onChange } = {}) {
  const el = make("div", "part part-switch");
  const button = make("button", "switch-track");
  const id = nextId("sw");

  button.type = "button";
  button.id = id;
  button.setAttribute("role", "switch");
  button.setAttribute("aria-checked", String(checked));
  button.append(make("span", "switch-thumb"));

  const text = make("label", "part-label", label);
  text.setAttribute("for", id);

  el.append(button, text);
  if (hint) {
    const hintEl = make("p", "part-hint", hint);
    hintEl.id = `${id}-hint`;
    button.setAttribute("aria-describedby", hintEl.id);
    el.append(hintEl);
  }

  let value = checked;

  function paint() {
    button.setAttribute("aria-checked", String(value));
    el.dataset.on = String(value);
  }

  button.addEventListener("click", () => {
    value = !value;
    paint();
    onChange?.(value);
  });

  paint();

  return {
    el,
    get: () => value,
    set(next) { value = Boolean(next); paint(); },
  };
}

/* ==========================================================================
   DIAL — a rotary knob for a continuous value.

   Drag up or down to turn it; the arrow keys do the same thing in steps.
   Vertical dragging, not circular: circular feels authentic and is miserable
   to control, and this site asks people to land on specific values.
   ========================================================================== */
export function createDial({
  label,
  min = 0,
  max = 1,
  step = 0.01,
  value = min,
  unit = "",
  format,
  onInput,
} = {}) {
  const el = make("div", "part part-dial");
  const knob = make("div", "dial-knob");
  const id = nextId("dial");
  const digits = decimalsFor(step);
  const show = format ?? ((v) => num(v, { digits }));

  knob.id = id;
  knob.tabIndex = 0;
  knob.setAttribute("role", "slider");
  knob.setAttribute("aria-valuemin", String(min));
  knob.setAttribute("aria-valuemax", String(max));

  const face = make("div", "dial-face");
  const pointer = make("i", "dial-pointer");
  face.append(pointer);
  knob.append(face);

  // Tick marks around the knob, drawn once.
  const ticks = make("div", "dial-ticks");
  const TICKS = 11;
  for (let i = 0; i < TICKS; i += 1) {
    const tick = make("i", "dial-tick");
    tick.style.setProperty("--a", `${-135 + (270 * i) / (TICKS - 1)}deg`);
    ticks.append(tick);
  }

  const dialBody = make("div", "dial-body");
  dialBody.append(ticks, knob);

  const text = make("label", "part-label", label);
  text.setAttribute("for", id);
  const readout = make("output", "dial-value");
  readout.setAttribute("for", id);

  el.append(dialBody, text, readout);

  let current = quantize(value, min, max, step);

  function paint() {
    const f = fractionOf(current, min, max);
    face.style.setProperty("--turn", `${-135 + f * 270}deg`);
    const shown = `${show(current)}${unit ? ` ${unit}` : ""}`;
    readout.textContent = shown;
    knob.setAttribute("aria-valuenow", String(current));
    knob.setAttribute("aria-valuetext", shown);
  }

  function commit(next, notify = true) {
    const snapped = quantize(next, min, max, step);
    if (snapped === current) return;
    current = snapped;
    paint();
    if (notify) onInput?.(current);
  }

  /* ---- dragging ---- */
  let dragFrom = null;

  knob.addEventListener("pointerdown", (event) => {
    knob.setPointerCapture(event.pointerId);
    dragFrom = { y: event.clientY, value: current };
    el.dataset.dragging = "true";
  });

  knob.addEventListener("pointermove", (event) => {
    if (!dragFrom) return;
    // 180px of travel covers the whole range: fine enough to be precise,
    // short enough that you never run out of screen.
    const delta = ((dragFrom.y - event.clientY) / 180) * (max - min);
    commit(dragFrom.value + delta);
  });

  const endDrag = (event) => {
    if (!dragFrom) return;
    dragFrom = null;
    delete el.dataset.dragging;
    if (knob.hasPointerCapture?.(event.pointerId)) knob.releasePointerCapture(event.pointerId);
  };
  knob.addEventListener("pointerup", endDrag);
  knob.addEventListener("pointercancel", endDrag);

  /* ---- keyboard ---- */
  knob.addEventListener("keydown", (event) => {
    const big = (max - min) / 10;
    let next = null;
    switch (event.key) {
      case "ArrowUp": case "ArrowRight": next = current + step; break;
      case "ArrowDown": case "ArrowLeft": next = current - step; break;
      case "PageUp":   next = current + big; break;
      case "PageDown": next = current - big; break;
      case "Home":     next = min; break;
      case "End":      next = max; break;
      default: return;
    }
    commit(next);
    event.preventDefault();
    event.stopPropagation();      // the page's own arrow keys must not fire
  });

  paint();

  return {
    el,
    get: () => current,
    set(next) { commit(next, false); },
  };
}

/* ==========================================================================
   SLIDER — a linear value.

   Built on a real <input type="range">: the browser already gives it correct
   keyboard behaviour, touch handling and screen-reader semantics, and none of
   that is worth rewriting for the sake of a custom look.
   ========================================================================== */
export function createSlider({
  label,
  min = 0,
  max = 1,
  step = 0.01,
  value = min,
  unit = "",
  format,
  onInput,
} = {}) {
  const el = make("div", "part part-slider");
  const id = nextId("sl");
  const digits = decimalsFor(step);
  const show = format ?? ((v) => num(v, { digits }));

  const head = make("div", "part-head");
  const text = make("label", "part-label", label);
  text.setAttribute("for", id);
  const readout = make("output", "part-value");
  readout.setAttribute("for", id);
  head.append(text, readout);

  const input = document.createElement("input");
  input.type = "range";
  input.id = id;
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(quantize(value, min, max, step));
  input.className = "slider-input";

  el.append(head, input);

  function paint() {
    const v = Number(input.value);
    const shown = `${show(v)}${unit ? ` ${unit}` : ""}`;
    readout.textContent = shown;
    input.setAttribute("aria-valuetext", shown);
    input.style.setProperty("--f", String(fractionOf(v, min, max)));
  }

  input.addEventListener("input", () => {
    paint();
    onInput?.(Number(input.value));
  });
  // Arrow keys belong to the slider here, not to the page.
  input.addEventListener("keydown", (event) => event.stopPropagation());

  paint();

  return {
    el,
    get: () => Number(input.value),
    set(next) { input.value = String(quantize(next, min, max, step)); paint(); },
  };
}

/* ==========================================================================
   SEGMENTED — pick one of a few. A radio group wearing a nicer coat.
   ========================================================================== */
export function createSegmented({ label, options = [], value, onChange } = {}) {
  const el = make("div", "part part-segmented");
  const group = make("div", "segmented");
  const groupId = nextId("seg");

  group.setAttribute("role", "radiogroup");
  if (label) {
    const text = make("span", "part-label", label);
    text.id = `${groupId}-label`;
    group.setAttribute("aria-labelledby", text.id);
    el.append(text);
  }

  let current = value ?? options[0]?.value;

  const buttons = options.map((option) => {
    const button = make("button", "segment", option.label);
    button.type = "button";
    button.setAttribute("role", "radio");
    button.dataset.value = String(option.value);
    button.addEventListener("click", () => pick(option.value));
    group.append(button);
    return button;
  });

  function paint() {
    buttons.forEach((button) => {
      const on = button.dataset.value === String(current);
      button.setAttribute("aria-checked", String(on));
      // Roving tabindex: one stop in the tab order, arrows move within.
      button.tabIndex = on ? 0 : -1;
    });
  }

  function pick(next, notify = true) {
    if (String(next) === String(current)) return;
    current = next;
    paint();
    if (notify) onChange?.(current);
  }

  group.addEventListener("keydown", (event) => {
    const i = buttons.findIndex((b) => b.tabIndex === 0);
    let target = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") target = (i + 1) % buttons.length;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") target = (i - 1 + buttons.length) % buttons.length;
    else return;
    pick(options[target].value);
    buttons[target].focus();
    event.preventDefault();
    event.stopPropagation();
  });

  el.append(group);
  paint();

  return {
    el,
    get: () => current,
    set(next) { pick(next, false); },
  };
}

/* ==========================================================================
   STEPPER — a whole number, nudged up or down.
   ========================================================================== */
export function createStepper({
  label, min = 0, max = 100, step = 1, value = min, unit = "", onChange,
} = {}) {
  const el = make("div", "part part-stepper");
  const id = nextId("st");

  const text = make("label", "part-label", label);
  text.setAttribute("for", id);

  const box = make("div", "stepper");
  const down = make("button", "stepper-btn", "−");
  const up = make("button", "stepper-btn", "+");
  const field = make("div", "stepper-value");

  down.type = "button"; up.type = "button";
  down.setAttribute("aria-label", `Decrease ${label}`);
  up.setAttribute("aria-label", `Increase ${label}`);

  field.id = id;
  field.tabIndex = 0;
  field.setAttribute("role", "spinbutton");
  field.setAttribute("aria-valuemin", String(min));
  field.setAttribute("aria-valuemax", String(max));

  box.append(down, field, up);
  el.append(text, box);

  let current = quantize(value, min, max, step);

  function paint() {
    const shown = `${num(current, { digits: decimalsFor(step) })}${unit ? ` ${unit}` : ""}`;
    field.textContent = shown;
    field.setAttribute("aria-valuenow", String(current));
    field.setAttribute("aria-valuetext", shown);
    down.disabled = current <= min;
    up.disabled = current >= max;
  }

  function commit(next, notify = true) {
    const snapped = quantize(next, min, max, step);
    if (snapped === current) return;
    current = snapped;
    paint();
    if (notify) onChange?.(current);
  }

  down.addEventListener("click", () => commit(current - step));
  up.addEventListener("click", () => commit(current + step));

  field.addEventListener("keydown", (event) => {
    switch (event.key) {
      case "ArrowUp": case "ArrowRight": commit(current + step); break;
      case "ArrowDown": case "ArrowLeft": commit(current - step); break;
      case "Home": commit(min); break;
      case "End": commit(max); break;
      default: return;
    }
    event.preventDefault();
    event.stopPropagation();
  });

  paint();

  return {
    el,
    get: () => current,
    set(next) { commit(next, false); },
  };
}
