/* ==========================================================================
   parts-page.js — builds the reference page at /site/parts.html.

   Two jobs. It is a catalogue, so later stops reach for a part that already
   exists instead of inventing a ninth kind of slider. And it is a test bench:
   if something here stops working, it stops working visibly, on one page,
   instead of quietly inside one stop out of twenty-five.
   ========================================================================== */

import * as store from "./core/store.js";
import { warnIfOpenedFromDisk, wireToolbar } from "./boot.js";
import {
  createSwitch, createDial, createSlider, createSegmented, createStepper,
} from "./ui/controls.js";
import {
  createReadout, createMeter, createGauge, createStatPlate, createDrawer,
} from "./ui/readouts.js";
import { num, pct, bytes, times } from "./ui/format.js";

store.apply();
warnIfOpenedFromDisk();
wireToolbar();

const page = document.getElementById("page");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function section(title, blurb) {
  const wrap = el("section", "page-section");
  wrap.append(el("h2", "section-title", title));
  if (blurb) wrap.append(el("p", "section-blurb", blurb));
  const grid = el("div", "demo-grid");
  wrap.append(grid);
  page.append(wrap);
  return grid;
}

function demo(parent, name, note, ...parts) {
  const card = el("div", "demo mat-panel");
  const head = el("div", "demo-head");
  head.append(el("span", "demo-name", name));
  card.append(head);
  if (note) card.append(el("p", "demo-note", note));
  const stage = el("div", "demo-stage");
  parts.forEach((p) => stage.append(p));
  card.append(stage);
  parent.append(card);
  return card;
}

/* ==========================================================================
   1. Controls
   ========================================================================== */
const controls = section(
  "Things you can touch",
  "Each one reports a number. What that number means is decided by the stop using it, never by the control."
);

/* -- switch -- */
const switchState = el("p", "demo-state", "off");
const freeze = createSwitch({
  label: "Freeze the shared half",
  hint: "Used at the merging stop, where freezing one half is the whole point.",
  onChange: (on) => { switchState.textContent = on ? "on" : "off"; },
});
demo(controls, "Switch", "One thing, on or off.", freeze.el, switchState);

/* -- dial -- */
const dialState = el("p", "demo-state", "0.010");
const mu = createDial({
  label: "Pull towards the group",
  min: 0, max: 0.1, step: 0.001, value: 0.01,
  onInput: (v) => { dialState.textContent = v.toFixed(3); },
});
demo(controls, "Dial",
  "A continuous value. Drag it up and down, or use the arrow keys.",
  mu.el, dialState);

/* -- slider -- */
const sliderState = el("p", "demo-state", "20%");
const testSize = createSlider({
  label: "Held back for the final test",
  min: 0.05, max: 0.4, step: 0.05, value: 0.2,
  format: (v) => pct(v, { digits: 0 }),
  onInput: (v) => { sliderState.textContent = pct(v, { digits: 0 }); },
});
demo(controls, "Slider",
  "The same idea, laid out flat. A real range input, so it behaves the way people expect.",
  testSize.el, sliderState);

/* -- segmented -- */
const segState = el("p", "demo-state", "random");
const splitMode = createSegmented({
  label: "How to split the data",
  options: [{ value: "random", label: "At random" }, { value: "temporal", label: "By date" }],
  value: "random",
  onChange: (v) => { segState.textContent = v; },
});
demo(controls, "Segmented",
  "Pick one of a few. Arrow keys move between the choices.",
  splitMode.el, segState);

/* -- stepper -- */
const stepState = el("p", "demo-state", "3");
const epochs = createStepper({
  label: "Passes over the data",
  min: 1, max: 10, step: 1, value: 3,
  onChange: (v) => { stepState.textContent = String(v); },
});
demo(controls, "Stepper", "A whole number, nudged.", epochs.el, stepState);

/* ==========================================================================
   2. Readouts
   ========================================================================== */
const readouts = section(
  "Things that show a number back",
  "Numbers count up to their new value rather than jumping, because a number that moves tells you which way it went. That stops when Motion is set to still."
);

const sent = createReadout({
  label: "Numbers sent per project, per round",
  value: 252484,
  format: (v) => num(Math.round(v)),
  caption: "The real figure from the run on disk.",
});
demo(readouts, "Readout", "One number, in a lit well.", sent.el);

const share = createMeter({
  label: "Share of the model that travels",
  min: 0, max: 1, value: 0.002019,
  marker: 0.5, markerLabel: "half the model",
  format: (v) => pct(v, { digits: 2 }),
});
demo(readouts, "Meter",
  "A share, with an optional mark showing what it is being judged against.",
  share.el);

const kappa = createGauge({
  label: "Agreement, beyond luck",
  min: 0, max: 1, value: 0.317,
  caption: "0 means no better than guessing; 1 means perfect.",
});
demo(readouts, "Gauge",
  "For a score whose position on the scale matters more than its digits.",
  kappa.el);

const plates = el("div", "part-grid");
[
  createStatPlate({ label: "Per project, per round", value: bytes(1009936), caption: "against 500 MB if we sent the whole model" }),
  createStatPlate({ label: "Smaller by", value: times(495) }),
  createStatPlate({ label: "Projects", value: "18", tone: "quiet" }),
].forEach((p) => plates.append(p.el));
demo(readouts, "Stat plate", "The workhorse of the results stops.", plates);

/* ==========================================================================
   3. Containers
   ========================================================================== */
const containers = section(
  "Things that hold other things",
  "Detail that would drown the main point, kept one click away."
);

const drawer = createDrawer({
  label: "Where these numbers come from",
  summary: "results/communication_cost.json",
});
drawer.body.textContent =
  "From build step 4 onward, every figure on this site is produced by a script "
  + "that reads the project's own data and results. None are typed by hand, and "
  + "hovering any of them will show its source.";
demo(containers, "Drawer", "Open it if you want the detail.", drawer.el);

const tableWrap = el("div", "table-wrap");
const table = el("table", "data");
table.innerHTML = `
  <thead><tr><th>Approach</th><th>MAE</th><th>Agreement</th></tr></thead>
  <tbody>
    <tr><td>Federated</td><td class="n">1.2156</td><td class="n">0.3171</td></tr>
    <tr><td>Each project alone</td><td class="n">1.2182</td><td class="n">0.3294</td></tr>
    <tr><td>All data pooled</td><td class="n">1.3241</td><td class="n">0.2981</td></tr>
  </tbody>`;
tableWrap.append(table);
demo(containers, "Table",
  "Real numbers from the run on disk. Scrolls sideways on its own when the screen is narrow.",
  tableWrap);

const badges = el("div", "part-row");
["Preliminary", "Simulated", "No run loaded"].forEach((word, i) => {
  const badge = el("span", `badge${i === 1 ? " is-dashed" : ""}${i === 0 ? " is-strong" : ""}`, word);
  badges.append(badge);
});
demo(containers, "Badge",
  "A stamped word. Step 4 builds the real provenance stamps on top of this.",
  badges);

/* ==========================================================================
   4. Everything wired together
   ========================================================================== */
const wired = section(
  "One control driving three readouts",
  "Every part shares the same shape — el, get, set — so a stop can wire any control to any display without either knowing about the other. This is the whole reason the bin exists."
);

const outMeter = createMeter({ label: "As a share", min: 0, max: 1, value: 0.35, format: (v) => pct(v, { digits: 0 }) });
const outGauge = createGauge({ label: "On a dial", min: 0, max: 1, value: 0.35 });
const outRead  = createReadout({ label: "As a count out of 42,002 issues", value: 14700, format: (v) => num(Math.round(v)), size: "sm" });

const driver = createSlider({
  label: "Drag me",
  min: 0, max: 1, step: 0.01, value: 0.35,
  format: (v) => pct(v, { digits: 0 }),
  onInput: (v) => {
    outMeter.set(v);
    outGauge.set(v);
    outRead.set(v * 42002);
  },
});

const bank = el("div", "part-grid");
[outMeter, outGauge, outRead].forEach((p) => bank.append(p.el));
demo(wired, "Wired up", null, driver.el, bank);
