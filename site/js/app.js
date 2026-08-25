/* ==========================================================================
   app.js — the one entry point. Builds the track, then keeps four things in
   step with each other whenever the stop changes:

       the camera   slides the track
       the rail     marks where you are
       the bars     say which stop this is
       the page     announces the change and moves keyboard focus

   Everything routes through router.subscribe, so there is exactly one path
   that a stop change can take, no matter what triggered it.
   ========================================================================== */

import * as store from "./core/store.js";
import { warnIfOpenedFromDisk, wireToolbar, powerOn } from "./boot.js";
import { STOPS, COUNT, areaLabel } from "./core/stops.js";
import * as router from "./core/router.js";
import * as camera from "./core/camera.js";
import * as rail from "./ui/rail.js";
import * as map from "./ui/map.js";
import * as data from "./core/data.js";
import * as glossary from "./ui/glossary.js";
import { fillFacts } from "./ui/provenance.js";

store.apply();
warnIfOpenedFromDisk();
wireToolbar();

/* --------------------------------------------------------------------------
   Elements
   -------------------------------------------------------------------------- */
const trackEl    = document.getElementById("track");
const backdropEl = document.getElementById("backdrop");
const railEl     = document.getElementById("rail");
const mapEl      = document.getElementById("map");
const whereNowEl = document.getElementById("where-now");
const areaEl     = document.getElementById("brand-area");
const announceEl = document.getElementById("announce");
const prevBtn    = document.querySelector('[data-nav="prev"]');
const nextBtn    = document.querySelector('[data-nav="next"]');
const openMapBtn = document.getElementById("open-map");

/* --------------------------------------------------------------------------
   Build one empty shell per stop. They all exist from the start so the track
   has its full width and the camera's arithmetic stays simple; the CONTENT
   of a stop is only built when someone walks to it.
   -------------------------------------------------------------------------- */
const sections = STOPS.map((stop, i) => {
  const section = document.createElement("section");
  section.className = "stop";
  section.id = `stop-${stop.id}`;
  section.dataset.index = String(i);
  section.tabIndex = -1;
  section.setAttribute("aria-label", `Stop ${i + 1} of ${COUNT}: ${stop.title}`);
  section.setAttribute("inert", "");

  const inner = document.createElement("div");
  inner.className = "stop-inner";
  section.append(inner);
  trackEl.append(section);
  return section;
});

/* --------------------------------------------------------------------------
   Filling a stop
   -------------------------------------------------------------------------- */
const filled = new Set();

function placeholderFor(stop, index) {
  const card = document.createElement("article");
  card.className = "card mat-panel is-pending";
  card.innerHTML = `
    <p class="overline label">
      <span></span>
      <span class="bar" aria-hidden="true"></span>
      <span>Stop ${index + 1} of ${COUNT}</span>
    </p>
    <h2></h2>
    <p class="lead"></p>
    <p class="pending-plate mat-recess">
      <span class="lamp" data-on="false" aria-hidden="true"></span>
      <span>Not built yet &mdash; arrives in build step ${stop.step}</span>
    </p>
  `;
  // Set as text, never as HTML: content never gets to inject markup.
  card.querySelector(".overline span").textContent = areaLabel(stop.area);
  card.querySelector("h2").textContent = stop.title;
  card.querySelector(".lead").textContent = stop.blurb;
  return card;
}

function errorFor(stop, error) {
  const card = document.createElement("article");
  card.className = "card mat-panel";
  card.innerHTML = `<h2></h2><p class="lead"></p><pre class="err mat-recess"></pre>`;
  card.querySelector("h2").textContent = stop.title;
  card.querySelector(".lead").textContent =
    "This stop failed to load. The rest of the site is unaffected.";
  card.querySelector(".err").textContent = String(error);
  return card;
}

async function fill(index) {
  if (filled.has(index)) return;
  filled.add(index);

  const stop = STOPS[index];
  const inner = sections[index].querySelector(".stop-inner");

  if (!stop.built) {
    inner.append(placeholderFor(stop, index));
    return;
  }

  try {
    const module = await import(`./stations/${stop.id}.js`);
    await module.mount(inner, { stop, index });
    // A stop writes <span data-fact="..."> and <span data-term="...">; the
    // real figures and definitions are swapped in here, so no stop has to
    // know where a number came from — only which one it wants.
    fillFacts(inner);
    glossary.fillTerms(inner);
  } catch (error) {
    filled.delete(index);            // let a later visit try again
    inner.append(errorFor(stop, error));
    console.error(`[stop:${stop.id}]`, error);
  }
}

/* --------------------------------------------------------------------------
   One place where a stop change is handled
   -------------------------------------------------------------------------- */
function onStopChange(index, stop, reason) {
  const animate = reason !== "initial" && reason !== "correct";

  camera.go(index, { animate });

  // Only the stop you are on may be reached by keyboard or read by a screen
  // reader. `inert` does both, and unlike aria-hidden it cannot leave a
  // focusable element stranded inside a hidden region.
  sections.forEach((section, i) => {
    const isCurrent = i === index;
    section.dataset.current = String(isCurrent);
    if (isCurrent) section.removeAttribute("inert");
    else section.setAttribute("inert", "");
  });

  fill(index);
  fill(index + 1);                   // so the next slide is not empty mid-flight
  if (index > 0) fill(index - 1);

  rail.update(index);

  const counter = `${String(index + 1).padStart(2, "0")} / ${COUNT}`;
  whereNowEl.textContent = `${counter} · ${stop.title}`;
  areaEl.textContent = areaLabel(stop.area);
  document.title = `${index + 1}. ${stop.title} — The story point workshop`;

  prevBtn.disabled = index === 0;
  nextBtn.disabled = index === COUNT - 1;

  if (reason !== "initial") {
    announceEl.textContent = `Stop ${index + 1} of ${COUNT}: ${stop.title}`;
    // Put keyboard focus where the reader's attention now is, without the
    // browser scrolling the track sideways to "reveal" it.
    sections[index].focus({ preventScroll: true });
  }
}

/* --------------------------------------------------------------------------
   Wiring
   -------------------------------------------------------------------------- */
camera.init({ trackEl, backdropEl });
rail.init({ el: railEl, onJump: (i) => router.goTo(i) });
map.init({ el: mapEl, onSelect: (i) => router.goTo(i) });

prevBtn.addEventListener("click", () => router.prev());
nextBtn.addEventListener("click", () => router.next());
openMapBtn.addEventListener("click", () => map.open(router.currentIndex()));

/* Keys. Ignored while typing, so stops with text inputs keep working later. */
window.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (document.body.dataset.boot !== "on") return;

  const target = event.target;
  if (target?.closest?.("input, textarea, select, [contenteditable='']," +
                        " [contenteditable='true']")) return;

  // M works both ways: it opens the map, and it closes it again.
  if (event.key === "m" || event.key === "M") {
    map.toggle(router.currentIndex());
    event.preventDefault();
    return;
  }

  // Everything else belongs to the dialog while it is open: Escape closes it,
  // Tab cycles inside it, and the stops must not move underneath it.
  if (map.isOpen()) return;

  switch (event.key) {
    case "ArrowRight": router.next(); break;
    case "ArrowLeft":  router.prev(); break;
    case "Home":       router.first(); break;
    case "End":        router.last(); break;
    default: return;
  }
  event.preventDefault();
});

/* The numbers and the glossary load before the first stop is drawn, so a fact
   never appears as a dash for a moment and then pops into place. Both fail
   softly: if their files are missing the site still runs, and says so. */
await data.init();
await glossary.init();

router.subscribe(onStopChange);
router.start();
powerOn();
