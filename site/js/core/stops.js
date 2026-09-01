/* ==========================================================================
   stops.js — the path through the workshop.

   This is the spine of the whole site: 25 stops, in the order a reader should
   meet them. Each stop exists to answer the question the stop before it ran
   into. Nothing here is decoration — the router, the rail, the map and the
   camera all read this one list.

   Fields
     id     the slug used in the address bar (#/archive)
     title  what the stop is called
     blurb  one plain sentence: what you will understand after it
     area   which part of the story it belongs to
     step   which build step creates it (so the site can say so honestly)
     built  true once its content exists; false shows a "not built yet" plate
   ========================================================================== */

export const AREAS = [
  { id: "why",     label: "Why this exists" },
  { id: "data",    label: "The data" },
  { id: "guess",   label: "Turning text into a guess" },
  { id: "together",label: "Training together" },
  { id: "trust",   label: "Can we trust it" },
  { id: "results", label: "Reading the results" },
  { id: "loose",   label: "Loose ends" },
];

export const STOPS = [
  /* ---- why this exists ------------------------------------------------ */
  { id: "front-door", area: "why", step: 5, built: true,
    title: "The front door",
    blurb: "What this project is, and the three questions it sets out to answer." },

  { id: "poker-table", area: "why", step: 6, built: true,
    title: "The poker table",
    blurb: "Why teams guess effort at all, and why one team's 8 is another team's 3." },

  { id: "why-not-pool", area: "why", step: 7, built: true,
    title: "Why not just pool it",
    blurb: "What would have to leave the building, against what actually leaves." },

  /* ---- the data ------------------------------------------------------- */
  { id: "archive", area: "data", step: 8, built: true,
    title: "The archive",
    blurb: "Nineteen projects, 42,002 issues, and the odd corners in them." },

  { id: "cleaning-bench", area: "data", step: 9, built: false,
    title: "The cleaning bench",
    blurb: "Turning messy tracker text into something a model can actually read." },

  { id: "cleaning-rest", area: "data", step: 10, built: false,
    title: "Links, numbers, leftovers",
    blurb: "The rest of the cleaning, and the alarm that refuses uncleaned data." },

  { id: "splitting", area: "data", step: 11, built: false,
    title: "Cutting the data three ways",
    blurb: "Train, check, final test — and why later issues make the honest test." },

  /* ---- turning text into a guess -------------------------------------- */
  { id: "reading-machine", area: "guess", step: 12, built: false,
    title: "The reading machine",
    blurb: "How a sentence becomes numbers, and what gets cut off when it runs long." },

  { id: "extras", area: "guess", step: 13, built: false,
    title: "Type and priority",
    blurb: "The two extra facts we feed in beside the words." },

  { id: "patches", area: "guess", step: 14, built: false,
    title: "Patches, not models",
    blurb: "Why we send 252,484 numbers instead of 125 million." },

  { id: "merging", area: "guess", step: 15, built: false,
    title: "The merging problem",
    blurb: "Why merging patches the obvious way breaks, and the one change that fixes it." },

  { id: "ordered", area: "guess", step: 16, built: false,
    title: "The answers are ordered",
    blurb: "Why guessing 8 for a 3 should hurt more than guessing 2." },

  /* ---- training together ---------------------------------------------- */
  { id: "one-turn", area: "together", step: 17, built: false,
    title: "One project's turn",
    blurb: "What a single project does with its own data, and what holds it to the group." },

  { id: "server-rack", area: "together", step: 18, built: false,
    title: "The server rack",
    blurb: "How eighteen separate patches become one." },

  { id: "main-floor", area: "together", step: 19, built: false,
    title: "The main floor",
    blurb: "Sixty rounds of training, running, with the real curves." },

  { id: "head-start", area: "together", step: 20, built: false,
    title: "The head start",
    blurb: "Warming the model up on the biggest project first — and what that costs us." },

  { id: "line-up", area: "together", step: 21, built: false,
    title: "The line-up",
    blurb: "Every approach side by side, with the real numbers." },

  /* ---- can we trust it ------------------------------------------------ */
  { id: "fair-judging", area: "trust", step: 22, built: false,
    title: "Fair judging",
    blurb: "Why every approach has to be judged the same way to be compared at all." },

  { id: "same-twice", area: "trust", step: 23, built: false,
    title: "The same run twice",
    blurb: "Stopping and resuming a run without changing the answer." },

  /* ---- reading the results -------------------------------------------- */
  { id: "dials", area: "results", step: 24, built: false,
    title: "The dials",
    blurb: "What each score means, and why most mistakes are near-misses." },

  { id: "lying-score", area: "results", step: 25, built: false,
    title: "The score that lies",
    blurb: "Zero in every project, half a point the moment you combine them." },

  { id: "scales", area: "results", step: 26, built: false,
    title: "The scales",
    blurb: "One megabyte against five hundred, per project, per round." },

  /* ---- loose ends ----------------------------------------------------- */
  { id: "new-team", area: "loose", step: 27, built: false,
    title: "A new team joins",
    blurb: "How much of its own history a newcomer needs before joining pays off." },

  { id: "is-it-real", area: "loose", step: 28, built: false,
    title: "Is the difference real?",
    blurb: "What the statistics can say, and what they cannot." },

  { id: "ledger", area: "loose", step: 29, built: false,
    title: "The library and the ledger",
    blurb: "What this work stands on, and what it honestly cannot claim." },
];

/* ---- lookups the rest of the site uses ---------------------------------- */

export const COUNT = STOPS.length;

export function indexOf(id) {
  return STOPS.findIndex((s) => s.id === id);
}

export function at(index) {
  return STOPS[clampIndex(index)];
}

export function clampIndex(index) {
  if (!Number.isFinite(index)) return 0;
  return Math.min(Math.max(Math.trunc(index), 0), COUNT - 1);
}

export function areaLabel(areaId) {
  return AREAS.find((a) => a.id === areaId)?.label ?? "";
}

/** Stops grouped by area, in path order — what the map is drawn from. */
export function byArea() {
  return AREAS.map((area) => ({
    ...area,
    stops: STOPS.filter((s) => s.area === area.id),
  })).filter((group) => group.stops.length > 0);
}
