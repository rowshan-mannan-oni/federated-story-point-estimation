/* ==========================================================================
   data.js — everything the site knows, loaded once.

   The numbers come from site/data/*.json, which site/tools/extract_facts.py
   writes by reading the project's own data and results. Nothing here invents
   a value: if a file is missing, the site says so rather than filling the gap.

   A fact looks like this:

     { value: 42002, text: "42,002", unit: null,
       source: "data_to_train_on/*.csv",
       how:    "summed the rows in every file",
       kind:   "measured" | "run" | "derived" | "missing" }

   `kind` is what the stamps are drawn from, so a reader can always tell a
   measurement from an assumption.
   ========================================================================== */

const BASE = new URL("../../data/", import.meta.url);

const cache = new Map();
let facts = null;
let about = null;
let failure = null;

async function loadJson(name) {
  if (cache.has(name)) return cache.get(name);

  const promise = fetch(new URL(`${name}.json`, BASE))
    .then((response) => {
      if (!response.ok) throw new Error(`${name}.json — ${response.status}`);
      return response.json();
    })
    .catch((error) => {
      // A missing data file is a state the site is designed for, not a crash.
      console.warn(`[data] ${name}.json could not be loaded:`, error.message);
      return null;
    });

  cache.set(name, promise);
  return promise;
}

/** Load the fact table. Call once, early; everything else can wait. */
export async function init() {
  const payload = await loadJson("facts");
  if (!payload) {
    failure = "site/data/facts.json is missing — run site/tools/extract_facts.py";
    facts = {};
    about = null;
    return false;
  }
  facts = payload.facts ?? {};
  about = payload.about ?? null;
  return true;
}

const MISSING = Object.freeze({
  value: null,
  text: "—",
  unit: null,
  source: "not generated",
  how: "this figure has not been extracted yet",
  kind: "missing",
});

/** A fact, or a clearly-marked stand-in. Never throws, never invents. */
export function fact(key) {
  if (!facts) return { ...MISSING, how: "facts have not finished loading" };
  return facts[key] ?? { ...MISSING, how: `no fact called "${key}"` };
}

export function value(key) { return fact(key).value; }
export function text(key)  { return fact(key).text; }
export function has(key)   { return Boolean(facts && key in facts); }
export function all()      { return facts ? { ...facts } : {}; }

/** What was available when the numbers were generated. */
export function provenance() {
  return { about, failure, ready: facts !== null };
}

/* The bigger tables, loaded only by the stops that need them. */
export const dataset    = () => loadJson("dataset");
export const validation = () => loadJson("validation");
export const split      = () => loadJson("split");
export const run        = () => loadJson("run");
export const params      = () => loadJson("params");
export const glossary    = () => loadJson("glossary");
export const examples    = () => loadJson("examples");
export const calibration = () => loadJson("calibration");
export const cleaning    = () => loadJson("cleaning");
export const categorical = () => loadJson("categorical");
