/* ==========================================================================
   facts-page.js — builds the reference page at /site/facts.html.

   This page is the site holding itself to account. Everything the walk-through
   is allowed to say, in one list, each entry with its source and method. If a
   number is not here, no stop may show it.
   ========================================================================== */

import * as store from "./core/store.js";
import { warnIfOpenedFromDisk, wireToolbar } from "./boot.js";
import * as data from "./core/data.js";
import * as glossary from "./ui/glossary.js";
import { factEl } from "./ui/provenance.js";
import { stamp, banner } from "./ui/stamp.js";

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

/* Facts are named area.thing, so the prefix groups them for free. */
const GROUP_TITLES = {
  corpus:     "The corpus",
  cleaning:   "Cleaning",
  text:       "How long the issues are",
  validation: "Is this data cleaned?",
  split:      "The split",
  params:     "What gets trained",
  comms:      "What gets sent",
  run:        "The run itself",
  result:     "Results",
  confusion:  "Where the mistakes land",
  rounds:     "The training curve",
  baseline:   "Simple comparisons",
  artifact:   "The score that lies",
};

async function build() {
  const loaded = await data.init();
  const { about, failure } = data.provenance();

  /* ---- what was available when the numbers were made ------------------- */
  const head = el("section", "page-section");
  head.append(el("h2", "section-title", "Where these came from"));

  if (!loaded) {
    head.append(banner("missing",
      failure || "The fact file has not been generated yet."));
    page.append(head);
    return;
  }

  const card = el("div", "facts-about mat-panel");
  const list = document.createElement("dl");
  const rows = [
    ["Generated", about?.generated],
    ["Issue data", about?.data_dir],
    ["Results", about?.results_dir],
  ];
  for (const [term, value] of rows) {
    list.append(el("dt", null, term));
    list.append(el("dd", null, value ?? "—"));
  }
  card.append(list);

  const state = el("p", null);
  state.append(stamp(about?.data_available ? "measured" : "missing"));
  state.append(document.createTextNode(
    about?.data_available
      ? " The issue files are on this machine, so the data figures are real measurements."
      : " The issue files are not on this machine, so the data figures are unavailable."));
  card.append(state);

  const runState = el("p", null);
  runState.append(stamp(about?.run_available ? "run" : "missing"));
  runState.append(document.createTextNode(
    about?.run_available
      ? " A finished run is present, so the result figures come from it."
      : " No finished run is present, so the result figures are unavailable."));
  card.append(runState);

  for (const note of about?.notes ?? []) {
    card.append(banner("preliminary", note));
  }
  head.append(card);
  page.append(head);

  /* ---- every fact, grouped ------------------------------------------- */
  const facts = data.all();
  const groups = new Map();
  for (const key of Object.keys(facts)) {
    const prefix = key.split(".")[0];
    if (!groups.has(prefix)) groups.set(prefix, []);
    groups.get(prefix).push(key);
  }

  const counts = {};
  for (const entry of Object.values(facts)) {
    counts[entry.kind] = (counts[entry.kind] ?? 0) + 1;
  }

  const summary = el("section", "page-section");
  summary.append(el("h2", "section-title", `${Object.keys(facts).length} figures`));
  summary.append(el("p", "section-blurb",
    Object.entries(counts)
      .map(([kind, n]) => `${n} ${kind}`)
      .join(" · ") +
    ". Hover or tab to any number on the site to see this same information."));
  page.append(summary);

  for (const [prefix, keys] of groups) {
    const section = el("section", "page-section facts-group");
    section.append(el("h2", "section-title", GROUP_TITLES[prefix] ?? prefix));

    const table = el("table", "facts-table mat-panel");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const label of ["Figure", "Value", "How it was worked out", "Source"]) {
      headRow.append(el("th", null, label));
    }
    thead.append(headRow);
    table.append(thead);

    const body = document.createElement("tbody");
    for (const key of keys.sort()) {
      const entry = facts[key];
      const row = document.createElement("tr");

      row.append(el("td", "k", key));

      const valueCell = el("td", "v");
      valueCell.append(factEl(key));
      row.append(valueCell);

      row.append(el("td", "h", entry.how));

      const sourceCell = el("td", "s");
      sourceCell.append(stamp(entry.kind));
      sourceCell.append(document.createTextNode(" " + entry.source));
      row.append(sourceCell);

      body.append(row);
    }
    table.append(body);
    section.append(table);
    page.append(section);
  }

  /* ---- the glossary --------------------------------------------------- */
  await glossary.init();
  const terms = glossary.allTerms();
  const gloss = el("section", "page-section facts-group");
  gloss.append(el("h2", "section-title", `The glossary — ${Object.keys(terms).length} terms`));
  gloss.append(el("p", "section-blurb",
    "Written by hand, not generated: these are words rather than measurements. " +
    "In the walk-through they appear underlined; hover or tab to one to read it."));

  const table = el("table", "facts-table mat-panel");
  const body = document.createElement("tbody");
  for (const [term, entry] of Object.entries(terms)) {
    const row = document.createElement("tr");
    const nameCell = el("td", "v");
    const marked = el("span");
    marked.dataset.term = term;
    marked.textContent = term;
    nameCell.append(marked);
    row.append(nameCell);
    row.append(el("td", "h", entry.short));
    body.append(row);
  }
  table.append(body);
  gloss.append(table);
  page.append(gloss);

  glossary.fillTerms(gloss);
}

build().catch((error) => {
  console.error(error);
  page.append(banner("missing", `The facts page failed to build: ${error.message}`));
});
