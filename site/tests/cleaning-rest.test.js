import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const rest = await import("../js/stations/cleaning-rest.js");
const categorical = await data.categorical();
const validation = await data.validation();

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "cleaning-rest");
check("cleaning-rest is marked built", entry.built === true);
check("it is stop 6", stops.indexOf("cleaning-rest") === 5);

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await rest.mount(host, { stop: entry, index: 5, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "Links, numbers, leftovers");

section("markers");
const baBoxes = card.querySelectorAll(".ba-box");
check("a before/after pair is shown", baBoxes.length === 2, String(baBoxes.length));
const after = baBoxes[1].querySelector(".ba-text").textContent;
check("the 'after' really is the cleaner's output",
  after === "See [ISSUE_REF] and the log at [URL]", after);
const text = deepText(card);
check("it explains why ticket numbers must go",
  text.includes("memorising a project") || text.includes("numbering"));

section("the length floor");
const drops = card.querySelectorAll(".drop-row");
check("the affected projects are named", drops.length === categorical.dropped_short.length,
  `${drops.length} vs ${categorical.dropped_short.length}`);
const dropTotal = categorical.dropped_short.reduce((s, d) => s + d.rows, 0);
check("they add up to the corpus-wide figure",
  dropTotal === data.value("corpus.rows_dropped"),
  `${dropTotal} vs ${data.value("corpus.rows_dropped")}`);

section("priorities");
const tables = card.querySelectorAll("table");
check("two tables: priorities and the alarm", tables.length === 2, String(tables.length));
const priorityRows = tables[0].querySelectorAll("tr").filter((r) => r.children.some((c) => c.tagName === "TD"));
check("every raw priority word is listed",
  priorityRows.length === categorical.priorities.length,
  `${priorityRows.length} vs ${categorical.priorities.length}`);
check("the flattening really does reduce 17 words to 6",
  data.value("categorical.priority_raw") === 17 && data.value("categorical.priority_canonical") === 6,
  `${data.value("categorical.priority_raw")} -> ${data.value("categorical.priority_canonical")}`);
const plateValues = card.querySelectorAll(".stat-value").map((v) => v.textContent);
check("the missing-priority share is shown on a plate",
  plateValues.some((v) => v.includes(data.text("categorical.priority_missing_pct"))),
  plateValues.join(" / "));
check("plate numbers carry their provenance (hoverable, not plain text)",
  card.querySelectorAll(".stat-value").every((v) => v.children.some((c) => c.className === "fact")),
  "some plates hold plain text");
check("it admits the unmapped word loses information",
  text.includes("categorical.priority_unmapped") && text.includes("loss of information"));
check("it explains why types are NOT merged",
  text.includes("not</em> mapped together") || text.includes("deliberately not merged"));

section("the alarm, on this machine's real files");
const failing = validation.files.filter((f) => !f.passes).length;
check("the run really did refuse files here", failing === 19, String(failing));
check("the banner states the real count",
  text.includes(`${failing} of ${validation.files.length} files on this machine fail`));
check("it says this is real output, not a demonstration",
  text.includes("not a demonstration"));
const alarmRows = tables[1].querySelectorAll("tr").filter((r) => r.children.some((c) => c.tagName === "TD"));
check("every file is listed with a verdict", alarmRows.length === validation.files.length,
  String(alarmRows.length));
check("refused files are marked",
  alarmRows.filter((r) => r.dataset.role === "refused").length === failing,
  String(alarmRows.filter((r) => r.dataset.role === "refused").length));
check("the reason is shortened but kept",
  alarmRows[0].children[2].textContent.length > 10 &&
  !alarmRows[0].children[2].textContent.includes("Re-export"),
  alarmRows[0].children[2].textContent);

section("the 1% threshold is justified, not asserted");
check("it explains the quote-mark title", text.includes("quoting an error message"));
check("it explains the typo'd noformat", text.includes("noformat)"));
check("it says why zero tolerance would be wrong",
  text.includes("force someone to") || text.includes("correct data"));

section("facts all resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();

