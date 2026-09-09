import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const archive = await import("../js/stations/archive.js");
const dataset = await data.dataset();

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "archive");
check("archive is marked built", entry.built === true);
check("it is stop 4", stops.indexOf("archive") === 3);

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await archive.mount(host, { stop: entry, index: 3, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "The archive");

section("the cabinet");
const bodyRows = card.querySelectorAll("tr").filter((r) => r.children.some((c) => c.tagName === "TD"));
check("every project has a row", bodyRows.length === dataset.projects.length,
  `${bodyRows.length} rows vs ${dataset.projects.length} projects`);
check("all nineteen are there", bodyRows.length === 19, String(bodyRows.length));

const headers = card.querySelectorAll("th");
check("six columns", headers.length === 6, String(headers.length));
check("it starts sorted by size",
  headers.some((h) => h.dataset.sorted === "true" && h.getAttribute("aria-sort") === "descending"));
check("unsorted columns say so",
  headers.filter((h) => h.dataset.sorted !== "true")
         .every((h) => h.getAttribute("aria-sort") === "none" || h.getAttribute("aria-sort") === null));

const firstCellBefore = bodyRows[0].children[0].textContent;
check("the largest project sorts to the top",
  firstCellBefore.includes(String(data.value("corpus.largest_project")).replace(/_/g, " ")),
  firstCellBefore);

/* sorting by a different column must actually reorder */
const issuesHeader = headers[1].querySelector(".dtable-sort");
issuesHeader.fire("click");   // desc -> asc
const rowsAfter = card.querySelectorAll("tr").filter((r) => r.children.some((c) => c.tagName === "TD"));
check("clicking a header re-sorts",
  rowsAfter[0].children[0].textContent !== firstCellBefore,
  rowsAfter[0].children[0].textContent);
check("ascending puts the smallest first",
  rowsAfter[0].children[0].textContent.includes(String(data.value("corpus.smallest_project")).replace(/_/g, " ")),
  rowsAfter[0].children[0].textContent);
check("the sort direction is announced",
  headers[1].getAttribute("aria-sort") === "ascending", headers[1].getAttribute("aria-sort"));

section("the two special projects");
const marked = rowsAfter.filter((r) => r.dataset.role);
check("exactly two projects carry a role", marked.length === 2, String(marked.length));
const roles = marked.map((r) => r.dataset.role).sort();
check("one is the head start, one is held out",
  roles.join(",") === "head start,held out", roles.join(","));

section("per-project detail");
check("every project has a detail card",
  card.querySelectorAll(".arch-card").length === 19,
  String(card.querySelectorAll(".arch-card").length));
check("detail cards carry a distribution strip",
  card.querySelectorAll(".arch-card").every((c) => c.querySelectorAll(".dist-bar").length === 1));

section("the awkward corners");
check("four corners are listed", card.querySelectorAll(".corner").length === 4,
  String(card.querySelectorAll(".corner").length));
const text = deepText(card);
check("it explains why empty descriptions are kept, not dropped",
  text.includes("would throw away hundreds") || text.includes("kept"));
check("it flags that the no-priority project is the warm-start one",
  text.includes("head start"));

section("getting ahead of a bad explanation");
check("it shows the overall class mix", text.includes("All nineteen projects together"));
check("it states the imbalance ratio", text.includes("imbalance_ratio"));
check("it says imbalance is not the explanation",
  text.includes("imbalance is not") || text.includes("not the explanation"));

section("facts referenced all exist");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

section("handover");
card.querySelector(".handover-next").fire("click");
check("it hands on to the cleaning bench", nexted === 1);

done();

