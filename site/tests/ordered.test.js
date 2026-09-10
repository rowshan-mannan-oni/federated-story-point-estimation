import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const station = await import("../js/stations/ordered.js");

const POINTS = [1, 2, 3, 5, 8];

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "ordered");
check("ordered is marked built", entry.built === true);
check("it is stop 12", stops.indexOf("ordered") === 11);

section("the head arithmetic");
check("the ordered head has one output per threshold",
  data.value("head.outputs") === POINTS.length - 1,
  `${data.value("head.outputs")} vs ${POINTS.length - 1}`);
check("the unordered head would have one per story point",
  data.value("head.outputs_other") === POINTS.length,
  String(data.value("head.outputs_other")));
check("the ordered head is the smaller of the two",
  data.value("params.head") < data.value("head.params_other"),
  `${data.value("params.head")} vs ${data.value("head.params_other")}`);
check("and the difference is what the facts say",
  data.value("head.params_other") - data.value("params.head") === data.value("head.params_delta"),
  String(data.value("head.params_delta")));

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await station.mount(host, { stop: entry, index: 11, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "The answers are ordered");

section("the cost grid");
const cells = () => card.querySelectorAll(".cost-cell");
check("a five by five grid is drawn", cells().length === 25, String(cells().length));
check("the diagonal is marked as correct",
  cells().filter((c) => c.dataset.right === "true").length === 5,
  String(cells().filter((c) => c.dataset.right === "true").length));

/* The ordered lens must charge by distance along the deck. */
const values = () => cells().map((c) => Number(c.textContent));
const ordered = values();
check("under the ordered head, a correct answer costs nothing",
  ordered[0] === 0 && ordered[6] === 0 && ordered[24] === 0);
check("one step out costs one",
  ordered[1] === 1, String(ordered[1]));
check("four steps out costs four",
  ordered[4] === 4, String(ordered[4]));
check("the cost is symmetric",
  ordered[4] === ordered[20], `${ordered[4]} vs ${ordered[20]}`);

const segs = card.querySelectorAll(".segment");
segs[1].fire("click");                       // unordered head
const flat = values();
check("under the unordered head every wrong answer costs the same",
  new Set(flat.filter((v) => v !== 0)).size === 1,
  [...new Set(flat)].join(", "));
check("and 8-for-1 costs no more than 2-for-1",
  flat[4] === flat[1], `${flat[4]} vs ${flat[1]}`);

segs[2].fire("click");                       // in story points
const points = values();
check("in story points the deck's uneven gaps show",
  points[4] === Math.abs(POINTS[4] - POINTS[0]) && points[3] === Math.abs(POINTS[3] - POINTS[0]),
  `${points[4]} / ${points[3]}`);
check("the 5-to-8 jump really is bigger than 3-to-5",
  Math.abs(POINTS[4] - POINTS[3]) > Math.abs(POINTS[3] - POINTS[2]));
segs[0].fire("click");

section("the threshold organ");
const switches = card.querySelectorAll(".switch-track");
check("four questions are asked", switches.length === 4, String(switches.length));
const answer = () => card.querySelector(".readout-value").textContent;
const trace = () => card.querySelector(".organ-trace");
check("no answers yet decodes to the smallest value", answer() === "1", answer());

switches[0].fire("click");
check("one yes decodes to 2", answer() === "2", answer());
switches[1].fire("click");
switches[2].fire("click");
check("three yeses decode to 5", answer() === "5", answer());
switches[3].fire("click");
check("all four decode to 8", answer() === "8", answer());
check("a consistent set is not flagged as odd", trace().dataset.odd === "false");

/* The property worth demonstrating: contradictory answers stay decodable. */
switches[0].fire("click");                   // no, yes, yes, yes
check("a self-contradictory set is flagged", trace().dataset.odd === "true", trace().dataset.odd);
check("but it still decodes to a real story point",
  POINTS.map(String).includes(answer()), answer());
check("and the decoding stops at the first no", answer() === "1", answer());
check("the trace explains what happened",
  trace().textContent.includes("stops at the first no"));

section("why distance costs more");
const rows = card.querySelectorAll(".grade-row");
check("all five guesses are graded", rows.length === 5, String(rows.length));
check("each is graded against four questions",
  rows.every((r) => r.querySelectorAll(".grade-mark").length === 4));
const wrongCounts = rows.map((r) =>
  r.querySelectorAll(".grade-mark").filter((m) => m.dataset.ok === "false").length);
check("when the truth is 3, the wrong-counts are the distances",
  wrongCounts.join(",") === "2,1,0,1,2", wrongCounts.join(","));
check("the correct guess is marked", rows[2].dataset.right === "true");

section("honesty");
const text = deepText(card);
check("it says the comparison has not been run",
  text.includes("has not been run"));
check("it predicts the asymmetry to look for",
  text.includes("barely move macro-F1"));
check("it says that asymmetry is itself the finding",
  text.includes("asymmetry is the finding"));
check("it notes class weights apply only to the unordered head",
  text.includes("class weights"));
check("it points at the near-miss evidence already in the run",
  text.includes("confusion.within_one"));

section("facts resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();
