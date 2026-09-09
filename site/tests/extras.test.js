import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const extrasStop = await import("../js/stations/extras.js");
const split = await data.split();
const extras = split.extras;

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "extras");
check("extras is marked built", entry.built === true);
check("it is stop 9", stops.indexOf("extras") === 8);

section("the measurement behind it");
check("type figures were computed", Array.isArray(extras?.types) && extras.types.length > 0);
check("priority figures were computed", Array.isArray(extras?.priorities));
check("the type buckets account for every training issue",
  extras.types.reduce((s, t) => s + t.n, 0) === data.value("split.train"),
  `${extras.types.reduce((s, t) => s + t.n, 0)} vs ${data.value("split.train")}`);
check("the priority buckets do too",
  extras.priorities.reduce((s, t) => s + t.n, 0) === data.value("split.train"));
/* The lookup table always reserves a row for a label never seen in training,
   so it is one bigger than the data unless the data already contains one. */
check("the type table reserves one row beyond the types seen",
  data.value("split.type_vocab") === extras.types.length + 1,
  `${data.value("split.type_vocab")} rows vs ${extras.types.length} types seen`);
check("priority needs no reservation, since Unknown is already in the data",
  data.value("split.priority_vocab") === extras.priorities.length &&
  extras.priorities.some((p) => p.value === "unknown"),
  `${data.value("split.priority_vocab")} rows vs ${extras.priorities.length} values`);
check("the counter's arithmetic uses the table sizes, not the seen counts",
  (data.value("split.type_vocab") + data.value("split.priority_vocab")) * 16
    === data.value("params.embeddings"),
  `${(data.value("split.type_vocab") + data.value("split.priority_vocab")) * 16} vs ${data.value("params.embeddings")}`);

section("the two findings are real, not asserted");
const big = [...extras.types].sort((a, b) => b.n - a.n).slice(0, 4);
const spreadOfBigFour = Math.max(...big.map((t) => t.mean_sp)) - Math.min(...big.map((t) => t.mean_sp));
check("the four biggest types really are near-identical",
  spreadOfBigFour < 0.6, `spread ${spreadOfBigFour.toFixed(2)}`);
check("but the extremes really are far apart",
  data.value("extras.type_spread") > 2, String(data.value("extras.type_spread")));
const byName = Object.fromEntries(extras.priorities.map((p) => [p.value, p.mean_sp]));
check("priority really is non-monotone (high sits below medium)",
  byName.high < byName.medium, `high ${byName.high} vs medium ${byName.medium}`);
check("and 'high' really is the biggest bucket",
  extras.priorities.reduce((a, b) => (a.n > b.n ? a : b)).value === "high");

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await extrasStop.mount(host, { stop: entry, index: 8, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "Type and priority");

section("the priority list shows the break in the pattern");
const rows = card.querySelectorAll(".prio-row");
check("every priority is listed", rows.length === extras.priorities.length, String(rows.length));
check("the order shown is the claimed order, not sorted by size",
  rows[0].querySelector(".prio-name").textContent === "highest",
  rows[0].querySelector(".prio-name").textContent);
const marked = rows.filter((r) => r.dataset.break === "true");
check("the out-of-order row is marked", marked.length >= 1, String(marked.length));
check("and it is the one that breaks the scale",
  marked.some((r) => r.querySelector(".prio-name").textContent === "medium"),
  marked.map((r) => r.querySelector(".prio-name").textContent).join(", "));

section("the live parameter counter");
const readouts = card.querySelectorAll(".readout-value").map((r) => r.textContent);
const real = data.value("params.trainable");
check("at the run's own settings it reproduces the real figure exactly",
  readouts.some((v) => v.replace(/[^0-9]/g, "") === String(real)),
  `${readouts.join(" / ")} vs ${real}`);
const verdict = card.querySelector(".count-verdict");
check("and it says so", verdict.dataset.match === "true", verdict.textContent);
check("the breakdown has three parts",
  card.querySelectorAll(".count-row").length === 3,
  String(card.querySelectorAll(".count-row").length));
const parts = card.querySelectorAll(".count-n").map((n) => Number(n.textContent.replace(/,/g, "")));
check("the parts add up to the total",
  parts.reduce((a, b) => a + b, 0) === real,
  `${parts.reduce((a, b) => a + b, 0)} vs ${real}`);

/* moving a dial must change the total, and moving it back must restore it */
const steppers = card.querySelectorAll(".stepper-btn");
const before = card.querySelector(".readout-value").textContent;
steppers[1].fire("click");           // increase "numbers per label"
const after = card.querySelector(".readout-value").textContent;
check("turning a dial changes the upload size", after !== before, `${before} -> ${after}`);
check("and the verdict stops claiming it is the run's setting",
  card.querySelector(".count-verdict").dataset.match !== "true");
steppers[0].fire("click");           // back down
check("turning it back restores the exact figure",
  card.querySelector(".readout-value").textContent === before,
  card.querySelector(".readout-value").textContent);

section("honesty");
const text = deepText(card);
check("it admits the big four types say almost nothing",
  text.includes("almost nothing"));
check("it explains why 'high' behaves oddly", text.includes("shrug"));
check("it admits the with/without-extras experiment was never run",
  text.includes("has not been run"));
check("it names shared-vs-local embeddings as unrun",
  text.includes("optional micro-ablation") && text.includes("not run it"));
check("it flags that two kinds of missing share one bucket",
  text.includes("kinds of missing") && text.includes("one bucket"));
check("it explains the reserved row for unseen labels",
  text.includes("reserved") && text.includes("never seen"));

section("facts resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();

