import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const splitting = await import("../js/stations/splitting.js");
const { overlaps } = await import("../js/viz/spans.js");
const split = await data.split();

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "splitting");
check("splitting is marked built", entry.built === true);
check("it is stop 7", stops.indexOf("splitting") === 6);

section("the data behind it");
check("both ways of cutting were computed",
  Boolean(split.modes?.random && split.modes?.temporal));
check("the piles add up to the corpus pool",
  split.modes.random.totals.train + split.modes.random.totals.val + split.modes.random.totals.test ===
  split.modes.temporal.totals.train + split.modes.temporal.totals.val + split.modes.temporal.totals.test,
  "the two cuts divide different totals");
check("the run's own split is the one shown as used",
  split.mode_used === data.value("split.mode_used"), split.mode_used);

section("the leakage the random cut allows is real, not claimed");
let leaky = 0, clean = 0;
for (const p of split.modes.random.projects) if (overlaps(p.train, p.test)) leaky++;
for (const p of split.modes.temporal.projects) if (overlaps(p.train, p.test)) clean++;
check(`all ${leaky} projects overlap train/test under a random cut`,
  leaky === split.modes.random.projects.length, String(leaky));
check("no project overlaps under a date cut", clean === 0, String(clean));

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await splitting.mount(host, { stop: entry, index: 6, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "Cutting the data three ways");

section("three piles");
const plates = card.querySelectorAll(".stat-value").map((v) => v.textContent);
check("the three pile sizes are shown",
  plates.some((v) => v.includes(data.text("split.train"))) &&
  plates.some((v) => v.includes(data.text("split.val"))) &&
  plates.some((v) => v.includes(data.text("split.test"))),
  plates.join(" / "));
check("plate numbers keep their provenance",
  card.querySelectorAll(".stat-value").every((v) => v.children.some((c) => c.className === "fact")));

const tables = card.querySelectorAll("table");
const perProject = tables[0].querySelectorAll("tr").filter((r) => r.children.some((c) => c.tagName === "TD"));
check("every project's division is listed",
  perProject.length === split.per_project.length,
  `${perProject.length} vs ${split.per_project.length}`);

section("the switch between cuts");
const bars = () => card.querySelectorAll(".span-bar");
check("three ranges are drawn", bars().length === 3, String(bars().length));
const verdict = card.querySelector(".split-verdict");
check("it starts on the random cut and calls out the overlap",
  verdict.dataset.leak === "true", verdict.dataset.leak);
check("the wording names the actual problem",
  verdict.textContent.includes("written after the ones it is judged on"),
  verdict.textContent);

const segs = card.querySelectorAll(".segment");
segs[1].fire("click");   // switch to "By date"
check("switching to the date cut removes the overlap",
  card.querySelector(".split-verdict").dataset.leak === "false",
  card.querySelector(".split-verdict").dataset.leak);
check("and says so plainly",
  card.querySelector(".split-verdict").textContent.includes("before everything it is judged on"));

const picker = card.querySelector("select");
check("a project picker is offered", Boolean(picker));
check("it lists every project", picker.children.length === split.modes.random.projects.length,
  String(picker.children.length));
picker.value = split.modes.random.projects[3].name;
picker.fire("change");
check("changing project redraws the ranges", bars().length === 3);

section("honesty");
const text = deepText(card);
check("it admits the finished run used the random cut",
  text.includes(`used the ${split.mode_used}`), split.mode_used);
check("it carries a preliminary stamp about that",
  card.querySelectorAll(".stamp-banner").length >= 1);
check("it explains team drift over time", text.includes("called a 5 in 2014"));
check("it states the opened-once rule", text.includes("opened exactly once") || text.includes("used exactly once"));
check("it explains why peeking ruins the number",
  text.includes("no longer an estimate"));
check("it points forward to the fair-judging stop", text.includes("judged by that same rule"));

section("facts resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();

