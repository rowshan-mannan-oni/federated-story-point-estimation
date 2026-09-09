import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const front = await import("../js/stations/front-door.js");

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "front-door");
check("front-door is marked built", entry.built === true);
check("it is the first stop", stops.indexOf("front-door") === 0);

section("the front door mounts");
const host = document.createElement("div");
let nexted = 0;
let wentTo = null;
let threw = null;
try {
  front.mount(host, {
    stop: entry,
    index: 0,
    next: () => { nexted++; },
    goTo: (i) => { wentTo = i; },
  });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();

const card = host.children[0];
check("it produces one card", host.children.length === 1 && card.className.includes("card"));
check("the headline is the thesis title",
  card.querySelector("h1").textContent.includes("Guessing effort"));

section("what it says");
check("six sentences, exactly six",
  card.querySelectorAll(".six-list")[0].children.length === 6,
  String(card.querySelectorAll(".six-list")[0].children.length));
check("three research questions, each in a drawer",
  card.querySelectorAll(".rq-exact").length === 3,
  String(card.querySelectorAll(".rq-exact").length));
check("three stat plates", card.querySelectorAll(".part-stat").length === 3);
check("plate numbers carry their provenance",
  card.querySelectorAll(".stat-value").every((v) => v.children.some((c) => c.className === "fact")));
check("the plates carry real values",
  card.querySelectorAll(".stat-value").some((v) => v.textContent.includes(data.text("corpus.projects"))),
  card.querySelectorAll(".stat-value").map((v) => v.textContent).join(" / "));
check("the road ahead lists all seven parts",
  card.querySelectorAll(".path-row").length === 7,
  String(card.querySelectorAll(".path-row").length));

section("it hands the reader onward");
const handoverBtn = card.querySelector(".handover-next");
check("there is a next control", Boolean(handoverBtn));
handoverBtn.fire("click");
check("clicking it advances the walk-through", nexted === 1, String(nexted));
card.querySelector(".path-row").fire("click");
check("a road-ahead row jumps to that part", wentTo === 0, String(wentTo));

section("honesty");
const text = deepText(card);
check("it warns that the task is hard for everyone", text.includes("barely beat trivial"));
check("it says unflattering numbers stay unflattering", text.includes("shown unflattering"));
check("it names all three questions", ["RQ1", "RQ2", "RQ3"].every((q) => text.includes(q)));

section("facts resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

done();
