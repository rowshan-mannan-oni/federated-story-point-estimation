import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const poker = await import("../js/stations/poker-table.js");
const stops = await import("../js/core/stops.js");

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "poker-table");
check("poker-table is marked built", entry.built === true);
check("it is stop 2", stops.indexOf("poker-table") === 1);

section("it mounts");
const host = document.createElement("div");
let nexted = 0;
let threw = null;
try {
  await poker.mount(host, {
    stop: entry, index: 1,
    next: () => { nexted++; }, goTo: () => {},
  });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();

const card = host.children[0];
check("one card is produced", host.children.length === 1);
check("the headline is the poker table", card.querySelector("h1").textContent === "The poker table");

section("the deck");
const cards = card.querySelectorAll(".poker-card");
check("five cards, one per story point", cards.length === 5, String(cards.length));
check("they are 1, 2, 3, 5, 8",
  cards.map((c) => c.dataset.value).join(",") === "1,2,3,5,8",
  cards.map((c) => c.dataset.value).join(","));
check("each card names itself for a screen reader",
  cards.every((c) => (c.getAttribute("aria-label") || "").startsWith("Estimate ")));

section("dealing and answering");
const issueTitle = card.querySelector(".poker-title");
const reveal = card.querySelector(".poker-reveal");
const nextBtn = card.querySelector(".poker-next");
const tally = card.querySelector(".poker-tally");

check("an issue is dealt on arrival", issueTitle.textContent.length > 10, issueTitle.textContent);
check("the answer stays hidden until you commit", reveal.hidden === true);
check("'deal another' is disabled before you answer", nextBtn.disabled === true);
check("the project name is not shown before the guess",
  !card.querySelector(".poker-answer"));

const firstTitle = issueTitle.textContent;
cards[0].fire("click");                      // guess "1"
check("answering reveals the result", reveal.hidden === false);
check("the real answer is stated", card.querySelector(".poker-answer").textContent.includes("recorded this as a"));
check("the team's card is marked", cards.some((c) => c.dataset.state === "truth"));
check("every card locks after answering", cards.every((c) => c.disabled === true));
check("the tally counts one judgement", tally.textContent.startsWith("1 judged"), tally.textContent);
check("'deal another' is now available", nextBtn.disabled === false);

const before = tally.textContent;
cards[3].fire("click");
check("a second guess on the same issue is ignored", tally.textContent === before);

nextBtn.fire("click");
check("dealing again shows a different issue", issueTitle.textContent !== firstTitle);
check("and hides the answer again", reveal.hidden === true);
check("and unlocks the deck", cards.every((c) => c.disabled === false));

section("the calibration evidence");
const strips = card.querySelectorAll(".dist-bar");
check("distribution strips are drawn", strips.length >= 4, String(strips.length));
check("every strip has five segments",
  strips.every((s) => s.children.filter((c) => c.className === "dist-seg").length === 5));
check("each strip describes itself in words",
  strips.every((s) => (s.getAttribute("aria-label") || "").includes("% are")));
check("all 18 projects are available in the drawer", strips.length >= 18 + 4, String(strips.length));

const plates = card.querySelectorAll(".stat-value").map((v) => v.textContent);
check("the extremes are shown as real figures",
  plates.includes(data.text("calibration.lowest_mean")) &&
  plates.includes(data.text("calibration.highest_mean")),
  plates.join(" / "));
check("project names are printed readably, not with underscores",
  !card.querySelectorAll(".stat-caption").some((c) => c.textContent.includes("_")));

section("honesty and handover");
const text = deepText(card);
check("it admits the contest is not fair", text.includes("Is this a fair contest") || text.includes("fair contest"));
check("it says the examples come from the training split", text.includes("training split"));
check("it warns that pooling makes things worse", text.includes("worse"));
card.querySelector(".handover-next").fire("click");
check("the handover advances the walk-through", nexted === 1);

done();

