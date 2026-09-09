import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const patches = await import("../js/stations/patches.js");

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "patches");
check("patches is marked built", entry.built === true);
check("it is stop 10", stops.indexOf("patches") === 9);

section("the arithmetic behind it closes");
const rank = data.value("lora.rank");
const sites = data.value("lora.sites");
const layers = data.value("lora.layers");
const loraB = data.value("params.lora_b");
check("the layer count was derived, not assumed",
  data.fact("lora.layers").kind === "derived", data.fact("lora.layers").kind);
check("sites = layers x 2 attention spots", sites === layers * 2, `${sites} vs ${layers} x 2`);
check("sites x width x rank reproduces the recorded patch total",
  sites * 768 * rank === loraB, `${sites * 768 * rank} vs ${loraB}`);
check("frozen + trainable = the whole model",
  data.value("params.frozen") + data.value("params.trainable") === data.value("params.total"),
  `${data.value("params.frozen") + data.value("params.trainable")} vs ${data.value("params.total")}`);
check("the two strips really are a small share of the square",
  Math.abs(data.value("lora.patch_share") - 100 * (2 * 768 * rank) / (768 * 768)) < 0.01,
  String(data.value("lora.patch_share")));
check("the share that moves is under one percent",
  data.value("params.share_pct") < 1, String(data.value("params.share_pct")));

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await patches.mount(host, { stop: entry, index: 9, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "Patches, not models");

section("the square against the strips");
const square = card.querySelector(".rank-square");
const strips = card.querySelectorAll(".rank-strip");
check("the full table is drawn", Boolean(square));
check("two strips are drawn", strips.length === 2, String(strips.length));
check("both describe themselves in words",
  Boolean(square.getAttribute("aria-label")) &&
  Boolean(card.querySelector(".rank-strips").getAttribute("aria-label")));
check("the strips are drawn at their true relative width",
  Math.abs(Number(strips[0].style.getPropertyValue("--share")) - rank / 768) < 1e-9,
  strips[0].style.getPropertyValue("--share"));

section("the rank dial");
const readout = card.querySelector(".readout-value");
const verdict = card.querySelector(".rank-verdict");
check("at the run's own rank it reproduces the recorded figure",
  readout.textContent.replace(/[^0-9]/g, "") === String(loraB),
  `${readout.textContent} vs ${loraB}`);
check("and it says so", verdict.dataset.match === "true", verdict.textContent);

const steppers = card.querySelectorAll(".stepper-btn");
const before = readout.textContent;
steppers[1].fire("click");                    // widen the strips
const after = card.querySelector(".readout-value").textContent;
check("widening the strips costs more numbers",
  Number(after.replace(/[^0-9]/g, "")) > Number(before.replace(/[^0-9]/g, "")),
  `${before} -> ${after}`);
check("the strips redraw wider",
  Number(card.querySelectorAll(".rank-strip")[0].style.getPropertyValue("--share")) > rank / 768);
check("the verdict stops claiming it is the run's setting",
  card.querySelector(".rank-verdict").dataset.match !== "true");
steppers[0].fire("click");                    // back to the run's rank
check("returning to the run's rank restores the exact figure",
  card.querySelector(".readout-value").textContent === before,
  card.querySelector(".readout-value").textContent);

section("where the patches attach");
const rows = card.querySelectorAll(".layer-row");
check("one row per layer", rows.length === layers, String(rows.length));
check("two patch sites per layer",
  rows.every((r) => r.querySelectorAll(".layer-slot").length === 2));
check("the slots are named query and value",
  rows[0].querySelectorAll(".layer-slot").map((s) => s.textContent).join(",") === "query,value",
  rows[0].querySelectorAll(".layer-slot").map((s) => s.textContent).join(","));
check("the stack describes itself for a screen reader",
  (card.querySelector(".layer-stack").getAttribute("aria-label") || "").includes("layers"));

section("what it buys");
const plates = card.querySelectorAll(".stat-value").map((v) => v.textContent);
check("the upload size is shown in megabytes",
  plates.some((v) => v.includes("MB")), plates.join(" / "));
check("the reduction factor is shown",
  plates.some((v) => v.includes(data.text("comms.reduction"))), plates.join(" / "));
check("plate numbers keep their provenance",
  card.querySelectorAll(".stat-value").every((v) => v.children.some((c) => c.className === "fact")));

section("honesty and handover");
const text = deepText(card);
check("it admits the thin-patch assumption is empirical, not proved",
  text.includes("empirical claim rather than a proof"));
check("it says the rank was not tuned here",
  text.includes("rather than something this thesis tuned"));
check("it defers the quality half of the comparison",
  text.includes("Cheap and useless is not a result"));
check("it sets up the frozen-half question for the next stop",
  text.includes("frozen at its random starting values"));

section("facts resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();
