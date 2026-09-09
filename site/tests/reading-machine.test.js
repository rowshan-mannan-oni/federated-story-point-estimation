import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const reading = await import("../js/stations/reading-machine.js");
const split = await data.split();
const lengths = split.lengths;

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "reading-machine");
check("reading-machine is marked built", entry.built === true);
check("it is stop 8", stops.indexOf("reading-machine") === 7);

section("the measurement behind it");
check("length buckets were computed", Array.isArray(lengths?.buckets) && lengths.buckets.length > 0);
check("the buckets account for every training issue",
  lengths.buckets.reduce((s, b) => s + b.n, 0) === lengths.total,
  `${lengths.buckets.reduce((s, b) => s + b.n, 0)} vs ${lengths.total}`);
check("the bucket total matches the training pile",
  lengths.total === data.value("split.train"),
  `${lengths.total} vs ${data.value("split.train")}`);
check("long issues really do average higher than short ones",
  data.value("length.mean_sp_long") > data.value("length.mean_sp_short"),
  `${data.value("length.mean_sp_long")} vs ${data.value("length.mean_sp_short")}`);
check("but the correlation is genuinely weak",
  Math.abs(data.value("length.correlation")) < 0.2,
  String(data.value("length.correlation")));

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await reading.mount(host, { stop: entry, index: 7, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "The reading machine");

section("the chain");
const steps = card.querySelectorAll(".chain-step");
check("five steps from text to one row of numbers", steps.length === 5, String(steps.length));
const tokens = card.querySelectorAll(".token");
check("the split is illustrated with pieces", tokens.length > 5, String(tokens.length));
check("markers are shown as special pieces",
  tokens.some((t) => t.dataset.special === "true") ||
  !tokens.some((t) => /^\[[A-Z_]+\]$/.test(t.textContent)),
  "a marker token was not highlighted");

section("the illustration is stamped, not passed off as real");
const stampsInChain = card.querySelectorAll(".stamp");
check("a SIMULATED stamp appears",
  stampsInChain.some((s) => s.textContent === "SIMULATED"),
  stampsInChain.map((s) => s.textContent).join(", "));
const text = deepText(card);
check("it says the real tokeniser is not shipped",
  text.includes("does not carry") || text.includes("does not ship"));

section("lengths");
const bars = card.querySelectorAll(".hist-col");
check("a histogram of lengths is drawn",
  bars.length === lengths.buckets.filter((b) => b.n > 0).length,
  `${bars.length} bars`);
check("bars past the limit are marked",
  bars.some((b) => b.dataset.mark === "true"));
check("each bar states its own count in text",
  bars.every((b) => b.querySelector(".hist-value").textContent.length > 0));
check("each bar carries the average story point of its issues",
  bars.every((b) => b.querySelector(".hist-note")));

section("the words-not-tokens caveat");
check("it warns the figures are words, not tokens",
  text.includes("words, not tokens") || text.includes("these are words"));
check("it says the true truncation share is higher",
  text.includes("lower bound"));

section("does truncation matter");
check("it states both averages",
  text.includes("length.mean_sp_long") && text.includes("length.mean_sp_short"));
check("it gives the correlation rather than hiding it",
  text.includes("length.correlation"));
check("it calls the effect real but small",
  text.includes("real, and small") || text.includes("small enough that it explains very little"));
check("it names the unrun 512 experiment as unrun",
  text.includes("has not yet run") || text.includes("optional and has not"));
check("the short-vs-long distributions are shown",
  card.querySelectorAll(".dist-bar").length >= 2,
  String(card.querySelectorAll(".dist-bar").length));

section("why CodeBERT, and the frozen hint");
check("it explains the code-flavoured English argument",
  text.includes("NullPointerException"));
check("the run's real settings are listed",
  text.includes(data.text("run.model")), data.text("run.model"));
check("it plants the frozen-encoder idea for later",
  text.includes("frozen") && text.includes("125 million"));

section("facts resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();

