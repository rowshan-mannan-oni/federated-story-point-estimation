import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const station = await import("../js/stations/merging.js");
const { mergeExperiment } = await import("../js/sim/lora.js");
const merging = await data.merging();

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "merging");
check("merging is marked built", entry.built === true);
check("it is stop 11", stops.indexOf("merging") === 10);

section("the maths, in the browser");
/* This is the claim the whole method rests on, so it is checked directly
   rather than only through the page that presents it. */
const naive = mergeExperiment({ spread: 0.02, frozen: false });
const frozen = mergeExperiment({ spread: 0.02, frozen: true });
const identical = mergeExperiment({ spread: 0, frozen: false });

check("merging separately is badly wrong when projects differ",
  naive.error > 0.2, `error ${naive.error.toFixed(4)}`);
check("freezing one half makes it exact, not merely small",
  frozen.error < 1e-12, `error ${frozen.error.toExponential(2)}`);
check("with no drift at all there is no error either",
  identical.error < 1e-12, `error ${identical.error.toExponential(2)}`);
check("the closed form matches the measured gap (drifting case)",
  naive.covResidual < 1e-12, `residual ${naive.covResidual.toExponential(2)}`);
check("and in the frozen case", frozen.covResidual < 1e-12,
  `residual ${frozen.covResidual.toExponential(2)}`);

/* More disagreement must mean more error — that is the formula's shape. */
const curve = [0, 0.01, 0.02, 0.04].map((s) => mergeExperiment({ spread: s }).error);
check("error grows as the projects drift further apart",
  curve.every((v, i) => i === 0 || v > curve[i - 1]),
  curve.map((v) => v.toFixed(3)).join(" -> "));
check("freezing beats every drift setting",
  curve.slice(1).every((v) => v > frozen.error));

section("the same experiment at full size");
check("it was run at the real width", merging.width === 768, String(merging.width));
check("with the run's own rank", merging.rank === data.value("lora.rank"),
  `${merging.rank} vs ${data.value("lora.rank")}`);
check("both halves trained is badly wrong there too",
  merging.error_independent > 0.5, String(merging.error_independent));
check("frozen is exact there too",
  merging.error_frozen < 1e-12, merging.error_frozen.toExponential(2));
check("the identity holds at full size",
  merging.identity_residual < 1e-12, merging.identity_residual.toExponential(2));
check("the full-size curve rises with drift",
  merging.curve.every((p, i) => i === 0 || p.error > merging.curve[i - 1].error),
  merging.curve.map((p) => p.error).join(" -> "));

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await station.mount(host, { stop: entry, index: 10, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "The merging problem");

section("the live rig");
const grids = card.querySelectorAll(".merge-cells");
check("three matrices are drawn", grids.length === 3, String(grids.length));
check("each is 12 by 12 cells",
  grids.every((g) => g.children.length === 144),
  grids.map((g) => g.children.length).join(", "));
check("each describes itself for a screen reader",
  grids.every((g) => (g.getAttribute("aria-label") || "").includes("sample of the matrix")));

const verdict = card.querySelector(".merge-verdict");
check("it starts with the error present", verdict.dataset.state === "bad", verdict.dataset.state);
check("and names the size of it", /\d+\.\d% wrong/.test(verdict.textContent), verdict.textContent);

const freezeSwitch = card.querySelector(".switch-track");
freezeSwitch.fire("click");
const after = card.querySelector(".merge-verdict");
check("throwing the freeze switch makes it exact",
  after.dataset.state === "exact", after.dataset.state);
check("and it says the error is rounding, not smallness",
  after.textContent.includes("it is rounding"), after.textContent);
freezeSwitch.fire("click");
check("switching back restores the error",
  card.querySelector(".merge-verdict").dataset.state === "bad");

section("the identity is stated and checked");
const text = deepText(card);
check("the formula is on the page", text.includes("(Bᵢ − B̄)(Aᵢ − Ā)"));
check("it is explained in words, not just symbols",
  text.includes("how far each project's tall strip drifted"));
check("it says the error is identically zero, not merely small",
  text.includes("identically nothing"));
check("the residual readout is present",
  text.includes("closed form vs measured gap"));

section("honesty");
check("it admits exactness is not evidence of better estimates",
  text.includes("It does <strong>not</strong> show") || text.includes("does <strong>not</strong> show"));
check("it names the unrun experiment",
  text.includes("has not been run"));
check("it explains why the demo is smaller than the model",
  text.includes("64-wide") || text.includes("running at 64"));
check("it admits freezing costs expressiveness",
  text.includes("no longer learning"));

section("facts resolve");
const keys = [...text.matchAll(/data-fact="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => data.fact(k).kind === "missing");
check(`${keys.length} fact references, all resolvable`, missing.length === 0, missing.join(", "));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();
