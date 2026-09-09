import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const pool = await import("../js/stations/why-not-pool.js");
const gloss = await import("../js/ui/glossary.js");
await gloss.init();

section("the stop is on the path");
const entry = stops.STOPS.find((s) => s.id === "why-not-pool");
check("why-not-pool is marked built", entry.built === true);
check("it is stop 3", stops.indexOf("why-not-pool") === 2);

section("it mounts");
const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await pool.mount(host, { stop: entry, index: 2, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];
check("the headline is right", card.querySelector("h1").textContent === "Why not just pool it");

section("the switch");
const segs = card.querySelectorAll(".segment");
check("two arrangements to compare", segs.length === 2, String(segs.length));
check("it starts on the pooled side", card.querySelector(".leaves-panel").dataset.mode === "pool");
check("the pooled side shows real issue text", card.querySelectorAll(".leaves-issue").length === 3);
check("real titles, not placeholders",
  card.querySelectorAll(".leaves-title").every((t) => t.textContent.length > 10));

segs[1].fire("click");
const panel = card.querySelector(".leaves-panel");
check("switching moves to the federated side", panel.dataset.mode === "fed", panel.dataset.mode);
check("no issue text remains on the federated side", card.querySelectorAll(".leaves-issue").length === 0);
const rows = card.querySelectorAll(".payload-row");
check("the payload is broken into its three parts", rows.length === 3, String(rows.length));

const counts = card.querySelectorAll(".payload-count").map((c) => Number(c.textContent.replace(/,/g, "")));
check("the parts are the real trained tensors",
  counts.includes(data.value("params.lora_b")) &&
  counts.includes(data.value("params.embeddings")) &&
  counts.includes(data.value("params.head")),
  counts.join(" + "));
check("and they add up to what is actually sent",
  counts.reduce((a, b) => a + b, 0) === data.value("params.trainable"),
  `${counts.reduce((a, b) => a + b, 0)} vs ${data.value("params.trainable")}`);

segs[0].fire("click");
check("switching back restores the pooled side",
  card.querySelector(".leaves-panel").dataset.mode === "pool");

section("the awkward arithmetic");
const text = deepText(card);
const ratio = Math.round(data.value("comms.total_bytes") / data.value("corpus.text_bytes"));
check("it states the real ratio, not a flattering one",
  text.includes(`${ratio} times more traffic`), `expected ${ratio}x`);
check("the ratio is genuinely unflattering (federation moves more)", ratio > 1, String(ratio));
check("it says federation is not a bandwidth saving",
  text.includes("does not save traffic"));
check("it points the reader at the comparison the thesis DOES make",
  text.includes("whole model every round"));

section("honesty about the claim");
check("it admits the corpus is public", text.includes("public"));
check("it names gradient inversion", text.includes("gradient inversion"));
check("it names the warm start as the one pooled place", text.includes("warmed up on a single large project"));
check("it says the server is trusted", text.includes("server is trusted") || text.includes("The server is trusted"));
check("it refuses the stronger claim",
  text.includes("not a proof that your text is") || text.includes("would be easier and would be wrong"));
check("four caveats are listed", card.querySelectorAll(".caveat").length === 4,
  String(card.querySelectorAll(".caveat").length));

section("glossary terms resolve");
/* The shim stores innerHTML without parsing it, so fillTerms cannot see the
   markup here the way a browser would. Assert the thing that actually matters:
   every term this stop marks up has a definition behind it. */
const marked = [...text.matchAll(/data-term="([^"]+)"/g)].map((m) => m[1]);
const known = gloss.allTerms();
check("the stop marks up privacy terms", marked.length >= 3, marked.join(", "));
check("every marked term has a definition",
  marked.every((t) => t.toLowerCase() in known),
  marked.filter((t) => !(t.toLowerCase() in known)).join(", ") || "all found");

section("handover");
card.querySelector(".handover-next").fire("click");
check("it hands on to the archive", nexted === 1);

done();

