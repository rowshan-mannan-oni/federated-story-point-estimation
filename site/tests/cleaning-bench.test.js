import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";
import { readFileSync } from "node:fs";

const cleaningData = JSON.parse(readFileSync(new URL("../data/cleaning.json", import.meta.url), "utf8"));
/* The vectors live in their own file so the page never downloads them. */
cleaningData.vectors = JSON.parse(readFileSync(new URL("../data/cleaning-vectors.json", import.meta.url), "utf8")).vectors;
const { clean, cleanStages, countSubstitutions, decodeExportQuoting, unescapeEntities } =
  await import("../js/sim/clean.js");

section("the browser's cleaner matches the Python, exactly");
let exact = 0;
const bad = [];
for (const v of cleaningData.vectors) {
  if (clean(v.raw) === v.clean) exact++;
  else if (bad.length < 3) bad.push(v);
}
check(`${exact} / ${cleaningData.vectors.length} real corpus fields clean identically`,
  exact === cleaningData.vectors.length);
for (const b of bad) {
  console.log("    RAW :", JSON.stringify(b.raw.slice(0, 120)));
  console.log("    WANT:", JSON.stringify(b.clean.slice(0, 120)));
  console.log("    GOT :", JSON.stringify(clean(b.raw).slice(0, 120)));
}

section("stage by stage, against the Python's own trace");
let stageMismatch = 0;
for (const show of cleaningData.showcase) {
  const mine = cleanStages(show.raw);
  check(`showcase from ${show.project}: same number of stages`,
    mine.length === show.stages.length, `${mine.length} vs ${show.stages.length}`);
  show.stages.forEach((theirs, i) => {
    if (mine[i].text !== theirs.text) {
      stageMismatch++;
      if (stageMismatch <= 2) {
        console.log(`    stage "${theirs.id}" differs`);
        console.log("      python:", JSON.stringify(theirs.text.slice(0, 110)));
        console.log("      js    :", JSON.stringify(mine[i].text.slice(0, 110)));
      }
    }
  });
}
check("every intermediate stage matches too", stageMismatch === 0, String(stageMismatch));

section("the individual rules");
check("wrapping quotes are stripped once",
  decodeExportQuoting('"hello ""world"""') === 'hello "world"',
  decodeExportQuoting('"hello ""world"""'));
check("unwrapped text is left alone", decodeExportQuoting("plain") === "plain");
check("entities decode", unescapeEntities("a &amp; b &lt;c&gt; &#39;d&#39;") === `a & b <c> 'd'`,
  unescapeEntities("a &amp; b &lt;c&gt; &#39;d&#39;"));
check("a closed code block becomes one marker",
  clean("before {code:java}x=1{code} after") === "before [CODE] after",
  clean("before {code:java}x=1{code} after"));
check("an unclosed code block still becomes a marker",
  clean("before {code}dangling") === "before [CODE] dangling",
  clean("before {code}dangling"));
check("links become one marker",
  clean("see https://example.org/a/b?c=d now") === "see [URL] now",
  clean("see https://example.org/a/b?c=d now"));
check("ticket references become one marker",
  clean("fixes ABC-123 today") === "fixes [ISSUE_REF] today",
  clean("fixes ABC-123 today"));
check("bold is unwrapped, not deleted",
  clean("a *bold* word") === "a bold word", clean("a *bold* word"));
check("snake_case survives (italics are left alone)",
  clean("call my_function_name now") === "call my_function_name now",
  clean("call my_function_name now"));
check("inline JSON is not mistaken for a macro",
  clean('config {"retries": 3} ok') === 'config {"retries": 3} ok',
  clean('config {"retries": 3} ok'));
check("known macros are stripped but their words kept",
  clean("{color:red}urgent{color} fix") === "urgent fix",
  clean("{color:red}urgent{color} fix"));
check("headings lose the marker, keep the words",
  clean("h2. Crash on startup") === "Crash on startup", clean("h2. Crash on startup"));

section("the counters");
const counts = countSubstitutions("{code}a{code} see https://x.y and ABC-1 and DEF-2");
check("counts one code block", counts.code_blocks === 1, String(counts.code_blocks));
check("counts one link", counts.urls === 1, String(counts.urls));
check("counts two ticket references", counts.issue_refs === 2, String(counts.issue_refs));
check("a link containing a ticket-like tail is not double counted",
  countSubstitutions("https://example.org/ABC-123").issue_refs === 0,
  String(countSubstitutions("https://example.org/ABC-123").issue_refs));

section("the stop");
const data = await import("../js/core/data.js");
await data.init();
const stops = await import("../js/core/stops.js");
const bench = await import("../js/stations/cleaning-bench.js");
const entry = stops.STOPS.find((s) => s.id === "cleaning-bench");
check("cleaning-bench is marked built", entry.built === true);

const host = document.createElement("div");
let nexted = 0, threw = null;
try {
  await bench.mount(host, { stop: entry, index: 4, next: () => { nexted++; }, goTo: () => {} });
} catch (e) { threw = e; }
check("mount runs without throwing", threw === null, threw && threw.stack);
if (threw) done();
const card = host.children[0];

const stageRows = card.querySelectorAll(".stage");
check("twelve stages are shown", stageRows.length === 12, String(stageRows.length));
check("stages that changed something are marked",
  stageRows.some((s) => s.dataset.changed === "true"));
check("stages that changed nothing are marked too",
  stageRows.some((s) => s.dataset.changed === "false"));
check("five live counters", card.querySelectorAll(".bench-counter").length === 5,
  String(card.querySelectorAll(".bench-counter").length));

const input = card.querySelector("textarea");
check("there is an editable input", Boolean(input));
input.value = "{code}x{code} and https://a.b";
input.fire("input");
const after = card.querySelectorAll(".stage");
check("editing the text recomputes the stages",
  after[after.length - 1].querySelector(".stage-text").textContent === "[CODE] and [URL]",
  after[after.length - 1].querySelector(".stage-text").textContent);

const text = deepText(card);
check("it admits the browser runs a copy of the rules",
  text.includes("It is a copy") || text.includes("a copy"));
check("it says the copy is tested against the original",
  text.includes("character-for-character"));
check("it reports the dormant bullet rule",
  text.includes("cleaning.bullet_hits") && text.includes("cleaning.newline_share"));
check("it lists what cleaning deliberately avoids",
  card.querySelectorAll(".restraint").length === 4,
  String(card.querySelectorAll(".restraint").length));
check("it explains why markers replace rather than delete",
  text.includes("rather than being deleted"));

card.querySelector(".handover-next").fire("click");
check("the handover advances", nexted === 1);

done();

