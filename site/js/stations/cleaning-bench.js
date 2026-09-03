/* ==========================================================================
   Stop 5 — The cleaning bench.

   Between the archive and the model sits a step nobody writes papers about:
   turning what an engineer typed into something a model can read. This stop
   shows it happening, one rule at a time, on real text.

   The bench is live — you can paste your own text in — which means the
   browser has its own copy of the cleaning rules (js/sim/clean.js). A copy is
   only worth having if it agrees with the original, so the extractor writes
   out hundreds of raw/cleaned pairs from the real corpus and the test suite
   checks this copy against every one of them. The stop says so, because "we
   reimplemented it in JavaScript, trust us" is not good enough.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { createTable } from "../viz/table.js";
import { factEl, factText } from "../ui/provenance.js";
import { cleanStages, countSubstitutions } from "../sim/clean.js";
import * as data from "../core/data.js";

export async function mount(host, ctx) {
  const cleaning = await data.cleaning();

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "The cleaning bench",
    standfirst: `Issue text is not prose. It is prose with a stack trace pasted into the
      middle of it, a link to a build that no longer exists, and a table somebody drew
      with pipe characters. Before any of it reaches the model, it has to be tidied
      &mdash; carefully, because the tidying can throw away the very thing that makes a
      task look hard.`,
  });

  /* ---- the bench -------------------------------------------------------- */
  card.append(buildBench(cleaning));

  /* ---- what it does across the whole corpus ----------------------------- */
  card.append(buildTotals(cleaning));

  /* ---- what we deliberately do not do ----------------------------------- */
  card.append(buildRestraint());

  card.append(handover({
    question: `That is the messy markup dealt with. But two of the rules you just watched
      barely fire on this data, one never fires at all, and the pipeline has an alarm
      that refuses to train on anything it has not cleaned. Worth a closer look.`,
    cta: "Links, numbers, leftovers",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   The bench: text in, stages out
   ========================================================================== */

function buildBench(cleaning) {
  const showcase = cleaning?.showcase ?? [];
  const starting = showcase[0]?.raw
    ?? 'A sample issue with {code}some pasted code{code} and a link https://example.org/build/42 (see ABC-123).';

  const section = block(
    "Watch it happen",
    prose(`Real text from a real tracker. Every stage below is one rule, applied in
      order; the ones that changed something are marked. Edit the text, or paste your
      own, and it all recomputes.`),
  );

  const input = document.createElement("textarea");
  input.className = "bench-input mat-recess";
  input.rows = 6;
  input.spellcheck = false;
  input.value = starting;
  input.setAttribute("aria-label", "Raw issue text to clean");

  const controls = el("div", "bench-controls");
  showcase.forEach((example, i) => {
    const button = el("button", "ctrl");
    button.type = "button";
    button.append(el("span", "ctrl-label", `Example ${i + 1}`));
    button.title = example.rules.join(", ");
    button.addEventListener("click", () => { input.value = example.raw; render(); });
    controls.append(button);
  });
  const messy = el("button", "ctrl");
  messy.type = "button";
  messy.append(el("span", "ctrl-label", "Something of everything"));
  messy.addEventListener("click", () => { input.value = KITCHEN_SINK; render(); });
  controls.append(messy);

  const counters = el("div", "bench-counters");
  const stagesEl = el("ol", "stage-list");

  function render() {
    const raw = input.value;
    const stages = cleanStages(raw);
    const counts = countSubstitutions(raw);

    counters.replaceChildren();
    [
      ["characters in", raw.length],
      ["characters out", stages[stages.length - 1].text.length],
      ["code blocks", counts.code_blocks],
      ["links", counts.urls],
      ["ticket references", counts.issue_refs],
    ].forEach(([label, value]) => {
      const plate = el("div", "bench-counter mat-sub");
      plate.append(el("strong", null, Number(value).toLocaleString()));
      plate.append(el("span", null, label));
      counters.append(plate);
    });

    stagesEl.replaceChildren();
    stages.forEach((stage, i) => {
      const item = el("li", "stage");
      item.dataset.changed = String(stage.changed);
      const head = el("div", "stage-head");
      head.append(el("span", "stage-n", String(i + 1).padStart(2, "0")));
      head.append(el("span", "stage-what", stage.what));
      head.append(el("span", "stage-flag", stage.changed ? "changed" : "no change"));
      item.append(head);
      const body = el("p", "stage-text");
      body.textContent = stage.text || "(nothing left)";
      item.append(body);
      stagesEl.append(item);
    });
  }

  input.addEventListener("input", render);
  render();

  section.append(controls, input, counters, stagesEl);

  const trust = createDrawer({
    label: "Is this the real cleaning, or a website imitation of it?",
    summary: "a fair question",
  });
  trust.body.append(prose(`It is a copy. The thesis cleans its data in Python; this page
    runs a JavaScript translation of those same rules so the text can change as you
    type.`));
  trust.body.append(prose(`A copy is worth nothing unless it agrees with the original, so
    it is checked rather than promised: the extraction script writes out
    <span data-fact="cleaning.vectors"></span> raw-and-cleaned pairs taken from the real
    corpus, and this site's tests run every one of them through the JavaScript and demand
    an exact character-for-character match. If the two ever diverge, the tests fail
    before the page is published.`, "note"));
  section.append(trust.el);

  return section;
}

const KITCHEN_SINK = [
  '"h2. Crash on startup',
  "The service dies when *config* is missing. See ABC-1234 and https://example.org/logs/9",
  "{code:java}Exception in thread main NullPointerException{code}",
  "{color:red}urgent{color} || col A || col B || &amp; entities &lt;too&gt;",
  '<p>Some <b>HTML</b> crept in as well</p>"',
].join(" ");

/* ==========================================================================
   What the rules do across the whole corpus
   ========================================================================== */

function buildTotals(cleaning) {
  const section = block(
    "Across all nineteen projects",
    statRow(
      createStatPlate({
        label: "code blocks replaced",
        value: factEl("cleaning.code_blocks"),
        caption: "pasted stack traces, logs, snippets",
      }),
      createStatPlate({
        label: "links replaced",
        value: factEl("cleaning.urls"),
        caption: "builds, docs, dead servers",
      }),
      createStatPlate({
        label: "ticket references replaced",
        value: factEl("cleaning.issue_refs"),
        caption: "pointers to other issues",
      }),
    ),
    prose(`Each of those becomes a single marker &mdash; <code>[CODE]</code>,
      <code>[URL]</code>, <code>[ISSUE_REF]</code> &mdash; rather than being deleted. That
      is deliberate. "This task involved pasting a stack trace" is a real signal about how
      hard a task was; the contents of the trace are not, and a model that reads them
      learns the names of somebody's Java packages instead of the shape of the work.`),
  );

  const hits = cleaning?.rule_hits;
  if (hits) {
    const rows = Object.entries(hits).map(([name, entry]) => ({
      rule: RULE_NAMES[name] ?? name,
      rows: entry.rows,
      pct: entry.pct,
    }));
    const table = createTable({
      caption: "How often each rule finds anything to do",
      rows,
      sort: { key: "rows", dir: "desc" },
      columns: [
        { key: "rule", label: "Rule", align: "left" },
        { key: "rows", label: "Descriptions", align: "right",
          format: (v) => Number(v).toLocaleString() },
        { key: "pct", label: "Share", align: "right", format: (v) => `${v.toFixed(1)}%` },
      ],
    });
    const scroller = el("div", "table-scroll");
    scroller.append(table.el);
    section.append(scroller);

    section.append(prose(`Note the bottom of that list. The rule that strips bullet
      markers finds <span data-fact="cleaning.bullet_hits"></span> in the entire corpus,
      because <span data-fact="cleaning.newline_share"></span> of these descriptions
      contain a line break &mdash; whatever produced this export had already flattened
      them. The rule is not wrong, it is simply dormant on this data, and it would matter
      again on a fresher export.`, "note"));
  }

  return section;
}

const RULE_NAMES = {
  code: "Code blocks, opened and closed",
  orphan: "Code blocks left unclosed",
  noformat: "Preformatted blocks",
  macro: "Jira macros (colour, panel, quote…)",
  heading: "Wiki headings",
  bullet: "List markers",
  bold: "Bold markers",
  table: "Table markers",
  url: "Web links",
  ref: "References to other tickets",
};

/* ==========================================================================
   The rules that were deliberately not written
   ========================================================================== */

function buildRestraint() {
  const section = block(
    "What the bench deliberately does not do",
    prose(`Most text-cleaning advice comes from an older generation of methods that
      counted words. Applying it here would quietly destroy the signal.`),
  );

  const list = el("ul", "restraint-list");
  [
    ["Nothing is lower-cased",
     "<code>NullPointerException</code> is not the same word as <code>nullpointerexception</code>. The model reads code-flavoured English, and capitalisation is part of it."],
    ["No stop-words are removed",
     "“the” and “of” carry grammar, and the model is built to read grammar. Stripping them helps a word-counter and hurts a language model."],
    ["No stemming, no lemmatising",
     "Chopping “running” down to “run” throws away tense for no gain here."],
    ["Numbers and punctuation stay",
     "“increase the timeout from 30s to 300s” is an estimate of effort hiding in two numbers."],
  ].forEach(([title, body]) => {
    const item = el("li", "restraint mat-sub");
    item.append(el("span", "restraint-title", title));
    const text = el("p", "restraint-body");
    text.innerHTML = body;
    item.append(text);
    list.append(item);
  });

  section.append(list);
  section.append(prose(`There is one more rule of the same kind, and it is about where the
    cleaning happens rather than what it does: all of it runs <em>once</em>, when the data
    is exported, and never again during training. That is what makes a run repeatable
    &mdash; and there is an alarm to enforce it, which is the next stop.`, "note"));
  return section;
}
