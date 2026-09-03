/* ==========================================================================
   Stop 7 — Cutting the data three ways.

   Everything measured later in this walk-through depends on this stop being
   right, and it is the easiest place in a study to fool yourself. Two things
   have to land:

     1. Three piles, three different jobs — and the last one is opened once.
     2. Cutting at random lets the model learn from issues written AFTER the
        ones it is tested on. Cutting by date does not. The site shows this
        rather than asserting it: the same data, cut both ways, with the real
        date ranges side by side.

   The honest complication, stated on the page: the finished run on this
   machine used the random cut. The date-based runs are the ones the thesis
   treats as primary, and they have not finished yet.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { createSegmented } from "../ui/controls.js";
import { createTable } from "../viz/table.js";
import { createSpans, overlaps } from "../viz/spans.js";
import { factEl, factText } from "../ui/provenance.js";
import { banner } from "../ui/stamp.js";
import * as data from "../core/data.js";

export async function mount(host, ctx) {
  const split = await data.split();

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "Cutting the data three ways",
    standfirst: `The data is clean. Now it has to be divided: some to learn from, some to
      choose between versions with, and some sealed in an envelope until the very end.
      How that cut is made decides whether every number later in this walk-through means
      anything at all.`,
  });

  card.append(buildPiles(split));
  card.append(buildComparison(split));
  card.append(buildHonestCut(split));
  card.append(buildOpenedOnce());

  card.append(handover({
    question: `The data is clean, divided, and sealed where it needs to be. Time to look
      at the thing that reads it &mdash; how a sentence about a bug becomes numbers a
      model can work with.`,
    cta: "The reading machine",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   Three piles
   ========================================================================== */

function buildPiles(split) {
  const section = block(
    "Three piles, three different jobs",
    statRow(
      createStatPlate({
        label: "to learn from",
        value: factEl("split.train"),
        caption: "the training pile",
      }),
      createStatPlate({
        label: "to choose with",
        value: factEl("split.val"),
        caption: "the check pile",
      }),
      createStatPlate({
        label: "to measure with, once",
        value: factEl("split.test"),
        caption: "the test pile",
      }),
    ),
    prose(`The <span data-term="training set">training pile</span> is what the model
      learns from. The <span data-term="check set">check pile</span> is what decides
      questions like "which round of training produced the best model" &mdash; questions
      you have to answer somehow, and must not answer with the test data. The
      <span data-term="test set">test pile</span> is opened once, at the end, to produce
      the numbers that get reported.`),
    prose(`Every project cuts its own data this way, rather than one cut being made across
      the pooled corpus. That matters here more than usual: each project has to appear in
      all three piles, because each one is going to be measured separately.`, "note"),
  );

  const projects = split?.per_project ?? [];
  if (projects.length) {
    const table = createTable({
      caption: "How each project's issues are divided",
      rows: projects.map((row) => ({
        name: row.name.replace(/_/g, " "),
        train: row.train,
        val: row.val,
        test: row.test,
        total: row.train + row.val + row.test,
      })),
      sort: { key: "total", dir: "desc" },
      columns: [
        { key: "name", label: "Project", align: "left" },
        { key: "train", label: "Learn", align: "right", format: fmt },
        { key: "val", label: "Check", align: "right", format: fmt },
        { key: "test", label: "Test", align: "right", format: fmt },
        { key: "total", label: "Total", align: "right", format: fmt },
      ],
    });
    const scroller = el("div", "table-scroll");
    scroller.append(table.el);
    section.append(scroller);
  }

  return section;
}

const fmt = (v) => Number(v).toLocaleString();

/* ==========================================================================
   The same data, cut both ways
   ========================================================================== */

function buildComparison(split) {
  const modes = split?.modes ?? {};
  const available = Object.keys(modes);

  const section = block(
    "Try it — cut at random, or cut by date",
    prose(`Below is one project's data, divided both ways. Watch what happens to
      <em>when</em> the issues in each pile were written.`),
  );

  if (available.length < 2) {
    section.append(prose(`Both split modes have not been generated yet. Run
      <code>python site/tools/extract_facts.py</code> to fill this in.`, "note"));
    return section;
  }

  const projectNames = modes[available[0]].projects.map((p) => p.name);
  let mode = "random";
  let project = projectNames[0];

  const chart = el("div", "split-chart");
  const verdict = el("p", "split-verdict");

  const modeChoice = createSegmented({
    label: "How to cut",
    options: [
      { value: "random", label: "At random" },
      { value: "temporal", label: "By date" },
    ],
    value: mode,
    onChange: (next) => { mode = next; paint(); },
  });

  const picker = document.createElement("select");
  picker.className = "split-picker";
  picker.setAttribute("aria-label", "Which project to show");
  projectNames.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name.replace(/_/g, " ");
    picker.append(option);
  });
  picker.addEventListener("change", () => { project = picker.value; paint(); });

  function paint() {
    const info = modes[mode]?.projects.find((p) => p.name === project);
    chart.replaceChildren();
    if (!info) return;

    chart.append(createSpans({
      rows: [
        { label: "Learn", ...info.train, tone: "plain" },
        { label: "Check", ...info.val, tone: "mid" },
        { label: "Test", ...info.test, tone: "strong" },
      ],
    }).el);

    const clash = overlaps(info.train, info.test);
    verdict.textContent = clash
      ? "The training and test piles cover the same years. The model can learn from issues written after the ones it is judged on."
      : "The piles follow one another. Everything the model learns from was written before everything it is judged on.";
    verdict.dataset.leak = String(clash);
  }

  const controls = el("div", "split-controls");
  controls.append(modeChoice.el, picker);
  section.append(controls, chart, verdict);

  paint();
  return section;
}

/* ==========================================================================
   Why the date cut is the honest one
   ========================================================================== */

function buildHonestCut(split) {
  const section = block(
    "Why cutting by date is the honest test",
    prose(`A random cut quietly asks the model an easier question than the real one. The
      real question is "here is a task nobody has estimated yet" &mdash; a task from
      <em>next</em> month. A random cut instead asks "here is a task from the middle of a
      history you have already read most of", and lets the model learn from issues written
      after it, beside it, sometimes about the very same piece of work.`),
    prose(`Teams also drift. What a team called a 5 in 2014 is not what it calls a 5 in
      2019 &mdash; and a random cut hides that drift by mixing the years together, while a
      date cut makes the model face it. That is the situation any real deployment is in:
      trained on the past, used on the future.`),
  );

  const used = split?.mode_used;
  if (used) {
    section.append(banner("preliminary",
      `The finished run on this machine used the ${used === "random" ? "random" : "date-based"} cut. ` +
      `The thesis treats the date-based cut as its primary setting and the random one as a ` +
      `robustness check; those runs are still to come. Every result later in this ` +
      `walk-through therefore comes from the ${used} cut, and says so.`));
  }

  const totals = createDrawer({
    label: "Do the two cuts even produce the same sized piles?",
    summary: "nearly, and the difference is instructive",
  });
  const modes = split?.modes ?? {};
  if (modes.random && modes.temporal) {
    const rows = ["train", "val", "test"].map((key) => ({
      pile: { train: "Learn", val: "Check", test: "Test" }[key],
      random: modes.random.totals[key],
      temporal: modes.temporal.totals[key],
      diff: modes.temporal.totals[key] - modes.random.totals[key],
    }));
    const table = createTable({
      caption: "Pile sizes under each way of cutting",
      rows,
      sort: { key: "random", dir: "desc" },
      columns: [
        { key: "pile", label: "Pile", align: "left" },
        { key: "random", label: "Cut at random", align: "right", format: fmt },
        { key: "temporal", label: "Cut by date", align: "right", format: fmt },
        { key: "diff", label: "Difference", align: "right",
          format: (v) => (v > 0 ? `+${v}` : String(v)) },
      ],
    });
    const scroller = el("div", "table-scroll");
    scroller.append(table.el);
    totals.body.append(scroller);
  }
  totals.body.append(prose(`Almost the same, but not exactly. The random cut is stratified
    &mdash; it deliberately keeps the mix of story points even across the piles &mdash;
    while the date cut simply takes the latest slice and accepts whatever mix that
    produces. A few issues land differently as a result, and small projects round
    differently.`, "note"));
  section.append(totals.el);

  return section;
}

/* ==========================================================================
   The rule that makes the numbers mean anything
   ========================================================================== */

function buildOpenedOnce() {
  return block(
    "The last pile is opened once",
    prose(`Here is the rule that everything after this depends on. The test pile is used
      exactly once per approach, at the end, to produce a number that gets reported. It is
      never used to choose anything.`),
    prose(`The moment you look at the test result and go back to change something &mdash;
      a setting, a stopping point, which round to keep &mdash; you have used the test data
      to make a decision, and the number it gives you is no longer an estimate of how the
      model does on data it has not seen. It becomes an estimate of how well you tuned
      against that particular pile.`),
    prose(`That is the whole reason for the middle pile. Every question that needs
      answering during training gets answered with the check pile, so the test pile can
      stay sealed. There is a stop later about making sure every approach in this study
      was judged by that same rule &mdash; because comparing a model that got to peek
      against one that did not is worse than not comparing at all.`, "note"),
  );
}
