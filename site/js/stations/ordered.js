/* ==========================================================================
   Stop 12 — The answers are ordered.

   The poker table established that 1, 2, 3, 5, 8 is a scale. Nothing since
   has used that fact. Most classification setups throw it away: five answers,
   five unrelated names, every wrong answer equally wrong.

   This stop makes the loss feel the distance. Two things have to land:

     1. Why the flat view is wrong — shown as a cost grid, where the ordered
        version rises away from the diagonal and the unordered one does not.
     2. How the ordinal head works — four yes/no questions instead of five
        scores, decoded by counting how many were answered yes. Rank
        consistency is not enforced afterwards; it falls out of the decoding,
        which is the part worth actually demonstrating.

   The honest limit, on the page: whether this helps is unmeasured here. The
   CE-versus-CORN comparison is one flag away and listed as optional in the
   thesis's own notes.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer, createReadout } from "../ui/readouts.js";
import { createSegmented, createSwitch } from "../ui/controls.js";
import { factEl, factText } from "../ui/provenance.js";
import * as data from "../core/data.js";

/* The deck, and the questions the ordinal head is asked about it. */
const POINTS = [1, 2, 3, 5, 8];
const QUESTIONS = ["more than 1?", "more than 2?", "more than 3?", "more than 5?"];

export async function mount(host, ctx) {
  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "The answers are ordered",
    standfirst: `Five possible answers &mdash; 1, 2, 3, 5, 8 &mdash; and they are not five
      names. They are a scale, and saying 8 when the answer was 3 is a worse mistake than
      saying 2. Almost every way of building a classifier forgets that. This stop is about
      remembering it.`,
  });

  card.append(buildCostGrid());
  card.append(buildOrgan());
  card.append(buildDecoding());
  card.append(buildCost());
  card.append(buildHonesty());

  card.append(handover({
    question: `The model is complete: a frozen reader, patches that merge exactly, two
      small tables, and a head that knows its answers are ordered. Nothing has trained it
      yet. What does one project actually <em>do</em> when its turn comes?`,
    cta: "One project's turn",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   What a mistake costs, under three different lenses
   ========================================================================== */

/** How many of the four threshold questions a guess gets wrong. */
function thresholdsWrong(truthIndex, guessIndex) {
  return Math.abs(truthIndex - guessIndex);
}

function buildCostGrid() {
  const section = block(
    "What a mistake is worth",
    prose(`Suppose the answer is 3 and the model says 8. How bad is that? It depends
      entirely on how the question was set up, and the three answers below are all in use
      in this thesis.`),
  );

  let lens = "corn";

  const grid = el("div", "cost-grid-wrap");
  const legend = el("p", "cost-legend");

  function costOf(truthIndex, guessIndex) {
    if (lens === "flat") return truthIndex === guessIndex ? 0 : 1;
    if (lens === "corn") return thresholdsWrong(truthIndex, guessIndex);
    return Math.abs(POINTS[truthIndex] - POINTS[guessIndex]);   // "points"
  }

  function paint() {
    grid.replaceChildren();
    const table = el("table", "cost-grid");

    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    headRow.append(el("th", "cost-corner", "truth ↓ / guess →"));
    POINTS.forEach((p) => headRow.append(el("th", null, String(p))));
    head.append(headRow);
    table.append(head);

    const body = document.createElement("tbody");
    const peak = Math.max(...POINTS.map((_, t) => Math.max(...POINTS.map((__, g) => costOf(t, g)))), 1);

    POINTS.forEach((truth, t) => {
      const row = document.createElement("tr");
      row.append(el("th", null, String(truth)));
      POINTS.forEach((guess, g) => {
        const cost = costOf(t, g);
        const cell = el("td", "cost-cell");
        cell.style.setProperty("--v", String(cost / peak));
        cell.textContent = lens === "points" ? String(cost) : String(cost);
        cell.title = `truth ${truth}, guess ${guess}: cost ${cost}`;
        if (t === g) cell.dataset.right = "true";
        row.append(cell);
      });
      body.append(row);
    });
    table.append(body);
    grid.append(table);

    legend.textContent = {
      flat: "Every wrong answer costs exactly the same. Saying 8 for a 3 is penalised no "
          + "more than saying 2 — the scale might as well be five colours.",
      corn: "The cost is how many of the four yes/no questions the guess gets wrong, which "
          + "is the distance along the deck. Being one step out costs one; being four "
          + "steps out costs four.",
      points: "The cost is the gap in actual story points, which is what MAE measures. "
          + "Note it is not evenly spaced — the deck jumps from 5 to 8, so that mistake "
          + "costs three.",
    }[lens];
  }

  const choice = createSegmented({
    label: "how the mistake is counted",
    options: [
      { value: "corn", label: "Ordered head" },
      { value: "flat", label: "Unordered head" },
      { value: "points", label: "In story points" },
    ],
    value: lens,
    onChange: (v) => { lens = v; paint(); },
  });

  paint();
  section.append(choice.el, grid, legend);

  section.append(prose(`Switch to the unordered head and look at the shape. Every cell off
    the diagonal is identical. That is a model being told, during every step of training,
    that a near miss and a wild miss are the same failure &mdash; and it will happily
    learn to make wild misses if they are marginally easier to get exactly right
    sometimes.`, "note"));

  return section;
}

/* ==========================================================================
   The threshold organ
   ========================================================================== */

function buildOrgan() {
  const section = block(
    "Try it — four questions instead of five scores",
    prose(`The ordered head does not score the five answers. It answers four yes/no
      questions about where the truth sits on the deck, and the answer is read off from
      how many it said yes to.`),
  );

  const answers = [false, false, false, false];

  const organ = el("div", "organ");
  const readout = createReadout({
    label: "the answer this decodes to",
    value: POINTS[0],
    format: (v) => String(v),
    caption: "count the leading yeses",
  });
  const trace = el("p", "organ-trace");

  function decode() {
    // CORN's rule: walk the questions in order and stop at the first "no".
    // Anything after that first no cannot change the answer — which is what
    // makes an inconsistent set of answers harmless rather than illegal.
    let count = 0;
    while (count < answers.length && answers[count]) count += 1;
    return count;
  }

  function paint() {
    const index = decode();
    readout.set(POINTS[index]);

    const said = answers.map((a) => (a ? "yes" : "no"));
    const firstNo = said.indexOf("no");
    const inconsistent = firstNo !== -1 && said.slice(firstNo).includes("yes");

    trace.dataset.odd = String(inconsistent);
    trace.textContent = inconsistent
      ? `Answers: ${said.join(", ")}. That is self-contradictory — it says the value is `
        + `not above ${POINTS[firstNo]} and yet is above something larger. The decoding `
        + `stops at the first no, so the later answers are simply ignored and the result `
        + `is still a real story point: ${POINTS[index]}.`
      : `Answers: ${said.join(", ")}. ${index} yes${index === 1 ? "" : "es"} in a row, `
        + `so the answer is ${POINTS[index]}.`;
  }

  QUESTIONS.forEach((question, i) => {
    const row = el("div", "organ-row");
    const sw = createSwitch({
      label: question,
      checked: false,
      onChange: (v) => { answers[i] = v; paint(); },
    });
    row.append(sw.el);
    organ.append(row);
  });

  paint();
  section.append(organ, readout.el, trace);

  section.append(prose(`Turn them on from the top and watch the answer climb the deck.
    Then try something contradictory &mdash; say no to the first question and yes to the
    last. A model that produced five independent scores could return something
    meaningless; this cannot. The reading rule stops at the first no, so every possible
    set of answers decodes to a real story point.`, "note"));

  return section;
}

/* ==========================================================================
   Why the loss follows the distance
   ========================================================================== */

function buildDecoding() {
  const section = block(
    "Why this makes distance cost more",
    prose(`During training each of those four questions is graded on its own, and the
      penalties add up. That is the whole mechanism &mdash; there is no special
      distance-aware loss function, just four ordinary yes/no questions.`),
  );

  const example = el("div", "grade-example");
  const truthIndex = 2;   // the answer is 3

  [1, 2, 3, 5, 8].forEach((guess, g) => {
    const wrong = thresholdsWrong(truthIndex, g);
    const row = el("div", "grade-row mat-sub");
    row.dataset.right = String(wrong === 0);
    row.append(el("span", "grade-guess", `says ${guess}`));

    const marks = el("span", "grade-marks");
    QUESTIONS.forEach((_, k) => {
      const truthSays = truthIndex > k;
      const guessSays = g > k;
      const mark = el("span", "grade-mark", truthSays === guessSays ? "✓" : "✗");
      mark.dataset.ok = String(truthSays === guessSays);
      mark.title = `${QUESTIONS[k]} truth ${truthSays ? "yes" : "no"}, guess ${guessSays ? "yes" : "no"}`;
      marks.append(mark);
    });
    row.append(marks);
    row.append(el("span", "grade-count",
      wrong === 0 ? "nothing wrong" : `${wrong} wrong`));
    example.append(row);
  });

  section.append(el("p", "grade-caption", "When the true answer is 3:"));
  section.append(example);

  section.append(prose(`Guessing 2 gets one question wrong. Guessing 8 gets three wrong,
    so it is penalised three times over. The further the guess, the more questions it
    fails &mdash; the distance is built into the arithmetic rather than bolted on.`));

  return section;
}

/* ==========================================================================
   What it costs to build
   ========================================================================== */

function buildCost() {
  return block(
    "What the ordered head costs",
    statRow(
      createStatPlate({
        label: "outputs, ordered head",
        value: factEl("head.outputs"),
        caption: "one per threshold question",
      }),
      createStatPlate({
        label: "outputs, unordered head",
        value: factEl("head.outputs_other"),
        caption: "one per story point",
      }),
      createStatPlate({
        label: "difference in size",
        value: factEl("head.params_delta"),
        caption: "numbers — out of a quarter of a million",
      }),
    ),
    prose(`Essentially nothing. The ordered head is
      <span data-fact="head.params_delta"></span> numbers <em>smaller</em>, because four
      questions need fewer outputs than five scores. Whatever the ordered head is worth,
      it is not costing anything to find out.`),
    prose(`One difference is worth knowing about. The unordered head is trained with
      class weights &mdash; rarer story points are given more importance so they are not
      ignored. The ordered head is not, because its threshold structure already spreads
      attention across the deck. The mix here is mild enough
      (<span data-fact="corpus.imbalance_ratio"></span>) that this is a minor point, and
      the archive stop settled that already.`, "note"),
  );
}

/* ==========================================================================
   What is not known
   ========================================================================== */

function buildHonesty() {
  const section = block(
    "Does it actually help?",
    prose(`This stop has argued that treating the answers as ordered is more sensible than
      treating them as five names. It has not shown that it produces better estimates,
      because that comparison has not been run.`),
  );

  const detail = createDrawer({
    label: "What the comparison would look like",
    summary: "and what to expect from it",
  });
  detail.body.append(prose(`Running the whole thing again with the unordered head is a
    single flag. The thesis lists it as an optional question rather than one of its three
    main ones, and it is not in the finished run on this machine.`));
  detail.body.append(prose(`There is a prediction worth writing down in advance, though.
    If the ordered head is doing what this page claims, it should barely move macro-F1
    &mdash; that score cannot see distance at all &mdash; while improving the two scores
    that can: average error in story points, and the chance-corrected one that weights
    distant mistakes more heavily. A result where only the distance-aware scores move
    would be strong evidence the model is learning the scale rather than five labels. That
    asymmetry is the finding to look for, not a raw improvement.`, "note"));
  section.append(detail.el);

  section.append(prose(`There is already a hint in the finished run, from a different
    direction. Of every mistake the federated model makes,
    <span data-fact="confusion.within_one"></span> land on the true answer or one step
    from it, and only <span data-fact="confusion.far"></span> are three or more steps out.
    The mistakes cluster along the scale rather than scattering across it &mdash; which is
    what a model that has learned an ordering looks like. Those numbers get properly
    examined at the dials.`, "note"));

  return section;
}
