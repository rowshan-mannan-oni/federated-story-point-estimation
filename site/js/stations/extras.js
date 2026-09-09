/* ==========================================================================
   Stop 9 — Type and priority.

   The words are dealt with. An issue carries two more facts that are not
   words: what kind of task it is, and how urgent somebody marked it. This
   stop puts them in, and asks the question a design decision deserves —
   do they actually carry anything?

   Both answers turn out to be interesting and neither is a clean win:

     - Type has a wide spread (over three story points between its extremes),
       but the four biggest types are indistinguishable from each other, so
       for most issues the field says almost nothing.
     - Priority looks like a scale and is not one. "High" outranks "Medium"
       in name and sits BELOW it in measured effort, because "High" is the
       default several trackers hand out.

   The stop also carries the live parameter counter, which is where the
   number the next stop is built on — what actually gets uploaded — first
   appears and can be poked at.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer, createReadout } from "../ui/readouts.js";
import { createStepper } from "../ui/controls.js";
import { createTable } from "../viz/table.js";
import { createDistStrip, createDistLegend } from "../viz/dist-strip.js";
import { factEl, factText } from "../ui/provenance.js";
import * as data from "../core/data.js";

export async function mount(host, ctx) {
  const split = await data.split();
  const extras = split?.extras;

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "Type and priority",
    standfirst: `Every issue arrives with two labels attached: a kind and an urgency.
      They are not sentences, so they cannot go through the reader &mdash; they take their
      own small route in. Whether they earn their place is a question with a measurable
      answer, and the answer is more interesting than a yes.`,
  });

  card.append(buildSignal(extras));
  card.append(buildRoute());
  card.append(buildCounter());
  card.append(buildSharedOrLocal());

  card.append(handover({
    question: `You have now seen everything the model is made of, and the counter above
      says how much of it gets trained. A hundred and twenty-five million numbers in the
      reader, a quarter of a million in the parts that move. How is that possible &mdash;
      and what happens to the other 99.8%?`,
    cta: "Patches, not models",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   Do they carry anything?
   ========================================================================== */

function buildSignal(extras) {
  const section = block(
    "Do these two labels carry anything?",
    prose(`Adding a field to a model is easy, and easy to leave unexamined. If every kind
      of issue carried the same effort on average, the type label would be dead weight and
      the model would be better off without it. So here is what each label is actually
      worth, measured on the training data.`),
  );

  if (!extras?.types?.length) {
    section.append(prose(`These figures have not been generated yet. Run
      <code>python site/tools/extract_facts.py</code>.`, "note"));
    return section;
  }

  /* ---- type ---- */
  section.append(el("h3", "sub-title", "Issue type"));
  section.append(statRow(
    createStatPlate({
      label: "heaviest kind of task",
      value: factEl("extras.type_highest_mean"),
      caption: factText("extras.type_highest"),
    }),
    createStatPlate({
      label: "lightest kind of task",
      value: factEl("extras.type_lowest_mean"),
      caption: factText("extras.type_lowest"),
    }),
    createStatPlate({
      label: "spread between them",
      value: factEl("extras.type_spread"),
      caption: "story points",
    }),
  ));

  section.append(typeTable(extras.types));

  section.append(prose(`A spread of <span data-fact="extras.type_spread"></span> points
    looks like a strong signal, and at the edges it is: a technical task really is bigger
    work than a build failure. But look at the top of that table. The four largest
    categories &mdash; which between them cover most of the corpus &mdash; average within
    a hundredth of a point of each other. For the typical issue, the type label is telling
    the model almost nothing.`));

  /* ---- priority ---- */
  section.append(el("h3", "sub-title", "Priority"));
  section.append(priorityTable(extras.priorities));

  section.append(prose(`Priority is the stranger of the two. It reads like a scale
    &mdash; Highest, High, Medium, Low, Lowest &mdash; so you would expect the average
    effort to fall steadily down the list. It does not.
    <strong>High sits below Medium.</strong>`));

  section.append(prose(`The likely reason is visible in the counts: "High" is by far the
    largest bucket, because it is the default several trackers hand out when nobody
    chooses. So it is less a level of urgency than a shrug, and it behaves like one. The
    two ends of the scale still separate cleanly &mdash;
    <span data-fact="extras.priority_lowest_mean"></span> against
    <span data-fact="extras.priority_highest_mean"></span> &mdash; which is where the
    field earns its keep.`, "note"));

  const honest = createDrawer({
    label: "Does adding them actually improve the estimates?",
    summary: "unmeasured, and worth saying so",
  });
  honest.body.append(prose(`This stop shows that the two labels <em>contain</em> signal.
    That is not the same as showing the model gets anything out of them. The experiment
    that would settle it &mdash; training with and without the categorical inputs and
    comparing &mdash; is not in the thesis's run plan, and has not been run.`));
  honest.body.append(prose(`So the honest position is: they are cheap, they plainly carry
    some information, and whether the model exploits it is unknown. The related question
    the run plan <em>does</em> list &mdash; whether these embeddings should be shared
    across projects or kept local &mdash; is also unrun, and is described below.`, "note"));
  section.append(honest.el);

  return section;
}

function typeTable(rows) {
  const table = createTable({
    caption: "Issue types by how many issues carry them and their average story point",
    rows: rows.filter((r) => r.n >= 50).map((r) => ({
      value: r.value,
      n: r.n,
      mean_sp: r.mean_sp,
      counts: r.counts,
    })),
    sort: { key: "n", dir: "desc" },
    columns: [
      { key: "value", label: "Type", align: "left" },
      { key: "n", label: "Issues", align: "right", format: (v) => Number(v).toLocaleString() },
      { key: "mean_sp", label: "Average points", align: "right", format: (v) => v.toFixed(2) },
      { key: "counts", label: "Mix", align: "left", sortable: false,
        format: (v) => createDistStrip({ counts: v, compact: true }).el },
    ],
  });
  const scroller = el("div", "table-scroll");
  scroller.append(table.el);
  const wrap = el("div", "sub-block");
  wrap.append(createDistLegend(), scroller);
  return wrap;
}

function priorityTable(rows) {
  /* Ordered as the label claims to be ordered, so the break in the pattern is
     visible where a reader expects to see it rather than hidden by sorting. */
  const order = ["highest", "high", "medium", "low", "lowest", "unknown"];
  const ordered = order
    .map((name) => rows.find((r) => r.value === name))
    .filter(Boolean);

  const list = el("ul", "prio-list");
  const peak = Math.max(...ordered.map((r) => r.mean_sp));

  ordered.forEach((row, i) => {
    const previous = ordered[i - 1];
    const item = el("li", "prio-row mat-sub");
    if (previous && row.value !== "unknown" && previous.value !== "unknown"
        && row.mean_sp > previous.mean_sp) {
      item.dataset.break = "true";
    }
    item.append(el("span", "prio-name", row.value));
    const bar = el("span", "prio-bar");
    const fill = el("span", "prio-fill");
    fill.style.setProperty("--w", String(row.mean_sp / peak));
    bar.append(fill);
    item.append(bar);
    item.append(el("span", "prio-mean", row.mean_sp.toFixed(2)));
    item.append(el("span", "prio-n", `${row.n.toLocaleString()} issues`));
    list.append(item);
  });

  const wrap = el("div", "sub-block");
  wrap.append(list);
  wrap.append(el("p", "hist-caption",
    "Listed in the order the labels claim, not sorted by size — a marked row is one that " +
    "carries more effort than the label above it, which a scale should never do."));
  return wrap;
}

/* ==========================================================================
   How they get in
   ========================================================================== */

function buildRoute() {
  return block(
    "How a label gets into a model",
    prose(`A model cannot take the word "bug". Each distinct label is given a row of
      numbers of its own &mdash; sixteen of them &mdash; and the model learns what those
      numbers should be, the same way it learns everything else. Two small lookup tables:
      one for the types, one for the priorities.`),
    prose(`The tables are not quite the size you would expect.
      <span data-fact="extras.types_seen"></span> types appear in the training data, but
      the table has <span data-fact="split.type_vocab"></span> rows &mdash; one is
      reserved, unused, for a label the model has never seen. Priority needs no such
      reservation: its <span data-fact="split.priority_vocab"></span> rows already include
      an Unknown, because <span data-fact="extras.priority_unknown_share"></span> of
      training issues have no priority recorded and land there.`, "note"),
    prose(`Those two rows are then simply glued onto the end of the row of numbers that
      came out of the reader. 768 from the words, 16 from the type, 16 from the priority,
      making 800 &mdash; and that single row is what the last part of the model actually
      looks at.`),
    routeDiagram(),
    prose(`That reserved row is what a project's private vocabulary falls into. If some
      tracker uses a type nobody else does, the model has never learned anything about it
      and treats it as Unknown &mdash; the same row an issue gets when its priority was
      simply never filled in. Two quite different kinds of missing, sharing one bucket.`,
      "note"),
  );
}

function routeDiagram() {
  const wrap = el("div", "route");
  [
    ["the words", "768", "from the reader"],
    ["the type", "16", "looked up"],
    ["the priority", "16", "looked up"],
  ].forEach(([label, size, note]) => {
    const box = el("div", "route-part mat-sub");
    box.append(el("span", "route-size", size));
    box.append(el("span", "route-label", label));
    box.append(el("span", "route-note", note));
    wrap.append(box);
  });

  const join = el("div", "route-join");
  join.append(el("span", "route-arrow", "→"));
  const out = el("div", "route-part route-out mat-recess");
  out.append(el("span", "route-size", "800"));
  out.append(el("span", "route-label", "one row"));
  out.append(el("span", "route-note", "goes to the head"));
  join.append(out);
  wrap.append(join);
  return wrap;
}

/* ==========================================================================
   The live counter
   ========================================================================== */

/* Fixed by the encoder and the run's own settings; the dials move the rest. */
const TEXT_DIM = 768;
const LORA_B = 147456;
const CORN_OUTPUTS = 4;

function buildCounter() {
  const types = data.value("split.type_vocab") ?? 18;
  const priorities = data.value("split.priority_vocab") ?? 6;

  let embDim = 16;
  let hidden = 128;

  const trainable = createReadout({
    label: "numbers trained and uploaded",
    value: 0,
    size: "lg",
    caption: "per project, per round",
  });
  const upload = createReadout({
    label: "size of one upload",
    value: 0,
    unit: "MB",
    format: (v) => v.toFixed(2),
    caption: "four bytes each",
  });

  const breakdown = el("ul", "count-breakdown");
  const verdict = el("p", "count-verdict");

  function recompute() {
    const fusion = TEXT_DIM + 2 * embDim;
    const embeddings = (types + priorities) * embDim;
    const head = 2 * fusion                      // LayerNorm: a scale and a shift
      + (fusion * hidden + hidden)               // Linear 800 -> 128
      + (hidden * CORN_OUTPUTS + CORN_OUTPUTS);  // Linear 128 -> 4 thresholds
    const total = LORA_B + embeddings + head;

    trainable.set(total);
    upload.set((total * 4) / 1e6);

    breakdown.replaceChildren();
    [
      ["the trained half of the patches", LORA_B, "fixed by the encoder"],
      ["the two lookup tables", embeddings, `(${types} + ${priorities}) × ${embDim}`],
      ["the head", head, `over a row of ${fusion}`],
    ].forEach(([label, count, note]) => {
      const item = el("li", "count-row");
      item.append(el("span", "count-n", Number(count).toLocaleString()));
      item.append(el("span", "count-label", label));
      item.append(el("span", "count-note", note));
      breakdown.append(item);
    });

    const real = data.value("params.trainable");
    const atDefaults = embDim === 16 && hidden === 128;
    if (real && atDefaults) {
      verdict.dataset.match = String(total === real);
      verdict.textContent = total === real
        ? `That is the run's own setting, and it comes to exactly the ${real.toLocaleString()} numbers recorded in its results.`
        : `Mismatch: this page computes ${total.toLocaleString()} where the run recorded ${real.toLocaleString()}.`;
    } else {
      verdict.dataset.match = "";
      verdict.textContent = real
        ? `The run itself used 16 and 128, which comes to ${real.toLocaleString()}.`
        : "";
    }
  }

  const embStepper = createStepper({
    label: "numbers per label",
    min: 4, max: 64, step: 4, value: embDim,
    onChange: (v) => { embDim = v; recompute(); },
  });
  const hiddenStepper = createStepper({
    label: "width of the head",
    min: 32, max: 512, step: 32, value: hidden,
    onChange: (v) => { hidden = v; recompute(); },
  });

  recompute();

  const controls = el("div", "count-controls");
  controls.append(embStepper.el, hiddenStepper.el);

  const readouts = el("div", "count-readouts");
  readouts.append(trainable.el, upload.el);

  return block(
    "Try it — what the size dials cost",
    prose(`Two settings decide how big the trainable part is: how many numbers each label
      gets, and how wide the head is. Move them and watch what one project would have to
      upload every round.`),
    controls,
    readouts,
    breakdown,
    verdict,
    prose(`Notice what barely moves. Even at the widest settings the head and the tables
      stay a small fraction of the upload &mdash; the bulk is the patch on the reader, and
      that is fixed by the encoder rather than by anything on this page.`, "note"),
  );
}

/* ==========================================================================
   Shared or local?
   ========================================================================== */

function buildSharedOrLocal() {
  return block(
    "Should everyone share the same tables?",
    prose(`These two tables are trained and averaged across projects like everything else
      that moves. That is defensible for priority: the words were flattened onto one
      vocabulary at export time, so "Highest" means the same thing everywhere, and
      pooling what nineteen projects learned about urgency should help.`),
    prose(`It is shakier for type. Those labels were deliberately <em>not</em> merged
      &mdash; a "Bug" and a "Defect" stay separate &mdash; so the shared table is being
      asked to learn one meaning for a word whose sense may differ between trackers. The
      alternative is to keep the tables local, like the head.`),
    prose(`The thesis lists shared-versus-local embeddings as an optional micro-ablation
      and has not run it. It is named here rather than quietly settled, because the
      default was a choice and not a finding.`, "note"),
  );
}
