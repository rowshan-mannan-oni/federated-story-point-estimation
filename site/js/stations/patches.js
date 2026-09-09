/* ==========================================================================
   Stop 10 — Patches, not models.

   The previous stop ended on a promise: the reader is frozen, and that
   decision is what makes the whole design possible. This stop cashes it.

   The idea to land is the low-rank one, and it is genuinely simple once seen
   as areas: a square of 768 by 768 against two strips 8 wide. The square is
   589,824 numbers; the strips are 12,288. Same job, two percent of the size.

   Every figure here is derived from the run's own recorded parameter counts —
   including the layer count, which is worked backwards from the patch total
   rather than assumed. The rank dial recomputes the same arithmetic live, and
   at the run's own rank it lands on the recorded number exactly.

   What is deliberately NOT here: why one of the two strips is frozen. That is
   the next stop, and it is the argument the method rests on.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer, createReadout } from "../ui/readouts.js";
import { createStepper } from "../ui/controls.js";
import { factEl, factText } from "../ui/provenance.js";
import * as data from "../core/data.js";

const WIDTH = 768;

export async function mount(host, ctx) {
  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "Patches, not models",
    standfirst: `The reader has <span data-fact="params.total"></span> numbers in it, and
      training never changes a single one of them. Everything the model learns about story
      points is held in <span data-fact="params.trainable"></span> numbers bolted on
      alongside &mdash; two tenths of one percent of the whole. This stop is how that is
      possible.`,
  });

  card.append(buildFrozen());
  card.append(buildRank());
  card.append(buildWhereTheyAttach());
  card.append(buildWhatItBuys());

  card.append(handover({
    question: `One detail was slipped past you just now. Each patch is <em>two</em> strips
      multiplied together &mdash; and only one of them is ever trained. The other is
      frozen at its random starting values and never learns anything. That sounds like a
      waste of half the patch. It is the single decision that makes federated averaging
      work at all.`,
    cta: "The merging problem",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   The frozen reader
   ========================================================================== */

function buildFrozen() {
  return block(
    "Nothing in the reader ever changes",
    statRow(
      createStatPlate({
        label: "numbers in the reader",
        value: factEl("params.frozen"),
        caption: "frozen — never touched by training",
      }),
      createStatPlate({
        label: "numbers actually trained",
        value: factEl("params.trainable"),
        caption: "the patches, tables and head",
      }),
      createStatPlate({
        label: "share that moves",
        value: factEl("params.share_pct"),
        caption: "of the whole model",
      }),
    ),
    prose(`The usual way to adapt a model like this is to keep training it: let every one
      of those hundred and twenty-five million numbers shift a little towards your task.
      It works, and for a federation it is hopeless. Every project would have to upload the
      entire model every round.`),
    prose(`So the reader is frozen solid and left exactly as it arrived. What it already
      knows about English and code is taken as given. The only question is how to nudge
      its behaviour towards story points without editing it &mdash; and the answer is to
      leave it alone and add something next to it.`),
  );
}

/* ==========================================================================
   The rank idea — a square against two strips
   ========================================================================== */

function buildRank() {
  const section = block(
    "A correction, in two thin strips",
    prose(`Inside the reader are large square tables of numbers &mdash;
      <span data-fact="lora.matrix_numbers"></span> in each one, arranged
      ${WIDTH} by ${WIDTH}. To change the reader's behaviour you would ordinarily adjust
      a table like that directly, which means carrying a whole new square around.`),
    prose(`The trick is to never build the square. Instead, keep two thin strips &mdash;
      one tall and narrow, one short and wide &mdash; and multiply them together when a
      correction is needed. Multiplying a ${WIDTH}&times;8 by an 8&times;${WIDTH} gives
      back something ${WIDTH}&times;${WIDTH} shaped, so it slots in exactly where the
      square would have gone. But you only ever store the strips.`),
  );

  /* ---- the live comparison ---- */
  const rank0 = data.value("lora.rank") ?? 8;
  const sites = data.value("lora.sites") ?? 24;
  const recorded = data.value("params.lora_b");
  let rank = rank0;

  const visual = el("div", "rank-visual");
  const trained = createReadout({
    label: "numbers trained per project",
    value: 0,
    caption: "one strip, at every place a patch attaches",
  });
  const verdict = el("p", "rank-verdict");

  function paint() {
    visual.replaceChildren();

    const square = el("div", "rank-box");
    square.append(el("span", "rank-cap", "the full table"));
    const squareArt = el("div", "rank-square mat-recess");
    squareArt.setAttribute("role", "img");
    squareArt.setAttribute("aria-label",
      `A ${WIDTH} by ${WIDTH} square: ${(WIDTH * WIDTH).toLocaleString()} numbers`);
    square.append(squareArt);
    square.append(el("span", "rank-n", `${(WIDTH * WIDTH).toLocaleString()} numbers`));
    visual.append(square);

    visual.append(el("span", "rank-vs", "against"));

    const strips = el("div", "rank-box");
    strips.append(el("span", "rank-cap", "the two strips"));
    const stripArt = el("div", "rank-strips");
    // The strips are drawn at their true relative width, so the picture is the
    // arithmetic rather than an illustration of it.
    const share = rank / WIDTH;
    const tall = el("div", "rank-strip rank-tall mat-recess");
    tall.style.setProperty("--share", String(share));
    tall.title = `B: ${WIDTH} by ${rank}`;
    const wide = el("div", "rank-strip rank-wide mat-recess");
    wide.style.setProperty("--share", String(share));
    wide.title = `A: ${rank} by ${WIDTH}`;
    stripArt.setAttribute("role", "img");
    stripArt.setAttribute("aria-label",
      `Two strips of width ${rank}: ${(2 * WIDTH * rank).toLocaleString()} numbers together`);
    stripArt.append(tall, wide);
    strips.append(stripArt);
    strips.append(el("span", "rank-n", `${(2 * WIDTH * rank).toLocaleString()} numbers`));
    visual.append(strips);

    const total = sites * WIDTH * rank;
    trained.set(total);

    const pct = (100 * (2 * WIDTH * rank) / (WIDTH * WIDTH)).toFixed(2);
    const atRun = rank === rank0;
    verdict.dataset.match = String(atRun && total === recorded);
    verdict.textContent = atRun && recorded
      ? `At width ${rank} the two strips come to ${pct}% of the square, and across all `
        + `${sites} places that is ${total.toLocaleString()} trained numbers — exactly `
        + `what the run recorded.`
      : `At width ${rank} the two strips come to ${pct}% of the square, and across all `
        + `${sites} places that is ${total.toLocaleString()} trained numbers.`;
  }

  const dial = createStepper({
    label: "width of the strips",
    min: 1, max: 64, step: 1, value: rank,
    onChange: (v) => { rank = v; paint(); },
  });

  paint();

  section.append(dial.el, visual, trained.el, verdict);

  section.append(prose(`Widen the strips and they carry more &mdash; more room to express a
    correction, more to upload. Narrow them and the patch gets cheap and blunt. The run
    settled on <span data-fact="lora.rank"></span>, which is the usual starting point in
    the literature rather than something this thesis tuned.`, "note"));

  const why = createDrawer({
    label: "Why should a thin patch be enough?",
    summary: "the assumption underneath",
  });
  why.body.append(prose(`It should not be obvious that two strips can stand in for a full
    square. They cannot, in general &mdash; a square of that size holds far more
    possibilities than the strips can reach.`));
  why.body.append(prose(`The bet is that the <em>change</em> needed is much simpler than
    the thing being changed. The reader already understands English and code; adapting it
    to judge effort is a nudge, not a rebuild, and a nudge may well be expressible in very
    few directions. That is the assumption LoRA rests on, and it is an empirical claim
    rather than a proof &mdash; it holds up well in practice, which is why the technique
    spread.`, "note"));
  section.append(why.el);

  return section;
}

/* ==========================================================================
   Where they attach
   ========================================================================== */

function buildWhereTheyAttach() {
  const layers = data.value("lora.layers") ?? 12;

  const section = block(
    "Where the patches go",
    prose(`Not everywhere. The reader is a stack of
      <span data-fact="lora.layers"></span> layers, and inside each layer the patches
      attach at two places only &mdash; the parts that decide what the model pays
      attention to. That is <span data-fact="lora.sites"></span> patches in total.`),
  );

  const stack = el("div", "layer-stack");
  for (let i = layers; i >= 1; i--) {
    const layer = el("div", "layer-row mat-sub");
    layer.append(el("span", "layer-n", String(i)));
    const slots = el("span", "layer-slots");
    ["query", "value"].forEach((name) => {
      const slot = el("span", "layer-slot", name);
      slot.dataset.patched = "true";
      slots.append(slot);
    });
    layer.append(slots);
    layer.append(el("span", "layer-rest", "everything else — untouched"));
    stack.append(layer);
  }
  stack.setAttribute("role", "img");
  stack.setAttribute("aria-label",
    `${layers} layers, each with a patch on its query and value parts and nothing else changed`);
  section.append(stack);

  section.append(prose(`Those <span data-fact="lora.sites"></span> tables hold
    <span data-fact="lora.qv_numbers"></span> numbers between them. The patches replace
    the job of adjusting all of those with
    <span data-fact="params.lora_b"></span> trained ones.`, "note"));

  return section;
}

/* ==========================================================================
   What it buys
   ========================================================================== */

function buildWhatItBuys() {
  return block(
    "What that buys",
    statRow(
      createStatPlate({
        label: "one upload",
        value: factEl("comms.per_round_bytes", {
          format: (v) => (v ? `${(v / 1e6).toFixed(2)} MB` : "—"),
        }),
        caption: "per project, per round",
      }),
      createStatPlate({
        label: "the same round, whole model",
        value: factEl("comms.per_round_bytes_full", {
          format: (v) => (v ? `${Math.round(v / 1e6)} MB` : "—"),
        }),
        caption: "if nothing were frozen",
      }),
      createStatPlate({
        label: "times smaller",
        value: factEl("comms.reduction"),
        caption: "the whole point",
      }),
    ),
    prose(`A megabyte is a photograph. Five hundred megabytes, from every project, every
      round, is the difference between a design somebody might actually deploy and a
      thought experiment.`),
    prose(`That number gets weighed properly at the scales, near the end &mdash; including
      the half of the comparison this stop has not made, which is whether estimates hold
      up when you train this way. Cheap and useless is not a result.`, "note"),
  );
}
