/* ==========================================================================
   Stop 1 — The front door.

   The job of this stop is to make someone want to walk the other twenty-four.
   So it does four things and stops: says what the project is, states the three
   questions it exists to answer, warns honestly about what the numbers will
   look like, and shows the shape of the road ahead.

   Numbers are written as <span data-fact="…"> and terms as <span data-term="…">;
   app.js swaps in the real figures after mounting.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, pathOverview, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { factText } from "../ui/provenance.js";

/* The three questions, in plain words and in the thesis's own words. Keeping
   both means a curious reader gets the idea and an examiner gets the exact
   claim, without the site having to choose between them. */
const QUESTIONS = [
  {
    n: "RQ1",
    plain: "Is training together worth doing at all?",
    what: `Three ways to build the model, judged on exactly the same issues:
           each project training alone on its own data, everything pooled into one
           place as if privacy did not matter, and the federated middle ground where
           projects train separately and share only what they learned.`,
    exact: `How does FedSP-PEFT compare to centralized pooling and to per-project
            local training for story point estimation?`,
    where: "Answered at the line-up, once you have seen how it all works.",
  },
  {
    n: "RQ2",
    plain: "Should every project use the same final judgement?",
    what: `A story point is whatever a team says it is: the same work can be an 8
           here and a 3 there. So perhaps projects should share their understanding
           of the words, while each keeps its own sense of scale.`,
    exact: `Does story point estimation improve when each project keeps its own
            prediction head, instead of federating the head along with the
            representation?`,
    where: "Set up at the personal-head stop, settled at the line-up.",
  },
  {
    n: "RQ3",
    plain: "How little can we get away with sending?",
    what: `Sending a whole trained model every round is enormous. Training a small
           patch instead is cheap — but only worth it if the estimates hold up.
           This question weighs both halves: the quality and the bytes.`,
    exact: `How does federating only LoRA adapters compare with fully fine-tuning and
            federating the whole encoder, in estimation quality and in communication
            cost?`,
    where: "Set up at the patches stop, weighed at the scales.",
  },
];

export function mount(el_, ctx) {
  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "Guessing effort, without sharing the evidence",
    standfirst: `Software teams estimate how much work a task will take, in
      <span data-term="story point">story points</span>. They are not very good at it.
      This thesis asks whether a model can learn to make that estimate from the words
      of the task alone &mdash; and whether separate projects can teach that model
      <span data-term="federated learning">together</span>, without any of them
      handing over their data.`,
  });

  /* ---- 1. What this is, in six sentences ------------------------------- */
  const six = el("ol", "six-list");
  [
    `Software teams guess how much work each task will be, using a small fixed set
     of numbers &mdash; 1, 2, 3, 5, 8.`,
    `Those guesses decide what fits in the next few weeks, and they are often wrong.`,
    `This project asks whether a model can make that guess from the text of the task:
     its title and description, nothing else.`,
    `A model needs a great many examples, and the examples live in
     <span data-term="issue">issue trackers</span> belonging to companies that will
     not hand them over.`,
    `So each project trains on its own issues and uploads only a small bundle of
     learned numbers &mdash; <span data-fact="params.trainable"></span> of them,
     <span data-fact="params.share_pct"></span> of the model &mdash; never the text.`,
    `A server averages those bundles into one shared model, and this entire
     walk-through is about whether that turns out to be worth doing.`,
  ].forEach((sentence) => {
    const item = document.createElement("li");
    item.innerHTML = sentence;
    six.append(item);
  });

  card.append(block("What this is, in six sentences", six));

  /* ---- 2. The evidence, in three numbers ------------------------------- */
  card.append(block(
    "What it is built on",
    statRow(
      createStatPlate({
        label: "projects",
        value: factText("corpus.projects"),
        caption: "public issue trackers",
      }),
      createStatPlate({
        label: "issues",
        value: factText("corpus.rows_raw"),
        caption: `${factText("corpus.first_issue")} to ${factText("corpus.last_issue")}`,
      }),
      createStatPlate({
        label: "train together",
        value: factText("split.clients"),
        caption: "one is held back to start the model off",
      }),
    ),
    prose(`Every number on this site can be questioned. Hover one, or tab to it, and it
      will name the file it came from and how it was worked out. Underlined words are
      explained the same way. Nothing here is typed in by hand:
      <a href="facts.html">see the whole list</a>.`),
  ));

  /* ---- 3. The three questions ------------------------------------------ */
  const questions = el("div", "rq-list");
  QUESTIONS.forEach((question) => {
    const drawer = createDrawer({
      label: `${question.n} — ${question.plain}`,
      summary: "",
    });

    drawer.body.append(prose(question.what, "rq-what"));

    const exact = el("div", "rq-exact mat-recess");
    exact.append(el("span", "label", "As the thesis puts it"));
    exact.append(el("p", null, collapse(question.exact)));
    drawer.body.append(exact);

    drawer.body.append(el("p", "rq-where", question.where));
    questions.append(drawer.el);
  });

  card.append(block(
    "The three questions it sets out to answer",
    prose(`Open any of them. Each one is written twice: in plain words, and in the exact
      wording the thesis uses &mdash; so you can check that the plain version has not
      quietly changed the meaning.`),
    questions,
  ));

  /* ---- 4. The honest warning ------------------------------------------- */
  card.append(block(
    "Before you go any further",
    prose(`Estimating story points from text is <em>hard</em>, and not only here. A
      well-known study re-ran the leading deep-learning methods for this task and found
      they barely beat trivial ones. You will see that here too: at one point a
      twenty-year-old method with no understanding of language at all beats the model
      on one of the three scores.`),
    prose(`So this walk-through is not building to a triumphant number. The interesting
      question is a comparison &mdash; what does keeping your data to yourself actually
      cost you? &mdash; and that comparison stays meaningful even when every method
      involved finds the task difficult. Where the numbers are unflattering, they are
      shown unflattering.`, "note"),
  ));

  /* ---- 5. The road ahead ----------------------------------------------- */
  card.append(block(
    "The road ahead",
    prose(`Twenty-five stops in seven parts. Each one answers the question the stop
      before it ran into, so the intended way through is simply forward &mdash; but you
      can jump to any part now if you would rather see the shape first.`),
    pathOverview(ctx.goTo),
  ));

  /* ---- 6. Getting around, tucked away for those who want it ------------ */
  const how = createDrawer({ label: "How to get around", summary: "keys, links, the map" });
  const list = el("ul", "how");
  [
    ["<kbd>→</kbd> or Next", "Move on to the next stop"],
    ["<kbd>M</kbd>", "See the whole path at once"],
    ["The rail below", "Jump anywhere; solid ticks are built, hollow ones are not yet"],
    ["The address bar", "Every stop has its own link you can share or bookmark"],
  ].forEach(([key, value]) => {
    const item = el("li", "how-item mat-sub");
    const k = el("span", "k");
    k.innerHTML = key;
    item.append(k, el("span", "v", value));
    list.append(item);
  });
  how.body.append(list);
  how.body.append(prose(`Reference pages, outside the walk-through:
    <a href="facts.html">every number the site knows</a> ·
    <a href="parts.html">the parts these stops are built from</a>`, "note"));
  card.append(how.el);

  /* ---- 7. Onward ------------------------------------------------------- */
  card.append(handover({
    question: `First, the thing being predicted. Why do teams put numbers on work at
      all &mdash; and why does the same task get a different number in a different team?`,
    cta: "The poker table",
    next: ctx.next,
  }));

  el_.append(card);
}

/** Collapse the line breaks in the quoted wording without changing a word. */
function collapse(text) {
  return text.replace(/\s+/g, " ").trim();
}
