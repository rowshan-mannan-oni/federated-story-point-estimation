/* ==========================================================================
   Stop 2 — The poker table.

   Two things have to land here, and the second one matters more.

   First: estimating effort from a written task is genuinely hard. The way to
   show that is not to say it, but to let the reader try it on real issues and
   watch their own accuracy.

   Second, and this is the hinge of the whole thesis: the five numbers do not
   mean the same thing in different teams. That is measured here, not claimed —
   one project calls 98% of its work small, another calls 10% of it small.
   Every design decision later in the walk-through is a response to this.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { createDistStrip, createDistLegend } from "../viz/dist-strip.js";
import { factEl, factText } from "../ui/provenance.js";
import * as data from "../core/data.js";

const DECK = [1, 2, 3, 5, 8];

export async function mount(host, ctx) {
  const [examples, calibration] = await Promise.all([
    data.examples(),
    data.calibration(),
  ]);
  const issues = examples?.issues ?? [];

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "The poker table",
    standfirst: `Before any work starts, a team has to say how big each task is. They do
      it by looking at the words in a tracker and picking a number from a small fixed
      deck. This stop is that moment &mdash; and the reason the number that comes out of
      it is so slippery.`,
  });

  /* ---- how the guess is actually made ---------------------------------- */
  card.append(block(
    "How the number gets made",
    prose(`Most teams do some version of the same ritual. Everyone reads the task,
      everyone privately picks a card from the deck &mdash; 1, 2, 3, 5, 8 &mdash; and
      then everyone turns their card over at once. If the cards disagree, the highest
      and the lowest argue their case, and the team picks again until it settles.`),
    prose(`The deck is deliberately gappy. There is no 4 and no 6, because arguing over
      the difference between a 5 and a 6 costs more time than it saves. And the numbers
      are not hours: a <span data-term="story point">story point</span> is whatever this
      particular team has come to mean by it.`),
  ));

  /* ---- the interactive table ------------------------------------------- */
  if (issues.length) {
    card.append(buildTable(issues));
  } else {
    card.append(block("Try it",
      prose(`The example issues have not been generated yet. Run
        <code>python site/tools/extract_facts.py</code> to fill this in.`)));
  }

  /* ---- the deck is a scale --------------------------------------------- */
  card.append(block(
    "The deck is a scale, not a set of names",
    prose(`Notice what a mistake feels like. Saying 3 when the answer was 2 is a near
      miss. Saying 8 when the answer was 1 is a different kind of wrong altogether. The
      five numbers have an order, and the distance between them means something.`),
    prose(`That sounds obvious, and most machine-learning setups throw it away: they
      treat the five values as five unrelated names, so being wildly out costs exactly
      what being nearly right costs. Holding on to the order turns out to matter, and
      there is a stop later that does nothing else.`, "note"),
  ));

  /* ---- the punchline: calibration -------------------------------------- */
  card.append(buildCalibration(calibration));

  card.append(handover({
    question: `So the labels are noisy <em>and</em> they mean different things in
      different teams. The obvious fix is to gather everyone's issues into one big pile
      and train on that. Why can't we?`,
    cta: "Why not just pool it",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   The table: deal a real issue, place a card, see what the team actually said
   ========================================================================== */

function buildTable(issues) {
  const state = { order: shuffle(issues.length), at: 0, guesses: [], answered: false };

  const wrap = el("div", "poker");

  /* the issue, with its project hidden until after the guess */
  const issueCard = el("article", "poker-issue mat-sub");
  const meta = el("p", "poker-meta");
  const title = el("h3", "poker-title");
  const body = el("p", "poker-body");
  issueCard.append(meta, title, body);

  /* the deck */
  const deck = el("div", "poker-deck");
  deck.setAttribute("role", "group");
  deck.setAttribute("aria-label", "Your estimate");
  const cards = DECK.map((value) => {
    const button = el("button", "poker-card mat-panel");
    button.type = "button";
    button.dataset.value = String(value);
    button.append(el("span", "poker-card-value", String(value)));
    button.setAttribute("aria-label", `Estimate ${value} points`);
    button.addEventListener("click", () => answer(value));
    deck.append(button);
    return button;
  });

  /* the reveal */
  const reveal = el("div", "poker-reveal mat-recess");
  reveal.hidden = true;

  /* the running tally */
  const tally = el("p", "poker-tally");
  tally.setAttribute("role", "status");
  tally.setAttribute("aria-live", "polite");

  const nextButton = el("button", "ctrl poker-next");
  nextButton.type = "button";
  nextButton.append(el("span", "ctrl-label", "Deal another"));
  nextButton.addEventListener("click", deal);

  const controls = el("div", "poker-controls");
  controls.append(nextButton, tally);

  wrap.append(issueCard, deck, reveal, controls);

  function current() {
    return issues[state.order[state.at % state.order.length]];
  }

  function deal() {
    if (state.answered) state.at += 1;
    state.answered = false;
    const issue = current();

    meta.textContent = `${issue.type} · ${issue.priority} priority`;
    title.textContent = issue.title;
    body.textContent = issue.description || "(no description — the title is all there is)";

    reveal.hidden = true;
    reveal.replaceChildren();
    cards.forEach((c) => {
      c.disabled = false;
      c.removeAttribute("data-state");
    });
    nextButton.disabled = true;
  }

  function answer(value) {
    if (state.answered) return;
    state.answered = true;

    const issue = current();
    const truth = issue.story_point;
    const steps = Math.abs(DECK.indexOf(value) - DECK.indexOf(truth));
    state.guesses.push({ guess: value, truth, steps, gap: Math.abs(value - truth) });

    cards.forEach((c) => {
      c.disabled = true;
      const v = Number(c.dataset.value);
      if (v === truth) c.dataset.state = "truth";
      else if (v === value) c.dataset.state = "guess";
    });

    reveal.hidden = false;
    reveal.replaceChildren();
    reveal.append(el("p", "poker-verdict",
      value === truth ? "Exactly what the team said."
        : steps === 1 ? "One step away."
          : `${steps} steps away.`));
    reveal.append(el("p", "poker-answer",
      `${prettyName(issue.project)} recorded this as a ${truth}.`));

    nextButton.disabled = false;
    paintTally();
  }

  function paintTally() {
    const n = state.guesses.length;
    if (!n) { tally.textContent = ""; return; }
    const exact = state.guesses.filter((g) => g.steps === 0).length;
    const near = state.guesses.filter((g) => g.steps <= 1).length;
    const meanGap = state.guesses.reduce((sum, g) => sum + g.gap, 0) / n;
    tally.textContent =
      `${n} judged · ${exact} exact · ${near} within one step · ` +
      `${meanGap.toFixed(2)} points out on average`;
  }

  deal();

  const section = block(
    "Try it — real issues, real answers",
    prose(`Twenty-five issues, taken from the training data of the projects in this
      study. Read one, pick a card, and see what the team that wrote it actually
      recorded. The project's name stays hidden until you have committed &mdash; which
      turns out to matter more than you would expect.`),
    wrap,
  );

  const caveat = createDrawer({
    label: "Is this a fair contest?",
    summary: "no — and the reason is the point",
  });
  caveat.body.append(prose(`You are guessing without knowing which team wrote the issue.
    That is exactly the handicap a single shared model would carry: it would have to
    settle on one meaning of &ldquo;5&rdquo; for everybody. Letting each project keep its
    own final judgement is this thesis's answer to that, and it is the whole of its
    second question.`));
  caveat.body.append(prose(`These issues were also picked to be short and readable, so
    they are easier than the average case. And they come from the training split on
    purpose: every number measured later in this walk-through is measured on issues
    that are held back, and putting those on a web page is a good way to quietly ruin
    them.`, "note"));
  section.append(caveat.el);

  return section;
}

/* ==========================================================================
   The calibration evidence — the hinge of the thesis
   ========================================================================== */

function buildCalibration(calibration) {
  const rows = calibration?.per_project ?? [];

  const section = block(
    "One team's 8 is another team's 3",
    prose(`Here is the difficulty that shapes everything after this stop. Every project
      in this study uses the same five numbers &mdash; and they use them to mean
      completely different things.`),
    statRow(
      createStatPlate({
        label: "average points, lowest project",
        value: factEl("calibration.lowest_mean"),
        caption: prettyName(factText("calibration.lowest_project")),
      }),
      createStatPlate({
        label: "average points, highest project",
        value: factEl("calibration.highest_mean"),
        caption: prettyName(factText("calibration.highest_project")),
      }),
      createStatPlate({
        label: "apart",
        value: factEl("calibration.mean_spread"),
        caption: "on the same five-card deck",
      }),
    ),
  );

  if (rows.length) {
    const sorted = [...rows].sort((a, b) => a.mean - b.mean);
    const show = [
      sorted[0],
      sorted[Math.floor(sorted.length / 3)],
      sorted[Math.floor((2 * sorted.length) / 3)],
      sorted[sorted.length - 1],
    ].filter(Boolean);

    const set = el("div", "dist-set");
    set.append(createDistLegend());
    show.forEach((row) => {
      set.append(createDistStrip({
        title: prettyName(row.name),
        counts: row.counts,
        note: `${row.n.toLocaleString()} issues · ${row.mean.toFixed(2)} average`,
      }).el);
    });
    section.append(set);

    const all = createDrawer({
      label: `All ${rows.length} projects`,
      summary: "the same picture everywhere",
    });
    const everything = el("div", "dist-set");
    sorted.forEach((row) => {
      everything.append(createDistStrip({
        title: prettyName(row.name),
        counts: row.counts,
        note: `${row.mean.toFixed(2)} average`,
        compact: true,
      }).el);
    });
    all.body.append(everything);
    section.append(all.el);
  }

  section.append(prose(`In <span data-fact="calibration.small_share_max_project"></span>,
    <span data-fact="calibration.small_share_max"></span> of all work is a 1 or a 2. In
    <span data-fact="calibration.small_share_min_project"></span>, only
    <span data-fact="calibration.small_share_min"></span> is. Nobody here is wrong: each
    team is consistent with itself, which is the only thing a story point ever promises
    to be.`));

  section.append(prose(`But it means a model trained on everyone's issues at once is
    being taught contradictions &mdash; the same kind of words, labelled 2 in one place
    and 8 in another. Keep this in mind. It explains a result later that looks at first
    glance like a bug: pooling all the data into one pile makes the estimates
    <em>worse</em>.`, "note"));

  return section;
}

/** Project folders are named with underscores; sentences are not. */
function prettyName(name) {
  return String(name).replace(/_/g, " ");
}

/** A fixed shuffle, so the order of the issues is the same on every visit. */
function shuffle(length) {
  const order = Array.from({ length }, (_, i) => i);
  let seed = 20260819;
  for (let i = order.length - 1; i > 0; i--) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    const j = seed % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}
