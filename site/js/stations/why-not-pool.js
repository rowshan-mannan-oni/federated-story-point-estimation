/* ==========================================================================
   Stop 3 — Why not just pool it.

   The reader now knows the labels are noisy and team-specific. The obvious
   response is "collect everyone's issues and train on the lot", so this stop
   has to deal with that properly rather than waving at the word "privacy".

   Three things happen here, and the third is the one most explainers skip:

     1. What is actually inside an issue tracker, and why it does not leave.
     2. A side-by-side of what would travel under each arrangement.
     3. The awkward arithmetic: federated training moves FAR more bytes than
        simply posting the text once. Federation is not a bandwidth saving,
        and pretending otherwise would be the easiest lie on this whole site.

   It also states the limits of the privacy claim up front, because the
   thesis does, and because an examiner asks this question first.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { createSegmented } from "../ui/controls.js";
import { factText } from "../ui/provenance.js";
import * as data from "../core/data.js";

export async function mount(host, ctx) {
  const examples = await data.examples();
  const issues = examples?.issues ?? [];

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "Why not just pool it",
    standfirst: `If teams label work differently and a model needs a great many examples,
      the obvious answer is to put every project's issues in one place and train on all of
      them. That would work. It is also the one thing nobody involved will agree to
      &mdash; and the reason has nothing to do with machine learning.`,
  });

  /* ---- 1. what is in a tracker ----------------------------------------- */
  card.append(block(
    "What is actually in an issue tracker",
    prose(`An issue is not a tidy row of features. It is whatever an engineer typed while
      annoyed at four in the afternoon: the customer who hit the bug, the internal service
      that fell over, the workaround nobody wants documented, the security hole that is
      still open, next quarter's plans in the comments of a ticket nobody closed.`),
    prose(`That is why "just send us your issue trackers" is a conversation that ends
      quickly. The obstacle is not technical and it is not solvable with a better export
      script. It is that the text itself is the sensitive thing.`),
    honestyAboutThisCorpus(),
  ));

  /* ---- 2. the switch --------------------------------------------------- */
  card.append(buildComparison(issues));

  /* ---- 3. the awkward arithmetic --------------------------------------- */
  card.append(block(
    "The bytes tell an awkward story",
    prose(`It is tempting to say federated training is the efficient option. It is not,
      and this walk-through is not going to pretend otherwise.`),
    statRow(
      createStatPlate({
        label: "all the issue text, once",
        value: factText("corpus.text_mb"),
        caption: "every title and description in the study",
      }),
      createStatPlate({
        label: "one project, one round",
        value: mb(data.value("comms.per_round_bytes")),
        caption: "the trained patch, uploaded",
      }),
      createStatPlate({
        label: "the whole federated run",
        value: gb(data.value("comms.total_bytes")),
        caption: `${factText("split.clients")} projects × ${factText("run.rounds")} rounds`,
      }),
    ),
    prose(ratioSentence()),
    prose(`So federation does not save traffic. What it changes is <em>what is in the
      traffic</em> and <em>where the data lives</em>. Nobody ever holds a copy of anybody
      else's issues. The saving that this thesis does claim is a different comparison
      entirely &mdash; a federated run that ships small patches against a federated run
      that ships the whole model every round &mdash; and that one is measured at the
      scales, near the end.`, "note"),
  ));

  /* ---- 4. the limits of the claim -------------------------------------- */
  card.append(buildPrivacyScope());

  card.append(handover({
    question: `So every project keeps its own issues, and only numbers travel. Before
      going any further into how that works: what <em>is</em> this data? Nineteen
      trackers, and some of them are stranger than you would expect.`,
    cta: "The archive",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   The side-by-side
   ========================================================================== */

function buildComparison(issues) {
  const panel = el("div", "leaves-panel mat-sub");

  const choice = createSegmented({
    label: "What leaves the building",
    options: [
      { value: "pool", label: "Pool it centrally" },
      { value: "fed", label: "Train federated" },
    ],
    value: "pool",
    onChange: paint,
  });

  function paint(mode) {
    panel.replaceChildren();
    if (mode === "pool") {
      panel.dataset.mode = "pool";
      panel.append(el("p", "leaves-head", "Every issue, in full, to one central machine"));

      const sample = el("div", "leaves-sample");
      issues.slice(0, 3).forEach((issue) => {
        const row = el("article", "leaves-issue mat-recess");
        row.append(el("p", "leaves-meta",
          `${prettyName(issue.project)} · ${issue.type} · ${issue.priority}`));
        row.append(el("p", "leaves-title", issue.title));
        row.append(el("p", "leaves-text", issue.description));
        sample.append(row);
      });
      panel.append(sample);

      panel.append(el("p", "leaves-foot",
        `…and ${Number(data.value("corpus.rows_clean") - 3).toLocaleString()} more like it. ` +
        `${data.text("corpus.text_mb")} of other people's words, readable by whoever holds the server.`));
    } else {
      panel.dataset.mode = "fed";
      panel.append(el("p", "leaves-head", "One bundle of trained numbers, per project, per round"));

      const parts = [
        ["the trained half of the patches", data.value("params.lora_b")],
        ["the type and priority tables", data.value("params.embeddings")],
        ["the part that picks a number", data.value("params.head")],
      ];
      const list = el("ul", "payload-list");
      const total = data.value("params.trainable") || 1;
      parts.forEach(([label, count]) => {
        const item = el("li", "payload-row");
        item.append(el("span", "payload-count", Number(count).toLocaleString()));
        item.append(el("span", "payload-label", label));
        const bar = el("span", "payload-bar");
        bar.style.setProperty("--share", String((count || 0) / total));
        item.append(bar);
        list.append(item);
      });
      panel.append(list);

      panel.append(el("p", "leaves-foot",
        `${Number(data.value("params.trainable")).toLocaleString()} numbers in total — ` +
        `${mb(data.value("comms.per_round_bytes"))}. Not one word of anybody's text, ` +
        `and no way to read an issue back out of it by simply looking.`));
    }
  }

  paint("pool");

  return block(
    "Try it — what leaves the building",
    prose(`Two arrangements, the same goal: one model that has learned from all nineteen
      projects. Switch between them and look at what physically travels.`),
    choice.el,
    panel,
  );
}

/* ==========================================================================
   Honesty blocks
   ========================================================================== */

function honestyAboutThisCorpus() {
  const drawer = createDrawer({
    label: "But these nineteen projects are public",
    summary: "the obvious objection, answered first",
  });
  drawer.body.append(prose(`They are, and it is worth saying before anyone catches it.
    Every tracker in this study is an open-source project whose issues anyone can read.
    Nothing here is being protected from anybody.`));
  drawer.body.append(prose(`The public corpus is a stand-in. It exists so the experiment
    can be repeated and checked by someone else, which a set of private trackers could
    never allow. The method is built for the private case; the measurements are taken on
    the public one. That is a real limitation of the study rather than a detail, and it
    belongs in plain sight.`, "note"));
  return drawer.el;
}

function buildPrivacyScope() {
  const section = block(
    "How private is this, really?",
    prose(`The claim this thesis makes is narrow, and worth stating exactly: <strong>the
      raw issue text never leaves the project that owns it</strong>. That is all. It is
      not a guarantee that nothing about the data can be inferred.`),
  );

  const list = el("ul", "caveat-list");
  [
    [`Weights can leak`,
     `Numbers trained on data carry traces of that data. There are published
      <span data-term="gradient inversion">attacks</span> that reconstruct fragments of
      training examples from exactly this kind of upload. This project does not defend
      against them and does not measure them.`],
    [`No formal guarantee`,
     `<span data-term="differential privacy">Differential privacy</span> and
      <span data-term="secure aggregation">secure aggregation</span> are the standard
      answers to that, and neither is used here. Both are named as future work.`],
    [`One place where data is pooled`,
     `The model is warmed up on a single large project before federated training starts.
      That project's data sits on the training machine. It is the one exception to
      everything on this page, and there is a stop about it later.`],
    [`The server is trusted`,
     `It sees each project's upload individually before averaging them. A less trusting
      arrangement is possible; this is not it.`],
  ].forEach(([title, body]) => {
    const item = el("li", "caveat mat-recess");
    item.append(el("span", "caveat-title", title));
    const text = el("p", "caveat-body");
    text.innerHTML = body;
    item.append(text);
    list.append(item);
  });

  section.append(list);
  section.append(prose(`None of this sinks the idea. It sets its size: this is a design
    that removes the need to hand over your text, not a proof that your text is
    unrecoverable. Saying the stronger thing would be easier and would be wrong.`,
    "note"));
  return section;
}

/* ==========================================================================
   Helpers
   ========================================================================== */

function ratioSentence() {
  const text = data.value("corpus.text_bytes");
  const total = data.value("comms.total_bytes");
  if (!text || !total) {
    return `The comparison needs both the corpus size and a finished run; one of them is
      missing on this machine.`;
  }
  const ratio = Math.round(total / text);
  return `Read those again. Posting every issue once would move
    <span data-fact="corpus.text_mb"></span>. Training the same model federated moved
    ${gb(total)} — about <strong>${ratio} times more traffic</strong>, to avoid moving
    the text at all.`;
}

function mb(bytes) {
  return bytes ? `${(bytes / 1e6).toFixed(2)} MB` : "—";
}

function gb(bytes) {
  return bytes ? `${(bytes / 1e9).toFixed(2)} GB` : "—";
}

function prettyName(name) {
  return String(name).replace(/_/g, " ");
}
