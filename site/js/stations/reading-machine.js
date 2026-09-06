/* ==========================================================================
   Stop 8 — The reading machine.

   The data is clean and divided. This stop is the first half of the model:
   the part that turns an issue into numbers.

   Two honesty problems shape it.

   First, the tokeniser. Splitting text the way CodeBERT actually splits it
   needs the model's own vocabulary, which this machine has but the website
   does not ship. So the split shown here is an illustration and is stamped as
   one — the shape of the idea is right, the exact pieces are not.

   Second, the length figures are measured in WORDS, because counting tokens
   needs that same tokeniser. Tokens outnumber words, so every truncation
   figure here is a lower bound. Said plainly rather than quietly rounded.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { createHistogram } from "../viz/histogram.js";
import { createDistStrip, createDistLegend } from "../viz/dist-strip.js";
import { factEl, factText } from "../ui/provenance.js";
import { banner, stamp } from "../ui/stamp.js";
import * as data from "../core/data.js";

export async function mount(host, ctx) {
  const [examples, split] = await Promise.all([data.examples(), data.split()]);
  const issues = examples?.issues ?? [];
  const lengths = split?.lengths;

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "The reading machine",
    standfirst: `A model cannot read. What it can do is turn text into a long row of
      numbers and work on those. This stop is that conversion &mdash; and the one place
      where information is thrown away for a reason that has nothing to do with the data:
      the machine has a fixed appetite, and long issues do not fit.`,
  });

  card.append(buildChain(issues));
  card.append(buildLengths(lengths));
  card.append(buildTruncation(lengths));
  card.append(buildWhyCodeBert());

  card.append(handover({
    question: `That is the words dealt with. But an issue carries two more facts that are
      not words at all &mdash; what kind of task it is, and how urgent someone marked it.
      Those go in too, and they take a different route.`,
    cta: "Type and priority",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   Text in, one row of numbers out
   ========================================================================== */

function buildChain(issues) {
  const example = issues.find((i) => i.description && i.title.length < 70) ?? issues[0];

  const section = block(
    "From a sentence to one row of numbers",
    prose(`The title and the description are joined into a single piece of text with a
      separator between them, so the model can tell where one ends and the other begins.
      That text is then chopped into <span data-term="token">tokens</span> &mdash; roughly
      words, but common word-endings and rare words get split further.`),
  );

  if (!example) {
    section.append(prose(`No example issues have been generated yet.`, "note"));
    return section;
  }

  const joined = `${example.title} [SEP] ${example.description}`;

  const steps = el("ol", "chain");
  [
    ["The issue, as filed",
     `<span class="chain-text">${escapeHtml(example.title)}</span>`],
    ["Title and description joined",
     `<span class="chain-text">${escapeHtml(truncate(joined, 220))}</span>`],
    ["Chopped into pieces", null],
    ["Each piece becomes a row of numbers",
     `Every token is looked up and passed through the
      <span data-term="encoder">encoder</span>, which returns 768 numbers for each one
      &mdash; a description of that piece <em>in the context of the whole issue</em>.`],
    ["Averaged into one row",
     `Those per-token rows are averaged into a single row of 768 numbers standing for the
      whole issue. Padding is excluded from the average, so a short issue is not diluted
      by the empty space after it.`],
  ].forEach(([title, body]) => {
    const item = el("li", "chain-step");
    item.append(el("span", "chain-title", title));
    if (body) {
      const text = el("div", "chain-body");
      text.innerHTML = body;
      item.append(text);
    }
    steps.append(item);
  });

  /* The illustrative split goes inside step 3. */
  const tokenBox = el("div", "chain-body");
  const pieces = illustrativeSplit(truncate(joined, 120));
  const row = el("div", "token-row");
  pieces.forEach((piece) => {
    const chip = el("span", "token", piece);
    if (piece === "[SEP]" || piece === "[CODE]" || piece === "[URL]" || piece === "[ISSUE_REF]") {
      chip.dataset.special = "true";
    }
    row.append(chip);
  });
  tokenBox.append(row);
  tokenBox.append(banner("simulated",
    "This split is an illustration. The real tokeniser uses CodeBERT's own vocabulary, " +
    "which this website does not carry — so the shape is right and the exact pieces are not."));
  steps.children[2].append(tokenBox);

  section.append(steps);
  return section;
}

/* ==========================================================================
   How long issues actually are
   ========================================================================== */

function buildLengths(lengths) {
  const section = block(
    "How long is an issue?",
    statRow(
      createStatPlate({
        label: "words, typical issue",
        value: factEl("text.words_p50"),
        caption: "half are shorter than this",
      }),
      createStatPlate({
        label: "words, long issue",
        value: factEl("text.words_p90"),
        caption: "nine in ten are shorter",
      }),
      createStatPlate({
        label: "run past 256 words",
        value: factEl("text.over_256_pct"),
        caption: "where the model stops reading",
      }),
    ),
  );

  if (lengths?.buckets?.length) {
    const bars = lengths.buckets
      .filter((b) => b.n > 0)
      .map((b) => ({
        label: b.label,
        value: b.n,
        note: b.mean_sp == null ? "" : b.mean_sp.toFixed(2),
        mark: b.label.startsWith("256") || b.label.startsWith("384") || b.label === "512+",
      }));
    section.append(createHistogram({
      bars,
      caption: "Training issues by length in words",
      noteLabel: "Number under each bar: the average story point of the issues in it. " +
                 "Marked bars are past the model's limit.",
    }).el);
  }

  section.append(prose(`Most issues are short. The typical one is
    <span data-fact="text.words_p50"></span> words &mdash; two or three sentences &mdash;
    and the long tail thins out quickly.`));

  const caveat = createDrawer({
    label: "Careful: these are words, not tokens",
    summary: "the real figure is worse",
  });
  caveat.body.append(prose(`The model's limit is 256 <em>tokens</em>, and tokens are
    smaller than words: punctuation splits off, and anything unusual &mdash; a class name,
    a file path, a stack frame &mdash; is broken into several pieces. Issue text is full
    of exactly those.`));
  caveat.body.append(prose(`Counting tokens properly needs CodeBERT's own vocabulary,
    which this machine has but the website does not load. So every length figure on this
    page is a <strong>lower bound</strong>: the real share of issues that get cut short is
    higher than <span data-fact="text.over_256_pct"></span>, probably by a third to a half
    again. The project's own notes put the ninetieth percentile at about 164 tokens
    against the <span data-fact="text.words_p90"></span> words measured here, which is
    roughly that ratio.`, "note"));
  section.append(caveat.el);

  return section;
}

/* ==========================================================================
   Does the cutting matter?
   ========================================================================== */

function buildTruncation(lengths) {
  const section = block(
    "Does cutting them short matter?",
    prose(`There is a worry worth taking seriously. If the issues that get cut off are
      also the <em>big</em> ones, then truncation is not a neutral loss &mdash; it would be
      removing evidence exactly where the model most needs it. That is checkable, so here
      it is checked.`),
    statRow(
      createStatPlate({
        label: "average points, long issues",
        value: factEl("length.mean_sp_long"),
        caption: "past 256 words",
      }),
      createStatPlate({
        label: "average points, the rest",
        value: factEl("length.mean_sp_short"),
        caption: "256 words or fewer",
      }),
      createStatPlate({
        label: "how strongly length tracks size",
        value: factEl("length.correlation"),
        caption: "correlation, where 1.00 would be perfect",
      }),
    ),
    prose(`So the worry is real, and small. Long issues do carry more points on average
      &mdash; <span data-fact="length.mean_sp_long"></span> against
      <span data-fact="length.mean_sp_short"></span> &mdash; so the tail being cut off is
      slightly weighted towards the harder work. But the link between length and size is
      weak: a correlation of <span data-fact="length.correlation"></span> means length
      barely predicts effort at all.`),
    prose(`And it affects <span data-fact="length.long_issues"></span> issues in the
      training data, out of <span data-fact="split.train"></span>. A real effect, in the
      direction that would flatter the model, small enough that it explains very little.
      Whether raising the limit to 512 would earn its cost is one of the questions the
      thesis lists as optional and has not yet run.`, "note"),
  );

  if (lengths?.buckets?.length) {
    const shortest = lengths.buckets.find((b) => b.label === "<16");
    const longest = lengths.buckets.filter((b) => b.n > 0).slice(-1)[0];
    if (shortest?.n && longest?.n) {
      const set = el("div", "dist-set");
      set.append(createDistLegend());
      set.append(createDistStrip({
        title: "The shortest issues (under 16 words)",
        counts: shortest.counts,
        note: `${shortest.n.toLocaleString()} issues · ${shortest.mean_sp} average`,
      }).el);
      set.append(createDistStrip({
        title: `The longest issues (${longest.label} words)`,
        counts: longest.counts,
        note: `${longest.n.toLocaleString()} issues · ${longest.mean_sp} average`,
      }).el);
      section.append(set);
      section.append(prose(`The clearest signal is at the other end: issues of under
        sixteen words really are small tasks. Somebody who writes one line about a job is
        usually describing one line of work.`, "note"));
    }
  }

  return section;
}

/* ==========================================================================
   Why this particular reader
   ========================================================================== */

function buildWhyCodeBert() {
  const section = block(
    "Why this particular reader",
    prose(`The encoder used here is
      <span data-term="codebert">CodeBERT</span> &mdash; a model trained on ordinary English
      <em>and</em> source code. That is not a fashionable choice, it is a practical one:
      look back at the cleaning bench and notice what issue text is actually made of.
      Class names, file paths, stack frames, half-sentences with a method name in the
      middle.`),
    prose(`A model trained only on prose treats <code>NullPointerException</code> as a
      strange unknown word. One trained on code has seen thousands of them. The same
      argument is why nothing is lower-cased on the way in.`, "note"),
  );

  const settings = el("dl", "settings-list");
  [
    ["Encoder", factText("run.model")],
    ["Input limit", `${factText("run.max_length")} tokens`],
    ["Numbers per token", "768"],
    ["How they are combined", "averaged, ignoring padding"],
  ].forEach(([term, value]) => {
    settings.append(el("dt", null, term));
    settings.append(el("dd", null, value));
  });
  section.append(settings);

  const frozen = createDrawer({
    label: "And almost none of it is trained",
    summary: "the important part comes next",
  });
  frozen.body.append(prose(`One detail to carry forward: this reader is
    <strong>frozen</strong>. Its 125 million numbers are left exactly as they arrived, and
    training never changes them.`));
  frozen.body.append(prose(`That sounds like it defeats the purpose &mdash; and it is the
    single decision that makes this whole design possible. Two stops from here it becomes
    the difference between sending a megabyte and sending five hundred.`, "note"));
  section.append(frozen.el);

  return section;
}

/* ==========================================================================
   Helpers
   ========================================================================== */

/**
 * A rough, honest-to-look-at split. NOT the real tokeniser: it separates
 * punctuation and breaks long words, which is the visible behaviour, without
 * pretending to reproduce CodeBERT's vocabulary. Always shown stamped.
 */
function illustrativeSplit(text) {
  const rough = String(text).match(/\[[A-Z_]+\]|[A-Za-z]+|\d+|[^\sA-Za-z\d]/g) ?? [];
  const out = [];
  for (const piece of rough) {
    if (/^\[[A-Z_]+\]$/.test(piece) || piece.length <= 6) { out.push(piece); continue; }
    // Long or camel-cased words get broken up, which is what really happens.
    const parts = piece.split(/(?=[A-Z])/).filter(Boolean);
    if (parts.length > 1) { out.push(...parts); continue; }
    for (let i = 0; i < piece.length; i += 5) {
      out.push((i ? "##" : "") + piece.slice(i, i + 5));
    }
  }
  return out.slice(0, 60);
}

function truncate(text, limit) {
  const value = String(text);
  return value.length <= limit ? value : `${value.slice(0, limit)}…`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
