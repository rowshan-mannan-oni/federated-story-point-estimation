/* ==========================================================================
   Stop 6 — Links, numbers, leftovers.

   The previous stop dealt with the markup. This one finishes the job and then
   asks the more interesting question: how does anyone know the cleaning
   actually happened?

   The answer is an alarm in the loading code that refuses uncleaned data. It
   is not a hypothetical: the CSV files sitting on the machine that built this
   website are the raw export, and all nineteen of them fail that check. So
   the alarm is demonstrated with real results rather than described.

   Also here: the two extra fields beside the text. Nineteen trackers name
   their priorities nineteen different ways, and flattening those onto one
   vocabulary loses information — which this stop shows rather than glosses.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { createTable } from "../viz/table.js";
import { factEl, factText } from "../ui/provenance.js";
import { stamp } from "../ui/stamp.js";
import { clean } from "../sim/clean.js";
import * as data from "../core/data.js";

export async function mount(host, ctx) {
  const [categorical, validation] = await Promise.all([
    data.categorical(),
    data.validation(),
  ]);

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "Links, numbers, leftovers",
    standfirst: `Three rules left, two extra fields to make comparable, and one question
      that matters more than any of them: once the data has been cleaned, how does the
      training code know that it <em>was</em>?`,
  });

  card.append(buildMarkers());
  card.append(buildLengthFloor(categorical));
  card.append(buildPriorities(categorical));
  card.append(buildAlarm(validation));

  card.append(handover({
    question: `The data is clean and the pipeline knows it. Now it has to be divided up:
      some to learn from, some to check against, some kept sealed until the very end. Get
      that division wrong and every number afterwards is worthless.`,
    cta: "Cutting the data three ways",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   The three remaining rules
   ========================================================================== */

function buildMarkers() {
  const section = block(
    "Links and ticket numbers become markers",
    statRow(
      createStatPlate({
        label: "links replaced",
        value: factEl("cleaning.urls"),
        caption: "each becomes [URL]",
      }),
      createStatPlate({
        label: "ticket references replaced",
        value: factEl("cleaning.issue_refs"),
        caption: "each becomes [ISSUE_REF]",
      }),
    ),
    prose(`Both follow the same reasoning as the code blocks. A link to a build server is
      not information about how hard a task is &mdash; but the fact that somebody needed
      to link to a build server might be. Replacing it with a marker keeps the second
      thing and throws away the first.`),
    prose(`Ticket references matter for a subtler reason. Left alone, a model could learn
      that issues mentioning <code>PROJ-4471</code> tend to be big &mdash; which is
      memorising a project's numbering, not learning about effort. It would score well in
      testing and be useless on anything new.`, "note"),
  );

  /* A tiny before/after, computed by the same code as the previous stop. */
  const demo = "See ABC-1234 and the log at https://ci.example.org/build/9912?full=1";
  const pair = el("div", "before-after");
  [["as written", demo], ["as the model sees it", clean(demo)]].forEach(([label, text]) => {
    const box = el("div", "ba-box mat-recess");
    box.append(el("span", "ba-label", label));
    box.append(el("p", "ba-text", text));
    pair.append(box);
  });
  section.append(pair);

  section.append(prose(`Then the last rule, which is pure housekeeping: newlines, tabs and
    runs of spaces all collapse to a single space, and the result is trimmed.`, "note"));

  return section;
}

/* ==========================================================================
   The length floor
   ========================================================================== */

function buildLengthFloor(categorical) {
  const dropped = categorical?.dropped_short ?? [];
  const section = block(
    "Too short to keep",
    prose(`After all that removing, a few issues have nothing left. If the cleaned title
      and description together come to under ten characters, the row is dropped &mdash;
      there is no task description there to learn from, only a fragment.`),
  );

  if (dropped.length) {
    const list = el("ul", "drop-list");
    dropped.forEach((entry) => {
      const item = el("li", "drop-row mat-sub");
      item.append(el("span", "drop-n", String(entry.rows)));
      item.append(el("span", "drop-name",
        `${entry.project.replace(/_/g, " ")} — ${entry.rows === 1 ? "one issue" : `${entry.rows} issues`}`));
      list.append(item);
    });
    section.append(list);
  }

  section.append(prose(`That is <span data-fact="corpus.rows_dropped"></span> rows out of
    <span data-fact="corpus.rows_raw"></span>, from two projects, and they are counted
    rather than quietly discarded. A pipeline that removes rows without saying how many
    is a pipeline nobody can check.`));

  return section;
}

/* ==========================================================================
   Priorities: nineteen vocabularies, one scale
   ========================================================================== */

function buildPriorities(categorical) {
  const section = block(
    "Nineteen ways to say urgent",
    prose(`Beside the words, the model is given two extra facts about each issue: what
      kind of task it is, and how urgent it was marked. The trouble is that every tracker
      invents its own words for urgency, so they have to be flattened onto one vocabulary
      before they can mean anything across projects.`),
    statRow(
      createStatPlate({
        label: "different priority words",
        value: factEl("categorical.priority_raw"),
        caption: "across the nineteen trackers",
      }),
      createStatPlate({
        label: "after flattening",
        value: factEl("categorical.priority_canonical"),
        caption: "Highest · High · Medium · Low · Lowest · Unknown",
      }),
      createStatPlate({
        label: "have no priority at all",
        value: factEl("categorical.priority_missing_pct"),
        caption: `${factText("categorical.priority_missing")} issues`,
      }),
    ),
  );

  const rows = categorical?.priorities ?? [];
  if (rows.length) {
    const table = createTable({
      caption: "Every priority word in the corpus and what it becomes",
      rows: rows.map((row) => ({
        raw: row.raw === "(missing)" ? "— nothing recorded —" : row.raw,
        count: row.count,
        canonical: row.canonical,
      })),
      sort: { key: "count", dir: "desc" },
      columns: [
        { key: "raw", label: "As written in the tracker", align: "left" },
        { key: "count", label: "Issues", align: "right",
          format: (v) => Number(v).toLocaleString() },
        { key: "canonical", label: "Becomes", align: "left" },
      ],
    });
    const scroller = el("div", "table-scroll");
    scroller.append(table.el);
    section.append(scroller);
  }

  section.append(prose(`Two things in that table are worth stopping on. Suffixed variants
    like <code>Major - P3</code> are mapped by their first word, so they land with plain
    <code>Major</code> rather than becoming a category of their own. And
    <span data-fact="categorical.priority_unmapped"></span> issues carry a word no rule
    covers &mdash; they end up in <code>Unknown</code>, the same bucket as an issue with
    no priority at all. That is a small, real loss of information, sitting in plain
    sight.`));

  const types = createDrawer({
    label: "Issue types are handled differently",
    summary: "deliberately not merged",
  });
  types.body.append(prose(`There are <span data-fact="categorical.type_raw"></span> distinct
    type words, and unlike priorities they are <em>not</em> mapped together. A "Bug" and a
    "Defect" stay separate even though they plainly mean the same thing.`));
  types.body.append(prose(`The reasoning is that urgency is a scale every tracker is
    trying to express &mdash; so aligning the words recovers a shared meaning &mdash;
    whereas issue types are project-specific categories, and merging them would be
    guessing at what each team meant. The model gets one embedding per type it saw during
    training, and anything unseen becomes Unknown.`, "note"));

  if (categorical?.types?.length) {
    const list = el("p", "note");
    list.textContent = categorical.types
      .map((t) => `${t.raw === "(missing)" ? "—" : t.raw} (${t.count.toLocaleString()})`)
      .join(" · ");
    types.body.append(list);
  }
  section.append(types.el);

  return section;
}

/* ==========================================================================
   The alarm — demonstrated live, on this machine's own files
   ========================================================================== */

function buildAlarm(validation) {
  const section = block(
    "The alarm that refuses uncleaned data",
    prose(`All of the cleaning happens <strong>once</strong>, when the data is exported,
      and never again while training. That is what makes a run repeatable: the training
      code reads files and does not transform them.`),
    prose(`Which raises the obvious risk. If someone points the training script at a
      folder of raw exports, everything still runs &mdash; it just trains on text full of
      markup, and the results are quietly worse for a reason nobody would spot. So the
      loader checks, and refuses.`),
  );

  const files = validation?.files ?? [];
  if (!files.length) {
    section.append(prose(`No validation results have been generated.`, "note"));
    return section;
  }

  const failing = files.filter((f) => !f.passes);
  const banner = el("div", "alarm-banner mat-recess");
  banner.append(stamp(failing.length ? "measured" : "measured"));
  const text = el("p", "alarm-text");
  text.innerHTML = failing.length
    ? `<strong>${failing.length} of ${files.length} files on this machine fail the check.</strong>
       The copy of the data sitting beside this website is the <em>raw</em> export, not the
       cleaned one &mdash; so the alarm you are about to read is not a demonstration. It is
       the real output of the project's own loader, run against these files.`
    : `All ${files.length} files pass the check.`;
  banner.append(text);
  section.append(banner);

  const rows = files.map((file) => ({
    file: file.file.replace(/\.csv$/, "").replace(/_/g, " "),
    verdict: file.passes ? "accepted" : "refused",
    reason: shortenReason(file.reason),
  }));

  const table = createTable({
    caption: "What the loader says about each file on this machine",
    rows,
    sort: { key: "file", dir: "asc" },
    columns: [
      { key: "file", label: "File", align: "left" },
      { key: "verdict", label: "Loader", align: "left" },
      { key: "reason", label: "Because", align: "left", sortable: false },
    ],
    onRow: (row, tr) => { tr.dataset.role = row.verdict === "refused" ? "refused" : ""; },
  });
  const scroller = el("div", "table-scroll");
  scroller.append(table.el);
  section.append(scroller);

  const threshold = createDrawer({
    label: "Why the check allows a little mess rather than none",
    summary: "1%, not zero",
  });
  threshold.body.append(prose(`The check does not demand perfection. It fires when more
    than one percent of a column still looks uncleaned, and a handful of rows are allowed
    through.`));
  threshold.body.append(prose(`That is because a few real issues survive cleaning with
    text that <em>looks</em> like leftover markup and is not. One title in the corpus
    genuinely begins and ends with a quotation mark, because it is quoting an error
    message. One description contains <code>{noformat)</code> &mdash; a closing bracket
    instead of a brace, typed by hand years ago, which no rule matches because no rule
    should. A zero-tolerance check would fail on those and force someone to "fix" correct
    data.`, "note"));
  threshold.body.append(prose(`One percent is loose enough to survive individual oddities
    and tight enough to catch a whole file of raw markup &mdash; which, as the table
    above shows, is exactly what it did here.`, "note"));
  section.append(threshold.el);

  return section;
}

function shortenReason(reason) {
  if (!reason) return "everything looks cleaned";
  return String(reason)
    .replace(/^\[[^\]]+\]\s*/, "")
    .replace(/\s*Re-export.*$/, "")
    .trim();
}
