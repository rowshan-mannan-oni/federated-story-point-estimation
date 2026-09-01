/* ==========================================================================
   Stop 4 — The archive.

   Every project keeps its own data, so the obvious next question is what that
   data actually is. This stop is the filing cabinet: all nineteen trackers,
   their real sizes, their real date ranges, and the awkward corners nobody
   would put in a summary table.

   Three things have to survive this stop, because later stops lean on them:

     1. The projects are wildly different sizes — 18 to 1 — which is why the
        server weights their contributions rather than treating them equally.
     2. Two projects are special: the biggest is held back to warm the model
        up, and one mid-sized one is held back entirely for the newcomer
        experiment at the end.
     3. The class imbalance is mild. When the scores look weak later, that is
        not the reason, and the reader should already know it.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer } from "../ui/readouts.js";
import { createTable } from "../viz/table.js";
import { createDistStrip, createDistLegend } from "../viz/dist-strip.js";
import { factText } from "../ui/provenance.js";
import * as data from "../core/data.js";

/* Named in the run's own configuration, not chosen here. */
const WARM_START = "Lsstcorp_Data_management";
const HOLDOUT = "Hyperledger_Sawtooth";

export async function mount(host, ctx) {
  const [dataset, runInfo] = await Promise.all([data.dataset(), data.run()]);
  const projects = dataset?.projects ?? [];

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "The archive",
    standfirst: `Nineteen issue trackers, <span data-fact="corpus.rows_raw"></span> tasks,
      sixteen years of somebody's working life. This is everything the model will ever
      see &mdash; so it is worth looking at properly, including the parts that are
      inconvenient.`,
  });

  /* ---- the cabinet ------------------------------------------------------ */
  card.append(buildCabinet(projects, runInfo));

  /* ---- they are not the same size --------------------------------------- */
  card.append(block(
    "They are not the same size",
    statRow(
      createStatPlate({
        label: "issues, largest project",
        value: factText("corpus.largest_rows"),
        caption: pretty(factText("corpus.largest_project")),
      }),
      createStatPlate({
        label: "issues, smallest project",
        value: factText("corpus.smallest_rows"),
        caption: pretty(factText("corpus.smallest_project")),
      }),
      createStatPlate({
        label: "size difference",
        value: factText("corpus.size_ratio"),
        caption: "between the two ends",
      }),
    ),
    prose(`One project has eighteen times the data of another. That is not a flaw in the
      dataset, it is what a real federation looks like &mdash; and it forces a decision
      the server cannot avoid: when eighteen projects each send back what they learned,
      does everyone get an equal say?`),
    prose(`They do not. Each project's contribution is weighted by how many examples it
      trained on, so the large projects pull harder. That is the standard choice, and it
      is worth remembering when a small project's estimates look poor later: it had less
      data <em>and</em> less influence over the shared model.`, "note"),
  ));

  /* ---- the odd corners --------------------------------------------------- */
  card.append(buildOddCorners(projects));

  /* ---- the balance is fine ---------------------------------------------- */
  card.append(buildBalance());

  card.append(handover({
    question: `That is the data as exported. But nobody writes issue descriptions for a
      machine to read &mdash; they are full of pasted stack traces, links, wiki markup and
      HTML. None of that can reach the model as it stands.`,
    cta: "The cleaning bench",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   The cabinet: all nineteen, sortable, each one openable
   ========================================================================== */

function buildCabinet(projects, runInfo) {
  const section = block(
    "The nineteen",
    prose(`Sort by any column. Two projects are marked: one is used to give the model a
      head start before federated training begins, and one is held out of the whole study
      so it can play the part of a newcomer at the end.`),
  );

  if (!projects.length) {
    section.append(prose(`The project list has not been generated yet. Run
      <code>python site/tools/extract_facts.py</code> to fill this in.`));
    return section;
  }

  const rows = projects.map((project) => ({
    ...project,
    label: pretty(project.name),
    mean: meanStoryPoint(project.story_points),
    span: `${project.first_issue.slice(0, 4)}–${project.last_issue.slice(0, 4)}`,
    role: project.name === WARM_START ? "head start"
      : project.name === HOLDOUT ? "held out" : "",
  }));

  const table = createTable({
    caption: "The nineteen projects, their sizes, date ranges and story point mix",
    rows,
    sort: { key: "rows_clean", dir: "desc" },
    columns: [
      {
        key: "label", label: "Project", align: "left",
        format: (value, row) => {
          const wrap = el("span", "arch-name");
          wrap.append(el("span", null, value));
          if (row.role) wrap.append(el("span", "arch-role", row.role));
          return wrap;
        },
      },
      { key: "rows_clean", label: "Issues", align: "right",
        format: (value) => Number(value).toLocaleString() },
      { key: "span", label: "Span", align: "right" },
      { key: "desc_missing_pct", label: "No description", align: "right",
        format: (value) => `${value.toFixed(1)}%` },
      { key: "mean", label: "Average points", align: "right",
        format: (value) => value.toFixed(2) },
      {
        key: "story_points", label: "Mix of story points", align: "left", sortable: false,
        format: (value) => createDistStrip({ counts: value, compact: true }).el,
      },
    ],
    onRow: (row, tr) => {
      if (row.role) tr.dataset.role = row.role;
    },
  });

  const scroller = el("div", "table-scroll");
  scroller.append(table.el);
  section.append(createDistLegend());
  section.append(scroller);

  /* one drawer per project, for the full detail */
  const details = createDrawer({
    label: "Open a project",
    summary: "priorities, types, what cleaning removed",
  });
  const list = el("div", "arch-details");
  [...rows].sort((a, b) => b.rows_clean - a.rows_clean).forEach((row) => {
    list.append(projectCard(row));
  });
  details.body.append(list);
  section.append(details.el);

  if (runInfo?.config?.warmstart_project && runInfo.config.warmstart_project !== WARM_START) {
    section.append(prose(`Note: the run on this machine warmed up on
      <strong>${pretty(runInfo.config.warmstart_project)}</strong>, not the project marked
      above. The marking follows this page's constant and should be corrected.`, "note"));
  }

  return section;
}

function projectCard(row) {
  const cardEl = el("article", "arch-card mat-recess");
  cardEl.append(el("h3", "arch-card-title", row.label));

  const facts = el("dl", "arch-facts");
  const add = (term, value) => {
    facts.append(el("dt", null, term));
    facts.append(el("dd", null, value));
  };
  add("Issues kept", Number(row.rows_clean).toLocaleString());
  if (row.rows_dropped) add("Dropped as too short", String(row.rows_dropped));
  add("First issue", row.first_issue);
  add("Last issue", row.last_issue);
  add("Without a description", `${row.desc_missing_pct.toFixed(1)}%`);
  add("Kinds of issue", String(row.types));
  add("Average story point", row.mean.toFixed(2));
  cardEl.append(facts);

  const priorities = Object.entries(row.priorities || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => `${name} ${count.toLocaleString()}`)
    .join(" · ");
  cardEl.append(el("p", "arch-priorities", priorities || "no priorities recorded"));

  cardEl.append(createDistStrip({
    counts: row.story_points,
    note: `${Number(row.rows_clean).toLocaleString()} issues`,
  }).el);

  return cardEl;
}

/* ==========================================================================
   The corners nobody would put in a summary
   ========================================================================== */

function buildOddCorners(projects) {
  const section = block(
    "The awkward corners",
    prose(`Real data has parts that do not fit the story. These are the ones that
      actually change how the pipeline had to be built.`),
  );

  const list = el("ul", "corner-list");

  const corners = [
    {
      title: "Nearly half of one project has no description at all",
      body: `In <span data-fact="corpus.desc_missing_worst_project"></span>,
        <span data-fact="corpus.desc_missing_worst_pct"></span> of issues are a title and
        nothing else. The tempting fix is to drop them. That would throw away hundreds of
        real tasks and quietly make the dataset easier than reality, so they are kept:
        the model reads the title alone and has to cope, exactly as a person would.`,
    },
    {
      title: "One project records no priorities whatsoever",
      body: `Every issue in <span data-fact="corpus.no_priority_project"></span> has an
        unrecorded priority &mdash; all ten thousand of them. Priority is one of the two
        extra facts fed to the model beside the words, so for this project that input
        carries no information at all. It is also, awkwardly, the project used to give
        the model its head start.`,
    },
    {
      title: "Sixteen years of changing habits",
      body: `The oldest issue here was written in
        <span data-fact="corpus.first_issue"></span> and the newest in
        <span data-fact="corpus.last_issue"></span>. Teams change, tools change, and what
        a team calls a 5 drifts over that time. It is the reason the honest way to split
        this data is by date rather than at random &mdash; which is its own stop shortly.`,
    },
    {
      title: "Seven issues were thrown away",
      body: `<span data-fact="corpus.rows_dropped"></span> of
        <span data-fact="corpus.rows_raw"></span> had nothing left after cleaning &mdash;
        under ten characters of actual text. They are logged rather than silently
        dropped, because "we removed some rows" is exactly the sort of sentence that
        should come with a number.`,
    },
  ];

  corners.forEach((corner) => {
    const item = el("li", "corner mat-sub");
    item.append(el("span", "corner-title", corner.title));
    const text = el("p", "corner-body");
    text.innerHTML = corner.body;
    item.append(text);
    list.append(item);
  });

  section.append(list);
  return section;
}

/* ==========================================================================
   Getting ahead of a bad explanation
   ========================================================================== */

function buildBalance() {
  const distribution = data.value("corpus.sp_distribution") || {};
  const counts = {};
  for (const [key, entry] of Object.entries(distribution)) {
    counts[key] = entry.count ?? entry;
  }

  const section = block(
    "Before anyone blames the balance",
    prose(`When a model scores poorly on a five-way choice, the first explanation reached
      for is usually "the classes are imbalanced". Here is the actual mix, so that
      explanation can be checked rather than assumed.`),
  );

  section.append(createDistLegend());
  section.append(createDistStrip({
    title: "All nineteen projects together",
    counts,
    note: `${Number(data.value("corpus.rows_clean")).toLocaleString()} issues`,
  }).el);

  const shares = Object.entries(distribution)
    .map(([sp, entry]) => `${sp}: ${entry.pct}%`)
    .join(" · ");
  section.append(el("p", "note", shares));

  section.append(prose(`The most common answer is
    <span data-fact="corpus.imbalance_ratio"></span> more frequent than the rarest. That
    is mild &mdash; problems where one class outnumbers another a hundred to one are
    routine. So when the scores look weak later in this walk-through, imbalance is not
    the explanation, and this stop is where that was settled.`));

  return section;
}

/* ==========================================================================
   Helpers
   ========================================================================== */

function meanStoryPoint(counts) {
  let total = 0;
  let sum = 0;
  for (const [value, count] of Object.entries(counts || {})) {
    total += count;
    sum += Number(value) * count;
  }
  return total ? sum / total : 0;
}

function pretty(name) {
  return String(name).replace(/_/g, " ");
}
