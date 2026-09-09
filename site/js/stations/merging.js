/* ==========================================================================
   Stop 11 — The merging problem.

   This is the argument the whole method rests on, and the docs assert it from
   the literature rather than showing it. So this stop shows it, three ways:

     1. A live experiment. Real matrix arithmetic in the browser — drag the
        heterogeneity control and watch the merged patch go wrong, throw the
        freeze switch and watch the error vanish.
     2. The closed form. The error is not merely small when A is shared, it is
        exactly zero, because the gap between the two ways of merging IS the
        cross-covariance of the clients' drift. The page checks that identity
        live and reports the residual.
     3. The same experiment at the model's real size, run by the extraction
        script, so nobody can wave away the small demo as a toy.

   The honest limit, stated on the page: exactness is a mathematical property,
   not evidence that the estimates come out better. The experiment that would
   show that (--no-ffa-lora) is listed as optional in the thesis and has not
   been run.
   ========================================================================== */

import { stopCard, block, prose, handover, statRow, el } from "../ui/stop-parts.js";
import { createStatPlate, createDrawer, createReadout, createMeter } from "../ui/readouts.js";
import { createSlider, createSwitch } from "../ui/controls.js";
import { factEl, factText } from "../ui/provenance.js";
import { mergeExperiment, thumbnail } from "../sim/lora.js";
import * as data from "../core/data.js";

export async function mount(host, ctx) {
  const merging = await data.merging();

  const card = stopCard({
    stop: ctx.stop,
    index: ctx.index,
    title: "The merging problem",
    standfirst: `Eighteen projects have each trained a patch. The server has to turn them
      into one. There is an obvious way to do that, it is what everybody reaches for first,
      and it is wrong &mdash; in a way that is easy to miss and possible to measure.`,
  });

  card.append(buildTheObviousWay());
  card.append(buildRig());
  card.append(buildIdentity());
  card.append(buildFullSize(merging));
  card.append(buildTheCost());

  card.append(handover({
    question: `The patches merge exactly now. But there is still the other end of the
      model to sort out &mdash; the part that turns all this into an answer. And the
      answers are not five unrelated labels: they are a scale, where being wildly wrong
      ought to cost more than being nearly right.`,
    cta: "The answers are ordered",
    next: ctx.next,
  }));

  host.append(card);
}

/* ==========================================================================
   The obvious way, and why it looks fine
   ========================================================================== */

function buildTheObviousWay() {
  return block(
    "The obvious way to merge",
    prose(`Each project sends back its patch: two strips, one tall and one wide. The
      server has eighteen tall strips and eighteen wide ones, and needs a single patch to
      send back out.`),
    prose(`So average them. Average the eighteen tall strips into one tall strip, average
      the eighteen wide ones into one wide strip, and there is your merged patch. This is
      what averaging means, it is what the server does with every other number it
      receives, and it takes a moment to see the problem.`),
    prose(`The problem is that a patch is not the two strips. It is the two strips
      <em>multiplied together</em>. And the average of a product is not the product of the
      averages &mdash; the same reason the average of two people's heights times their
      weights is not the average height times the average weight.`, "note"),
  );
}

/* ==========================================================================
   The live rig
   ========================================================================== */

function buildRig() {
  const CLIENTS = 18;
  const WIDTH = 64;
  const RANK = 4;

  let spread = 0.02;
  let frozen = false;

  const meter = createMeter({
    label: "how wrong the merged patch is",
    min: 0, max: 1, value: 0,
    format: (v) => `${(v * 100).toFixed(1)}%`,
  });
  const residual = createReadout({
    label: "closed form vs measured gap",
    value: 0,
    format: (v) => v.toExponential(1),
    caption: "zero means the explanation below is exact",
  });

  const grids = el("div", "merge-grids");
  const verdict = el("p", "merge-verdict");

  function paint() {
    const result = mergeExperiment({
      spread, frozen, clients: CLIENTS, width: WIDTH, rank: RANK,
    });

    meter.set(Math.min(result.error, 1));
    residual.set(result.covResidual);

    grids.replaceChildren();
    [
      ["what the update really is", result.truth],
      ["what separate averaging gives", result.naive],
      ["the difference", result.gap],
    ].forEach(([label, m]) => {
      const box = el("div", "merge-grid");
      box.append(el("span", "merge-grid-label", label));
      box.append(gridArt(m, label));
      grids.append(box);
    });

    verdict.dataset.state = frozen ? "exact" : result.error > 0.2 ? "bad" : "off";
    verdict.textContent = frozen
      ? `With one shared frozen half, the two ways of merging agree to ${result.error.toExponential(1)} `
        + `— that is not a small error, it is rounding. The third panel is empty because there is nothing in it.`
      : `The merged patch is ${(result.error * 100).toFixed(1)}% wrong. Not off in one corner — `
        + `the third panel shows the difference is spread over the whole thing.`;
  }

  const drift = createSlider({
    label: "how differently the projects trained",
    min: 0, max: 0.05, step: 0.005, value: spread,
    format: (v) => (v === 0 ? "identical" : v.toFixed(3)),
    onInput: (v) => { spread = v; paint(); },
  });

  const freeze = createSwitch({
    label: "freeze one half, shared by everyone",
    checked: frozen,
    hint: "what this thesis actually does",
    onChange: (v) => { frozen = v; paint(); },
  });

  paint();

  const controls = el("div", "merge-controls");
  controls.append(drift.el, freeze.el);
  const readouts = el("div", "merge-readouts");
  readouts.append(meter.el, residual.el);

  const section = block(
    "Try it — eighteen patches, merged both ways",
    prose(`Below, eighteen projects each train a patch, and the server merges them the
      obvious way. The first panel is what the merged update should be; the second is what
      separate averaging produces; the third is the difference between them.`),
    controls,
    readouts,
    grids,
    verdict,
    prose(`Start by dragging the drift control down to zero &mdash; when every project
      trains identically the error disappears, which is exactly why this bug survives
      testing on tidy data. Then put it back up and throw the freeze switch.`, "note"),
  );

  const scale = createDrawer({
    label: "Why is this running at 64 and not 768?",
    summary: "so it answers while you drag",
  });
  scale.body.append(prose(`The real patches stand in for a 768-wide square. Merging
    eighteen of those on every drag would take long enough to be annoying, so the demo
    uses a 64-wide one with strips of 4.`));
  scale.body.append(prose(`Nothing about the argument depends on the size &mdash; and to
    make sure that is not just a claim, the extraction script runs the same experiment at
    the full <span data-fact="merge.width"></span>, with the real rank and the real number
    of projects. Those numbers are further down this page.`, "note"));
  section.append(scale.el);

  return section;
}

function gridArt(matrix, label) {
  const art = el("div", "merge-cells");
  const cells = thumbnail(matrix, 12);
  const peak = Math.max(...cells.flat().map(Math.abs), 1e-12);
  cells.forEach((row) => {
    row.forEach((value) => {
      const cell = el("span", "merge-cell");
      // Greyscale only: magnitude becomes darkness, sign becomes a border.
      cell.style.setProperty("--v", String(Math.min(Math.abs(value) / peak, 1)));
      if (value < 0) cell.dataset.sign = "neg";
      art.append(cell);
    });
  });
  art.setAttribute("role", "img");
  art.setAttribute("aria-label", `${label}: a 12 by 12 sample of the matrix`);
  return art;
}

/* ==========================================================================
   Where the error comes from
   ========================================================================== */

function buildIdentity() {
  const section = block(
    "The error is not mysterious — it has a formula",
    prose(`It would be easy to leave this at "averaging products is not the same as
      multiplying averages" and move on. But the gap between the two is not vague. It is
      exactly one thing:`),
  );

  const formula = el("div", "formula mat-recess");
  formula.append(el("span", "formula-line", "avg(Bᵢ·Aᵢ)  −  avg(B)·avg(A)"));
  formula.append(el("span", "formula-eq", "="));
  formula.append(el("span", "formula-line", "Σ pᵢ (Bᵢ − B̄)(Aᵢ − Ā)"));
  section.append(formula);

  section.append(prose(`In words: the error is how far each project's tall strip drifted
    from the average, multiplied by how far its wide strip drifted, added up across
    projects. It is a measure of <em>disagreement</em>. Projects that all trained the same
    way contribute nothing to it; projects that pulled in different directions contribute
    a lot.`));

  section.append(prose(`That is why the drift control does what it does, and why the error
    vanishes at zero drift. The readout above labelled "closed form vs measured gap"
    is this identity being checked on every redraw: the page computes the error one way,
    computes that sum the other way, and reports the largest difference between them. It
    sits at rounding noise.`, "note"));

  section.append(prose(`Now look at the formula again with one half frozen. If every
    project has the same A, then every <strong>(Aᵢ − Ā) is zero</strong>. Every term in
    the sum is zero. The error is not reduced or bounded or made acceptable &mdash; it is
    identically nothing, for any patches, any drift, any number of projects.`));

  return section;
}

/* ==========================================================================
   The same thing at full size
   ========================================================================== */

function buildFullSize(merging) {
  const section = block(
    "The same experiment at the model's real size",
    prose(`Run by the extraction script rather than the browser, at
      <span data-fact="merge.width"></span> wide with the run's own rank and
      <span data-fact="merge.clients"></span> projects.`),
    statRow(
      createStatPlate({
        label: "both halves trained",
        value: factEl("merge.error_independent"),
        caption: "how wrong the merged patch is",
      }),
      createStatPlate({
        label: "one half frozen and shared",
        value: factEl("merge.error_frozen"),
        caption: "floating-point rounding, nothing more",
      }),
      createStatPlate({
        label: "the formula's residual",
        value: factEl("merge.identity_residual"),
        caption: "how exact the explanation is",
      }),
    ),
    prose(`With both halves trained the merged patch is
      <span data-fact="merge.error_independent"></span> wrong &mdash; which at that
      magnitude means it has almost nothing to do with what the projects actually learned.
      The server would be sending out a patch nobody trained.`),
  );

  if (merging?.curve?.length) {
    const list = el("ul", "drift-list");
    merging.curve.forEach((point) => {
      const item = el("li", "drift-row");
      item.append(el("span", "drift-x",
        point.spread === 0 ? "identical" : point.spread.toFixed(3)));
      const bar = el("span", "drift-bar");
      const fill = el("span", "drift-fill");
      fill.style.setProperty("--w", String(point.error));
      bar.append(fill);
      item.append(bar);
      item.append(el("span", "drift-y", `${(point.error * 100).toFixed(1)}%`));
      list.append(item);
    });
    const wrap = el("div", "sub-block");
    wrap.append(el("p", "hist-caption",
      "Error against how differently the projects trained, at full size."));
    wrap.append(list);
    section.append(wrap);
  }

  section.append(prose(`Which is the shape you would expect from the formula: no
    disagreement, no error; more disagreement, more error. And federated story point
    estimation is a setting built out of disagreement &mdash; nineteen projects that label
    work differently is the whole reason this thesis exists.`, "note"));

  return section;
}

/* ==========================================================================
   What the fix costs
   ========================================================================== */

function buildTheCost() {
  const section = block(
    "What freezing costs",
    prose(`The fix is to fix. One half of every patch is generated randomly at the start,
      shared with every project, and never trained by anyone. Only the other half learns,
      and only that half is ever uploaded &mdash; which is also why the upload is
      <span data-fact="params.lora_b"></span> numbers and not twice that.`),
    prose(`It is not free. Half the patch is no longer learning, so each patch can express
      less than it could. The published argument is that exact merging is worth more than
      the lost flexibility, and that is the argument this thesis follows.`),
  );

  const honest = createDrawer({
    label: "Is that trade actually worth it here?",
    summary: "unmeasured — and it is a named gap",
  });
  honest.body.append(prose(`Everything above proves a mathematical property: with a shared
    frozen half, merging is exact. It does <strong>not</strong> show that the resulting
    model estimates story points better than one merged the naive way.`));
  honest.body.append(prose(`Those are different claims, and only the first is established
    here. The experiment that would settle the second &mdash; running the whole thing again
    with both halves trained and comparing the estimates &mdash; is listed in the thesis's
    own notes as an optional question, one flag away, and has not been run. It is on the
    page rather than in a footnote because it is the obvious thing an examiner asks.`,
    "note"));
  section.append(honest.el);

  return section;
}
