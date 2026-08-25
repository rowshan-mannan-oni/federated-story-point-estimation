/* ==========================================================================
   stamp.js — saying out loud how solid a number is.

   The site shows three quite different sorts of figure, and a reader must
   never have to guess which one they are looking at:

     MEASURED     recomputed from the project's data, here, just now
     FROM A RUN   read out of a finished training run's files
     WORKED OUT   arithmetic over other figures
     SIMULATED    produced by a demonstration, not by an experiment
     NO DATA      the input for it is not on this machine

   Anything that is not a real measurement wears its stamp visibly. That is
   the whole point: a chart with invented data must never be mistakable for
   a result.
   ========================================================================== */

const STAMPS = {
  measured:    { text: "MEASURED",    tone: "solid" },
  run:         { text: "FROM A RUN",  tone: "solid" },
  derived:     { text: "WORKED OUT",  tone: "solid" },
  simulated:   { text: "SIMULATED",   tone: "loud" },
  preliminary: { text: "PRELIMINARY", tone: "loud" },
  missing:     { text: "NO DATA",     tone: "quiet" },
};

/** A small stamp element. `kind` is one of the keys above. */
export function stamp(kind, { title } = {}) {
  const spec = STAMPS[kind] ?? STAMPS.missing;
  const el = document.createElement("span");
  el.className = "stamp";
  el.dataset.kind = kind;
  el.dataset.tone = spec.tone;
  el.textContent = spec.text;
  if (title) el.title = title;
  return el;
}

export function stampFor(entry) {
  return stamp(entry?.kind ?? "missing");
}

/**
 * A banner for a whole panel — used where a stamp on one number is not
 * enough, e.g. a chart drawn from a demonstration rather than an experiment.
 */
export function banner(kind, message) {
  const el = document.createElement("p");
  el.className = "stamp-banner";
  el.dataset.tone = (STAMPS[kind] ?? STAMPS.missing).tone;
  el.append(stamp(kind));
  el.append(Object.assign(document.createElement("span"), {
    className: "stamp-message",
    textContent: message,
  }));
  return el;
}

/**
 * The standing warning for results that came from one preliminary run.
 * Every stop that shows run numbers uses this exact wording, so the caveat
 * cannot quietly go missing from one of them.
 */
export function preliminaryBanner(runFacts) {
  const bits = [
    runFacts?.condition && `condition: ${runFacts.condition}`,
    runFacts?.seed != null && `seed ${runFacts.seed}`,
    runFacts?.split && `${runFacts.split} split`,
  ].filter(Boolean).join(" · ");

  return banner(
    "preliminary",
    `One run only${bits ? ` (${bits})` : ""}. Final numbers are still being produced ` +
    `on a bigger machine, and this run pre-dates two later fixes — treat it as a ` +
    `shape to reason about, not a final result.`
  );
}
