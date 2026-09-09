# Tests for the walk-through

```
node site/tests/run.mjs             every suite
node site/tests/run.mjs extras      only suites whose name matches
node site/tests/extras.test.js      one suite, with its full output
```

Needs Node and nothing else — no install, no browser, no network. Each suite runs in its
own process, so one crashing cannot take the rest with it, and the run exits non-zero if
anything fails.

## What these check

Not that the code runs. That it tells the truth.

- **Every figure a stop shows is the figure in `site/data/`.** The suites load the same
  JSON the browser does, so a page cannot quietly drift from the data.
- **The arithmetic closes.** The parts of the upload add up to the recorded total; the
  length buckets add up to the training pile; the dropped rows add up to the corpus-wide
  count.
- **The claims are the measurements.** Where a stop says a random split lets the model
  see the future, a suite checks that 18 of 18 projects really do overlap. Where it says
  priority is not a clean scale, a suite checks that "high" really does sit below
  "medium".
- **The caveats are still on the page.** If a paragraph admitting an experiment was never
  run gets edited away, the suite that looks for it fails.
- **The browser's copy of the cleaning rules matches the Python**, character for
  character, on several hundred real corpus fields.

## How it works

`dom.js` is a small DOM — enough for the stops to build themselves in Node. It is
deliberately faithful where faithfulness matters (elements have `nodeType`, `style` has
`setProperty`, `dataset` maps to `data-` attributes), because every place it lied, a test
passed that should not have.

Its one significant limitation: **`innerHTML` is stored, not parsed.** Prose written that
way cannot be queried as elements, so text assertions read the stored markup instead —
that is what `deepText()` is for. It also collapses whitespace, since a line break inside
a template literal is not a difference HTML would render.

`harness.js` holds `check`, `section` and `done`, and points `fetch` at the real files on
disk. A suite that runs **zero** checks fails rather than passes — a bad edit once left a
suite with nothing but its imports, and it reported success for as long as nobody looked.

## Adding a suite

Name it `<stop-id>.test.js`; the runner picks it up automatically.

```js
import { check, section, done, deepText } from "./harness.js";
import { document } from "./dom.js";

const data = await import("../js/core/data.js");
await data.init();

section("what this group is about");
check("what should be true", actual === expected, `got ${actual}`);

done();
```

Re-run `python site/tools/extract_facts.py` first if the data has changed — the suites
assert against whatever is in `site/data/` at the time.
