/* ==========================================================================
   run.mjs — run every suite.

       node site/tests/run.mjs            all suites
       node site/tests/run.mjs extras     just the ones whose name matches

   Each suite is a separate process, so one crashing cannot take the rest with
   it, and a failure exits non-zero for anything that wants to gate on it.
   ========================================================================== */

import { readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const filter = process.argv[2];

const suites = readdirSync(here)
  .filter((name) => name.endsWith(".test.js"))
  .filter((name) => !filter || name.includes(filter))
  .sort();

if (!suites.length) {
  console.error(filter ? `No suite matches "${filter}".` : "No suites found.");
  process.exit(1);
}

let failed = 0;
const results = [];

for (const suite of suites) {
  const label = suite.replace(/\.test\.js$/, "");
  const run = spawnSync(process.execPath, [join(here, suite)], { encoding: "utf8" });
  const output = (run.stdout || "") + (run.stderr || "");
  const passed = run.status === 0;
  if (!passed) failed += 1;

  const summary = output.trim().split("\n").pop() ?? "";
  results.push({ label, passed, summary });

  if (!passed) {
    console.log(`\n──────── ${label} ────────`);
    console.log(output.trimEnd());
  }
}

console.log("\n──────── summary ────────");
for (const { label, passed, summary } of results) {
  console.log(`  ${passed ? "pass" : "FAIL"}  ${label.padEnd(18)} ${summary}`);
}
console.log(failed
  ? `\n${failed} of ${suites.length} suites failed.`
  : `\nAll ${suites.length} suites passed.`);

process.exit(failed ? 1 : 0);
