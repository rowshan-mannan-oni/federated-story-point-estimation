/* ==========================================================================
   harness.js — the small amount of machinery every suite shares.

   Loading data: the site fetches its JSON with fetch(), which Node has no
   business answering, so fetch is pointed at the real files on disk. The
   suites therefore test against the same site/data/*.json the browser reads,
   not against fixtures that could drift from it.
   ========================================================================== */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { install } from "./dom.js";

install();

globalThis.fetch = async (url) => {
  try {
    const body = readFileSync(fileURLToPath(url), "utf8");
    return { ok: true, status: 200, json: async () => JSON.parse(body) };
  } catch {
    return { ok: false, status: 404, json: async () => null };
  }
};

let failures = 0;
let checks = 0;

/** One assertion. `detail` is printed only when it fails, and should say why. */
export function check(name, condition, detail = "") {
  checks += 1;
  const ok = Boolean(condition);
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${!ok && detail ? " -> " + detail : ""}`);
  if (!ok) failures += 1;
  return ok;
}

export function section(title) {
  console.log(`=== ${title} ===`);
}

/**
 * Call at the end of a suite. Exits non-zero if anything failed.
 *
 * A suite that ran no checks is treated as a failure rather than a pass. That
 * case is not hypothetical: a bad edit once left a suite containing nothing
 * but its imports, and it reported success for as long as nobody looked.
 */
export function done() {
  if (checks === 0) {
    console.log("\nNO CHECKS RAN — a suite that asserts nothing cannot pass.");
    process.exit(1);
  }
  console.log(failures
    ? `\n${failures} FAILURE(S) out of ${checks}`
    : `\nall ${checks} checks passed`);
  process.exit(failures ? 1 : 0);
}

/** Load a station and mount it, returning the card plus what it called back. */
export async function mountStop(id, index) {
  const stops = await import("../js/core/stops.js");
  const entry = stops.STOPS.find((s) => s.id === id);
  if (!entry) throw new Error(`no stop called "${id}"`);

  const { document } = await import("./dom.js");
  const host = document.createElement("div");
  const calls = { next: 0, goTo: [], goToId: [] };

  const module = await import(`../js/stations/${id}.js`);
  await module.mount(host, {
    stop: entry,
    index,
    next: () => { calls.next += 1; },
    goTo: (i) => { calls.goTo.push(i); },
    goToId: (v) => { calls.goToId.push(v); },
  });

  return { entry, host, card: host.children[0], calls, stops };
}

export { deepText } from "./dom.js";
