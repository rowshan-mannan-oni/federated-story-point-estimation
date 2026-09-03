/* ==========================================================================
   clean.js — the cleaning pipeline, in the browser.

   This is a port of export_issues.py's clean_text_field, so a reader can type
   their own tracker text in and watch it change. A port is a liability: if it
   drifts from the Python, the site shows a pipeline the thesis does not run.

   So it is held to the real thing by test. site/tools/extract_facts.py writes
   several hundred raw/cleaned pairs taken from the actual corpus into
   site/data/cleaning.json, and the test suite runs every one of them through
   this file and demands an exact match. If you change a rule here and not
   there — or there and not here — that test fails.

   Rule order matters and is not arbitrary; see the comments below.
   ========================================================================== */

/* Same patterns as export_issues.py, translated where the two dialects differ:
   Python's re.DOTALL becomes [\s\S], and (?m) becomes the m flag. */
const RE_HTML_TAG        = /<[^>]+>/g;
const RE_CODE_BLOCK      = /\{code[^}]*\}[\s\S]*?\{code\}/gi;
const RE_CODE_ORPHAN     = /\{code[^}]*\}/gi;
const RE_NOFORMAT_BLOCK  = /\{noformat[^}]*\}[\s\S]*?\{noformat\}/gi;
const RE_NOFORMAT_ORPHAN = /\{noformat[^}]*\}/gi;
const RE_JIRA_MACRO      = /\{(color|panel|quote|html|anchor|toc|info|note|warning|tip|cloak|section|column)[^}]*\}/gi;
const RE_WIKI_HEADING    = /\bh[1-6]\.\s*/g;
const RE_BULLET          = /^[ \t]*[*#\-]{1,3}[ \t]+/gm;
const RE_BOLD            = /(?<!\w)\*([^*\n]+)\*(?!\w)/g;
const RE_TABLE_HEADER    = /\|\|/g;
const RE_URL             = /https?:\/\/\S+/g;
const RE_ISSUE_KEY       = /\b[A-Z][A-Z0-9]{1,9}-\d+\b/g;
const RE_WS              = /[\r\n\t]+/g;
const RE_MULTISPACE      = / {2,}/g;

/* Python's html.unescape covers the whole HTML5 named-reference table. Issue
   text uses a small corner of it, so the table below is the corner — and the
   vector test is what proves the corner is big enough. */
const ENTITIES = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ",
  ndash: "–", mdash: "—", hellip: "…", copy: "©", reg: "®", trade: "™",
  laquo: "«", raquo: "»", ldquo: "“", rdquo: "”", lsquo: "‘", rsquo: "’",
  bull: "•", middot: "·", deg: "°", plusmn: "±", times: "×", divide: "÷",
  frac12: "½", frac14: "¼", sup2: "²", sup3: "³", micro: "µ", para: "¶",
  sect: "§", dagger: "†", euro: "€", pound: "£", yen: "¥", cent: "¢",
  larr: "←", rarr: "→", harr: "↔", uarr: "↑", darr: "↓",
  ne: "≠", le: "≤", ge: "≥", infin: "∞", radic: "√", asymp: "≈",
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", lambda: "λ", mu: "μ",
  pi: "π", sigma: "σ", tau: "τ", phi: "φ", omega: "ω",
  Agrave: "À", Aacute: "Á", auml: "ä", ouml: "ö", uuml: "ü", szlig: "ß",
  eacute: "é", egrave: "è", agrave: "à", ccedil: "ç", ntilde: "ñ",
  iexcl: "¡", iquest: "¿", brvbar: "¦", uml: "¨", macr: "¯", acute: "´",
  cedil: "¸", ordf: "ª", ordm: "º", not: "¬", shy: "­", curren: "¤",
};

/**
 * Turn &amp;, &#39; and &#x2F; back into characters.
 * Python's html.unescape also decodes references with no trailing semicolon;
 * this does the same for the named ones it knows.
 */
export function unescapeEntities(text) {
  return String(text).replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]{1,31});?/g,
    (whole, body) => {
      if (body[0] === "#") {
        const code = body[1] === "x" || body[1] === "X"
          ? parseInt(body.slice(2), 16)
          : parseInt(body.slice(1), 10);
        if (!Number.isFinite(code) || code < 0 || code > 0x10ffff) return whole;
        // Windows-1252 rescue, exactly as the HTML spec (and Python) do it.
        if (code >= 0x80 && code <= 0x9f) return WINDOWS_1252[code - 0x80] ?? whole;
        if (code === 0 || (code >= 0xd800 && code <= 0xdfff)) return "�";
        try { return String.fromCodePoint(code); } catch { return whole; }
      }
      return Object.prototype.hasOwnProperty.call(ENTITIES, body)
        ? ENTITIES[body]
        : whole;
    });
}

const WINDOWS_1252 = [
  "€", "", "‚", "ƒ", "„", "…", "†", "‡", "ˆ", "‰", "Š", "‹", "Œ",
  "", "Ž", "", "", "‘", "’", "“", "”", "•", "–", "—",
  "˜", "™", "š", "›", "œ", "", "ž", "Ÿ",
];

/**
 * Undo the export's wrapping quotes. Applied once, and only when the field
 * really is wrapped — it is not idempotent, so it must never run twice.
 */
export function decodeExportQuoting(raw) {
  if (typeof raw !== "string") return raw;
  const trimmed = raw.trim();
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    return trimmed.slice(1, -1).replaceAll('""', '"');
  }
  return raw;
}

/**
 * Run the whole pipeline, keeping the text after every stage.
 * Returns [{ id, what, text, changed }], oldest first.
 */
export function cleanStages(raw) {
  const stages = [];
  let text = decodeExportQuoting(String(raw ?? ""));
  const push = (id, what) => {
    const previous = stages.length ? stages[stages.length - 1].text : null;
    stages.push({ id, what, text, changed: previous !== null && previous !== text });
  };

  push("unwrap", "Strip the export's wrapping quotes");

  text = unescapeEntities(text);
  push("entities", "Turn &lt; and friends back into characters");

  text = text.replace(RE_HTML_TAG, " ");
  push("html", "Remove HTML tags");

  // Whole blocks first, then unclosed openers — the other way round would eat
  // the opening marker and leave the block's contents behind.
  text = text.replace(RE_CODE_BLOCK, " [CODE] ");
  text = text.replace(RE_NOFORMAT_BLOCK, " [CODE] ");
  text = text.replace(RE_CODE_ORPHAN, " [CODE] ");
  text = text.replace(RE_NOFORMAT_ORPHAN, " [CODE] ");
  push("code", "Replace pasted code blocks with one marker");

  text = text.replace(RE_JIRA_MACRO, " ");
  push("macros", "Strip known Jira macros, leaving their contents");

  text = text.replace(RE_WIKI_HEADING, "");
  push("headings", "Remove wiki heading markers, keep the words");

  // Bullets before bold: a collapsed list of "* item" lines would otherwise
  // pair its markers into false *bold* matches.
  text = text.replace(RE_BULLET, " ");
  push("bullets", "Remove list markers at the start of lines");

  text = text.replace(RE_BOLD, "$1");
  push("bold", "Unwrap *bold*, leave _italics_ alone");

  text = text.replace(RE_TABLE_HEADER, " ");
  push("tables", "Remove table markers");

  text = text.replace(RE_URL, "[URL]");
  push("urls", "Replace web links with one marker");

  text = text.replace(RE_ISSUE_KEY, "[ISSUE_REF]");
  push("refs", "Replace references to other tickets with one marker");

  text = text.replace(RE_WS, " ").replace(RE_MULTISPACE, " ").trim();
  push("space", "Collapse newlines, tabs and runs of spaces");

  return stages;
}

/** Just the cleaned text. */
export function clean(raw) {
  const stages = cleanStages(raw);
  return stages[stages.length - 1].text;
}

/** How many of each thing the pipeline replaced, for the live counters. */
export function countSubstitutions(raw) {
  const afterHtml = unescapeEntities(decodeExportQuoting(String(raw ?? "")))
    .replace(RE_HTML_TAG, " ");
  const count = (text, re) => (text.match(re) || []).length;

  let text = afterHtml;
  let code = 0;
  code += count(text, RE_CODE_BLOCK);
  text = text.replace(RE_CODE_BLOCK, " [CODE] ");
  code += count(text, RE_NOFORMAT_BLOCK);
  text = text.replace(RE_NOFORMAT_BLOCK, " [CODE] ");
  code += count(text, RE_CODE_ORPHAN);
  text = text.replace(RE_CODE_ORPHAN, " [CODE] ");
  code += count(text, RE_NOFORMAT_ORPHAN);
  text = text.replace(RE_NOFORMAT_ORPHAN, " [CODE] ");

  const prepared = text
    .replace(RE_JIRA_MACRO, " ")
    .replace(RE_WIKI_HEADING, "")
    .replace(RE_BULLET, " ")
    .replace(RE_BOLD, "$1")
    .replace(RE_TABLE_HEADER, " ");

  const urls = count(prepared, RE_URL);
  const refs = count(prepared.replace(RE_URL, "[URL]"), RE_ISSUE_KEY);

  return { code_blocks: code, urls, issue_refs: refs };
}
