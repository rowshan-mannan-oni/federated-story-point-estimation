"""
export_issues.py
----------------
Connects to a MySQL database, identifies qualifying projects using the
canonical filtering query, CLEANS each issue's text and categorical fields,
then exports each project's issues to a separate CSV under data_to_train_on/.

PREPROCESSING HAPPENS HERE, AT EXPORT TIME.
The training pipeline (fl/data.py) consumes these CSVs as-is and must NOT
re-clean. The model-dependent title/description join (e.g. "[SEP]" vs "</s>")
is intentionally NOT done here -- it belongs to the training pipeline because
it depends on the encoder's tokenizer.

Filtering rules (kept consistent in both stages):
  - Story_Point IN (1, 2, 3, 5, 8)
  - Per-project: COUNT(*) >= 500 AND COUNT(DISTINCT Story_Point) >= 4

Cleaning rules (order matters -- see CLAUDE.md "Data & text preprocessing"):
  1. Strip one layer of wrapping double quotes (export artifact)
  2. HTML: unescape entities, then strip tags
  3. Jira markup: {code}/{noformat} blocks -> [CODE]; known macros stripped;
     wiki headings h1.-h6. stripped; *bold* unwrapped; table '||' removed
  4. URLs -> [URL]; Jira issue keys -> [ISSUE_REF]
  5. Whitespace normalisation
  6. Length floor: combined cleaned title+description < 10 chars -> drop row
  7. Priority normalised to {Highest, High, Medium, Low, Lowest, Unknown};
     Type stripped, NaN -> Unknown

Deliberate NON-goals (documented thesis decisions):
  - NO lowercasing, NO stopword removal, NO stemming, NO number stripping.

A per-project + global summary is written to
data_to_train_on/preprocessing_report.json -- the thesis Data chapter cites
these numbers verbatim.

pip install:
    mysql-connector-python pandas python-dotenv

Usage:
    python export_issues.py
"""

import html
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

OUTPUT_DIR = "data_to_train_on"
REPORT_PATH = os.path.join(OUTPUT_DIR, "preprocessing_report.json")

# The exact Fibonacci values treated as valid story points
VALID_STORY_POINTS = (1, 2, 3, 5, 8)

# Project-level thresholds (must match the canonical query)
MIN_ISSUES = 500
MIN_CLASSES = 4

# Minimum combined cleaned text length (title + description, chars)
MIN_TEXT_CHARS = 10

# ---------------------------------------------------------------------------
# Text cleaning (pure functions -- no I/O, individually testable)
# ---------------------------------------------------------------------------

RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_CODE_BLOCK = re.compile(r"\{code[^}]*\}.*?\{code\}", re.DOTALL | re.IGNORECASE)
RE_CODE_ORPHAN = re.compile(r"\{code[^}]*\}", re.IGNORECASE)
RE_NOFORMAT_BLOCK = re.compile(r"\{noformat[^}]*\}.*?\{noformat\}", re.DOTALL | re.IGNORECASE)
RE_NOFORMAT_ORPHAN = re.compile(r"\{noformat[^}]*\}", re.IGNORECASE)
# Known Jira macros only -- deliberately NOT a generic {...} pattern, so that
# inline JSON snippets in issue text are left untouched.
RE_JIRA_MACRO = re.compile(
    r"\{(color|panel|quote|html|anchor|toc|info|note|warning|tip|cloak|section|column)[^}]*\}",
    re.IGNORECASE,
)
RE_WIKI_HEADING = re.compile(r"\bh[1-6]\.\s*")
RE_BULLET = re.compile(r"(?m)^[ \t]*[*#\-]{1,3}[ \t]+")  # Jira list markers: * item / ** sub / # numbered
RE_BOLD = re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)")  # *bold* -> bold; '_italic_' left alone (snake_case risk)
RE_TABLE_HEADER = re.compile(r"\|\|")
RE_URL = re.compile(r"https?://\S+")
RE_ISSUE_KEY = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b")
RE_WS = re.compile(r"[\r\n\t]+")
RE_MULTISPACE = re.compile(r" {2,}")


def decode_export_quoting(raw):
    """
    Decode the LEGACY export artifact: fields wrapped in one layer of double
    quotes, with interior quotes doubled CSV-style ("" -> ").

    Applied ONCE at ingest, only when the field shows evidence of wrapping
    (starts AND ends with a quote). Unwrapped fields pass through untouched,
    which confines any risk to fields that were actually encoded. This step is
    deliberately separate from clean_text_field(), which is idempotent;
    quote-decoding is not, so it must never be re-applied to cleaned text.

    On a fresh export straight from MySQL this is a no-op (values arrive
    unwrapped) -- kept as a defensive measure and for re-processing old CSVs.
    """
    if raw is None or not isinstance(raw, str):
        return raw
    t = raw.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        return t[1:-1].replace('""', '"')
    return raw


def clean_text_field(raw, counters: Counter) -> str:
    """
    Clean a single Title or Description value. Idempotent for ~99.5% of real
    fields; the only re-run effect is removal of stray '*' markup characters
    (Jira bullets not at line start), which is non-corrupting. The pipeline
    contract is single-pass regardless. Quote-decoding is NOT done here --
    see decode_export_quoting().
    Increments `counters` for each substitution type (code_blocks, urls,
    issue_refs) so the export report can cite exact numbers.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw)

    # 1. HTML -- unescape entities first so '&lt;p&gt;' becomes a tag and is removed
    text = html.unescape(text)
    text = RE_HTML_TAG.sub(" ", text)

    # 2. Jira markup
    n_code = 0
    text, k = RE_CODE_BLOCK.subn(" [CODE] ", text);      n_code += k
    text, k = RE_NOFORMAT_BLOCK.subn(" [CODE] ", text);  n_code += k
    text, k = RE_CODE_ORPHAN.subn(" [CODE] ", text);     n_code += k  # unclosed blocks
    text, k = RE_NOFORMAT_ORPHAN.subn(" [CODE] ", text); n_code += k
    counters["code_blocks"] += n_code
    text = RE_JIRA_MACRO.sub(" ", text)
    text = RE_WIKI_HEADING.sub("", text)
    text = RE_BULLET.sub(" ", text)  # list markers removed BEFORE bold, so '* item' never pairs into '*bold*'
    text = RE_BOLD.sub(r"\1", text)
    text = RE_TABLE_HEADER.sub(" ", text)

    # 3. URLs and issue references -> placeholder tokens
    text, k = RE_URL.subn("[URL]", text);        counters["urls"] += k
    text, k = RE_ISSUE_KEY.subn("[ISSUE_REF]", text); counters["issue_refs"] += k

    # 4. Whitespace
    text = RE_WS.sub(" ", text)
    text = RE_MULTISPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Categorical normalisation
# ---------------------------------------------------------------------------

# Cited in the thesis Data chapter -- keep this mapping explicit and visible.
PRIORITY_BASE_MAP = {
    "blocker": "Highest",
    "critical": "Highest",
    "highest": "Highest",
    "urgent": "Highest",
    "major": "High",
    "high": "High",
    "medium": "Medium",
    "normal": "Medium",
    "minor": "Low",
    "low": "Low",
    "trivial": "Lowest",
    "lowest": "Lowest",
}
CANONICAL_PRIORITIES = ("Highest", "High", "Medium", "Low", "Lowest", "Unknown")


def normalize_priority(raw) -> str:
    """
    Map raw Jira priority to {Highest, High, Medium, Low, Lowest, Unknown}.
    Suffixed variants like 'Major - P3' are mapped by their base word.
    NaN and unmapped values -> 'Unknown'.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "Unknown"
    base = str(raw).split("-")[0].strip().lower()
    return PRIORITY_BASE_MAP.get(base, "Unknown")


def normalize_type(raw) -> str:
    """Strip whitespace; NaN -> 'Unknown'. Distinct types are NOT merged."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "Unknown"
    return str(raw).strip()


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_project_df(df: pd.DataFrame, project_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Apply full cleaning to one project's issues.
    Returns (cleaned_df, stats_dict). Cleaned df keeps Title and Description
    as SEPARATE columns -- the model-dependent join happens in the pipeline.
    """
    stats: dict = {"project": project_name, "rows_in": int(len(df))}
    counters: Counter = Counter()

    desc_nan = df["Description"].isna().sum()
    stats["desc_nan_pct"] = round(float(desc_nan) / max(len(df), 1) * 100, 2)

    df = df.copy()
    # decode_export_quoting runs ONCE here on raw values; clean_text_field is idempotent
    df["Title"] = df["Title"].map(lambda v: clean_text_field(decode_export_quoting(v), counters))
    df["Description"] = df["Description"].map(lambda v: clean_text_field(decode_export_quoting(v), counters))
    df["Priority"] = df["Priority"].map(normalize_priority)
    df["Type"] = df["Type"].map(normalize_type)

    # Length floor on combined cleaned text
    combined_len = (df["Title"] + " " + df["Description"]).str.strip().str.len()
    short_mask = combined_len < MIN_TEXT_CHARS
    stats["rows_dropped_short_text"] = int(short_mask.sum())
    df = df[~short_mask].reset_index(drop=True)

    df["Story_Point"] = df["Story_Point"].astype(int)

    stats["rows_out"] = int(len(df))
    stats["substitutions"] = dict(counters)
    stats["class_distribution"] = {
        int(k): int(v) for k, v in df["Story_Point"].value_counts().sort_index().items()
    }
    stats["priority_distribution"] = df["Priority"].value_counts().to_dict()
    return df, stats


# ---------------------------------------------------------------------------
# Database helpers (mysql.connector imported lazily so this module stays
# importable for unit-testing the cleaning functions without a DB driver)
# ---------------------------------------------------------------------------

PROJECT_SUMMARY_QUERY = """
SELECT
    p.Name AS Project_Name,
    COUNT(*) AS Total_Issues,
    COUNT(DISTINCT i.Story_Point) AS Num_Classes
FROM Issue i
JOIN Project p ON i.Project_ID = p.ID
WHERE i.Story_Point IN (1, 2, 3, 5, 8)
GROUP BY p.Name
HAVING COUNT(*) >= 500
  AND COUNT(DISTINCT i.Story_Point) >= 4
ORDER BY Total_Issues DESC;
"""

ISSUE_QUERY = """
SELECT
    i.Issue_Key,
    i.Title,
    i.Description,
    i.Story_Point,
    i.Type,
    i.Priority,
    i.Creation_Date
FROM Issue i
JOIN Project p ON i.Project_ID = p.ID
WHERE p.Name = %s
  AND i.Story_Point IN (1, 2, 3, 5, 8)
ORDER BY i.Creation_Date ASC;
"""


def build_db_config() -> dict:
    """Load DB credentials from .env; fail fast with a clear message."""
    from dotenv import load_dotenv
    load_dotenv()
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "connect_timeout": 10,
    }
    missing = [k for k, v in config.items() if k != "connect_timeout" and v is None]
    if missing:
        print(f"[ERROR] Missing required .env variable(s): "
              f"{', '.join(f'DB_{k.upper()}' for k in missing)}")
        sys.exit(1)
    return config


def connect(config: dict):
    """Open and return a MySQL connection; exit on failure."""
    import mysql.connector
    from mysql.connector import Error
    try:
        conn = mysql.connector.connect(**config)
        if conn.is_connected():
            print(f"[DB] Connected - MySQL server version {conn.server_info}")
            return conn
    except Error as exc:
        print(f"[ERROR] Could not connect to MySQL: {exc}")
        sys.exit(1)


def fetch_qualifying_projects(conn) -> list:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(PROJECT_SUMMARY_QUERY)
        rows = cursor.fetchall()
        print(f"[DB] Found {len(rows)} qualifying project(s).")
        return rows
    finally:
        cursor.close()


def fetch_project_issues(conn, project_name: str) -> pd.DataFrame:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(ISSUE_QUERY, (project_name,))
        return pd.DataFrame(cursor.fetchall())
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    safe = re.sub(r"[\s_]+", "_", safe).strip("_. ")
    return safe or "project"


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    print(f"[IO] Output directory: {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def export_project(conn, project_info: dict, output_dir: str) -> dict | None:
    """Fetch, clean, and write one project's issues. Returns stats dict."""
    project_name = project_info["Project_Name"]
    print(f"\n[EXPORT] Project: '{project_name}' "
          f"| Expected issues: {project_info['Total_Issues']} "
          f"| Classes: {project_info['Num_Classes']}")

    df = fetch_project_issues(conn, project_name)
    if df.empty:
        print(f"  [WARN] No issues returned for '{project_name}' - skipping.")
        return None

    # Defensive: guard against NULL coerced to 0 by the MySQL driver.
    invalid_mask = ~df["Story_Point"].isin(VALID_STORY_POINTS)
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        print(f"  [WARN] Dropping {n_invalid} row(s) with unexpected Story_Point values.")
        df = df[~invalid_mask].reset_index(drop=True)

    df, stats = preprocess_project_df(df, project_name)
    stats["rows_dropped_invalid_sp"] = n_invalid

    safe_name = sanitize_filename(project_name)
    file_path = os.path.join(output_dir, f"{safe_name}.csv")
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"  [OK] Wrote {stats['rows_out']} issues -> {file_path} "
          f"(dropped: {n_invalid} invalid SP, {stats['rows_dropped_short_text']} short text)")
    print(f"       Class distribution: {stats['class_distribution']}")
    return stats


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    conn = connect(build_db_config())
    all_stats = []
    try:
        projects = fetch_qualifying_projects(conn)
        if not projects:
            print("[WARN] No projects satisfy the filtering criteria. Nothing exported.")
            return

        print("\nQualifying projects:")
        for i, p in enumerate(projects, 1):
            print(f"  {i:>2}. {p['Project_Name']}"
                  f" - {p['Total_Issues']} issues, {p['Num_Classes']} classes")

        for project_info in projects:
            stats = export_project(conn, project_info, OUTPUT_DIR)
            if stats:
                all_stats.append(stats)
    finally:
        if conn.is_connected():
            conn.close()
            print("\n[DB] Connection closed.")

    # Global preprocessing report -- the thesis Data chapter cites this file.
    totals = {
        "rows_in": sum(s["rows_in"] for s in all_stats),
        "rows_out": sum(s["rows_out"] for s in all_stats),
        "rows_dropped_short_text": sum(s["rows_dropped_short_text"] for s in all_stats),
        "rows_dropped_invalid_sp": sum(s["rows_dropped_invalid_sp"] for s in all_stats),
        "substitutions": dict(sum((Counter(s["substitutions"]) for s in all_stats), Counter())),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"projects": all_stats, "totals": totals}, fh, indent=2)
    print(f"\n[REPORT] Preprocessing report written to {REPORT_PATH}")
    print(f"[DONE] All projects exported to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()