"""
clean_local_data.py
--------------------
One-off offline cleaner for pre-existing raw CSVs in data_to_train_on/.

These CSVs were exported before the cleaning pipeline in export_issues.py
existed, so they still contain {code}/{noformat} Jira markup and
quote-wrapped Title/Description fields. fl/data.py's validate_cleaned_dataframe
now rejects them.

This script reuses the exact same pure cleaning function
(preprocess_project_df) that export_issues.py applies at export time -- no
MySQL connection required -- and writes cleaned CSVs to a new directory.

Usage:
    python clean_local_data.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from export_issues import VALID_STORY_POINTS, preprocess_project_df

INPUT_DIR = Path("data_to_train_on")
OUTPUT_DIR = Path("data_to_train_on_clean")
REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_stats = []

    for csv_path in sorted(INPUT_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        invalid_mask = ~df["Story_Point"].isin(VALID_STORY_POINTS)
        n_invalid = int(invalid_mask.sum())
        df = df[~invalid_mask].reset_index(drop=True)

        cleaned, stats = preprocess_project_df(df, csv_path.stem)
        stats["rows_dropped_invalid_sp"] = n_invalid

        out_path = OUTPUT_DIR / csv_path.name
        cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")
        all_stats.append(stats)

        print(
            f"[{csv_path.name}] {stats['rows_in']} -> {stats['rows_out']} rows "
            f"(dropped: {n_invalid} invalid SP, {stats['rows_dropped_short_text']} short text)"
        )

    totals = {
        "rows_in": sum(s["rows_in"] for s in all_stats),
        "rows_out": sum(s["rows_out"] for s in all_stats),
        "rows_dropped_short_text": sum(s["rows_dropped_short_text"] for s in all_stats),
        "rows_dropped_invalid_sp": sum(s["rows_dropped_invalid_sp"] for s in all_stats),
        "substitutions": dict(sum((Counter(s["substitutions"]) for s in all_stats), Counter())),
    }
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"projects": all_stats, "totals": totals}, fh, indent=2)

    print(f"\n[DONE] Cleaned CSVs written to '{OUTPUT_DIR}/'.")
    print(f"[REPORT] {REPORT_PATH}")


if __name__ == "__main__":
    main()
