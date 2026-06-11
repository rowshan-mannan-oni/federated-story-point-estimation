"""
tests/test_data_validation.py
------------------------------
Tests for fl.data.validate_cleaned_dataframe.
No database or data directory required for the fast tests.

Run:
    pytest -m "not slow" tests/test_data_validation.py
    pytest tests/test_data_validation.py          # includes slow integration test
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from fl.data import validate_cleaned_dataframe


# ---------------------------------------------------------------------------
# Clean reference fixture
# ---------------------------------------------------------------------------

def _clean_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Issue_Key":     ["A-1", "A-2", "A-3"],
        "Title":         ["Fix login bug", "Add retry logic", "Update docs"],
        "Description":   ["Users cannot log in", "Use exponential backoff", ""],
        "Story_Point":   [1, 3, 5],
        "Type":          ["Bug", "Story", "Task"],
        "Priority":      ["High", "Medium", "Lowest"],
        "Creation_Date": ["2023-01-01", "2023-02-01", "2023-03-01"],
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_clean_dataframe_passes():
    validate_cleaned_dataframe(_clean_df(), "clean.csv")  # must not raise


def test_nan_description_is_accepted():
    df = _clean_df()
    df.loc[0, "Description"] = None
    validate_cleaned_dataframe(df, "clean.csv")  # NaN description is fine


# ---------------------------------------------------------------------------
# Violation: raw Jira markup
# ---------------------------------------------------------------------------

def test_code_markup_in_title_raises():
    df = _clean_df()
    df.loc[0, "Title"] = "{code:java}int x = 1;{code}"
    with pytest.raises(ValueError, match="code"):
        validate_cleaned_dataframe(df, "dirty.csv")


def test_code_markup_in_description_raises():
    df = _clean_df()
    df.loc[0, "Description"] = "{code}\nsome stack trace\n{code}"
    with pytest.raises(ValueError, match="code"):
        validate_cleaned_dataframe(df, "dirty.csv")


def test_noformat_in_description_raises():
    df = _clean_df()
    df.loc[0, "Description"] = "{noformat}raw text{noformat}"
    with pytest.raises(ValueError, match="noformat"):
        validate_cleaned_dataframe(df, "dirty.csv")


def test_noformat_in_title_raises():
    df = _clean_df()
    df.loc[0, "Title"] = "{noformat}snippet{noformat}"
    with pytest.raises(ValueError, match="noformat"):
        validate_cleaned_dataframe(df, "dirty.csv")


# ---------------------------------------------------------------------------
# Violation: quote-wrapped title
# ---------------------------------------------------------------------------

def test_wrapped_title_raises():
    df = _clean_df()
    df.loc[0, "Title"] = '"Report executor terminations to framework schedulers."'
    with pytest.raises(ValueError, match="quote-wrapped"):
        validate_cleaned_dataframe(df, "dirty.csv")


def test_unwrapped_title_with_interior_quotes_does_not_raise():
    df = _clean_df()
    df.loc[0, "Title"] = "Fix the \"important\" bug"
    validate_cleaned_dataframe(df, "clean.csv")  # interior quotes are fine


# ---------------------------------------------------------------------------
# Violation: non-canonical priority
# ---------------------------------------------------------------------------

def test_raw_priority_minor_raises():
    df = _clean_df()
    df.loc[0, "Priority"] = "Minor"  # should have been normalised to "Low"
    with pytest.raises(ValueError, match="non-canonical"):
        validate_cleaned_dataframe(df, "dirty.csv")


def test_raw_priority_major_raises():
    df = _clean_df()
    df.loc[0, "Priority"] = "Major"
    with pytest.raises(ValueError, match="non-canonical"):
        validate_cleaned_dataframe(df, "dirty.csv")


def test_nan_priority_raises():
    df = _clean_df()
    df.loc[0, "Priority"] = None  # should have been normalised to "Unknown"
    with pytest.raises(ValueError, match="non-canonical"):
        validate_cleaned_dataframe(df, "dirty.csv")


# ---------------------------------------------------------------------------
# Violation: invalid story point
# ---------------------------------------------------------------------------

def test_invalid_story_point_4_raises():
    df = _clean_df()
    df.loc[0, "Story_Point"] = 4
    with pytest.raises(ValueError, match="invalid story"):
        validate_cleaned_dataframe(df, "dirty.csv")


def test_invalid_story_point_13_raises():
    df = _clean_df()
    df.loc[0, "Story_Point"] = 13
    with pytest.raises(ValueError, match="invalid story"):
        validate_cleaned_dataframe(df, "dirty.csv")


# ---------------------------------------------------------------------------
# Error messages are informative
# ---------------------------------------------------------------------------

def test_error_message_includes_source_name():
    df = _clean_df()
    df.loc[0, "Priority"] = "Minor"
    with pytest.raises(ValueError, match="myproject.csv"):
        validate_cleaned_dataframe(df, "myproject.csv")


def test_error_message_mentions_export_issues():
    df = _clean_df()
    df.loc[0, "Story_Point"] = 4
    with pytest.raises(ValueError, match="export_issues"):
        validate_cleaned_dataframe(df, "any.csv")


# ---------------------------------------------------------------------------
# Integration test (requires data_to_train_on/ + preprocessing_report.json)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_csvs_pass_validation_and_total_matches_report():
    """Load every real project CSV through the validating loader and assert
    that the total row count matches preprocessing_report.json totals.rows_out.
    Skips cleanly when data_to_train_on/ or its report are absent."""
    data_dir = Path("data_to_train_on")
    report_path = data_dir / "preprocessing_report.json"

    if not data_dir.exists():
        pytest.skip("data_to_train_on/ not present")
    if not report_path.exists():
        pytest.skip(
            "preprocessing_report.json absent — re-export via export_issues.py first"
        )

    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)
    expected_total = report["totals"]["rows_out"]

    total_loaded = 0
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        validate_cleaned_dataframe(df, csv_path.name)  # raises if uncleaned
        total_loaded += len(df)

    assert total_loaded == expected_total, (
        f"Loaded {total_loaded} rows but preprocessing_report.json reports {expected_total}"
    )
