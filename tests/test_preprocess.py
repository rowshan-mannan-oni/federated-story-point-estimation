"""
tests/test_preprocess.py
------------------------
Unit tests for the pure cleaning functions in export_issues.py.
No database or data directory required.

Run:
    pytest tests/test_preprocess.py
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from export_issues import (
    CANONICAL_PRIORITIES,
    decode_export_quoting,
    clean_text_field,
    normalize_priority,
    preprocess_project_df,
)


# ---------------------------------------------------------------------------
# decode_export_quoting
# ---------------------------------------------------------------------------

class TestDecodeExportQuoting:
    def test_wrapped_plain_text(self):
        assert decode_export_quoting('"HDFS ItemWriter"') == "HDFS ItemWriter"

    def test_wrapped_with_doubled_interior_quotes(self):
        assert (
            decode_export_quoting('"configure ""error channel"" properly"')
            == 'configure "error channel" properly'
        )

    def test_unwrapped_with_interior_quotes_is_unchanged(self):
        val = 'not wrapped "inner" quotes'
        assert decode_export_quoting(val) == val

    def test_none_passes_through(self):
        assert decode_export_quoting(None) is None

    def test_float_nan_passes_through(self):
        val = float("nan")
        result = decode_export_quoting(val)
        # Must come back as the same float NaN (not converted to string)
        assert isinstance(result, float) and math.isnan(result)

    def test_applied_exactly_once_in_preprocess_project_df(self):
        """preprocess_project_df applies decode_export_quoting once per row.
        A title that is already clean must pass through unchanged;
        a wrapped description must be unwrapped exactly once."""
        df = pd.DataFrame({
            "Issue_Key": ["T-1"],
            "Title": ["Already clean title"],
            "Description": ['"wrapped description"'],
            "Story_Point": [3.0],
            "Type": ["Bug"],
            "Priority": ["High"],
            "Creation_Date": ["2023-01-01"],
        })
        cleaned, _ = preprocess_project_df(df, "test")
        assert cleaned.iloc[0]["Title"] == "Already clean title"
        assert cleaned.iloc[0]["Description"] == "wrapped description"


# ---------------------------------------------------------------------------
# clean_text_field
# ---------------------------------------------------------------------------

class TestCleanTextField:
    def _c(self):
        return Counter()

    # --- code blocks ---

    def test_code_block_java(self):
        c = self._c()
        raw = '{code:java}\nSystem.out.println("hello");\n{code}'
        result = clean_text_field(raw, c)
        assert "[CODE]" in result
        assert "{code" not in result.lower()
        assert c["code_blocks"] == 1

    def test_unclosed_code_opener(self):
        c = self._c()
        raw = "See {code:java} for an example"
        result = clean_text_field(raw, c)
        assert "[CODE]" in result
        assert "{code" not in result.lower()
        assert c["code_blocks"] >= 1

    def test_noformat_block(self):
        c = self._c()
        raw = "{noformat}\nsome raw text\n{noformat}"
        result = clean_text_field(raw, c)
        assert "[CODE]" in result
        assert "{noformat" not in result.lower()
        assert c["code_blocks"] >= 1

    # --- HTML ---

    def test_html_entity_and_tag(self):
        c = self._c()
        result = clean_text_field("&lt;p&gt;hello&lt;/p&gt; <b>world</b>", c)
        assert result == "hello world"

    # --- inline JSON survives ---

    def test_inline_json_braces_survive(self):
        """Macros are stripped BY NAME, not by a generic {...} regex.
        This test protects that design decision: inline JSON must be preserved."""
        c = self._c()
        raw = 'sample: {"key": 1} untouched'
        result = clean_text_field(raw, c)
        assert '{"key": 1}' in result

    # --- URLs and issue keys ---

    def test_url_replaced_and_issue_key_replaced(self):
        c = self._c()
        raw = "See https://example.com and PROJ-123 for details"
        result = clean_text_field(raw, c)
        assert "[URL]" in result
        assert "[ISSUE_REF]" in result
        assert "https://" not in result
        assert "PROJ-123" not in result
        assert c["urls"] == 1
        assert c["issue_refs"] == 1

    def test_issue_key_inside_url_becomes_url_token_not_issue_ref(self):
        """URL regex runs before issue-key regex; PROJ-123 inside a URL is
        consumed as part of [URL] and must NOT produce a separate [ISSUE_REF]."""
        c = self._c()
        raw = "See https://jira.example.com/browse/PROJ-123 for context"
        result = clean_text_field(raw, c)
        assert "[URL]" in result
        assert c["urls"] == 1
        assert result.count("[ISSUE_REF]") == 0

    # --- wiki markup ---

    def test_wiki_heading(self):
        c = self._c()
        result = clean_text_field("h2. Steps to reproduce", c)
        assert "Steps to reproduce" in result
        assert "h2." not in result

    def test_line_leading_bullets_removed_no_stray_asterisks(self):
        c = self._c()
        raw = "Steps:\n* first\n* second"
        result = clean_text_field(raw, c)
        assert "first" in result
        assert "second" in result
        # Bullet markers must be gone — no pairing into false *bold* matches
        assert "*first*" not in result
        assert "*second*" not in result

    def test_bold_unwrapped(self):
        c = self._c()
        result = clean_text_field("This is *bold* text", c)
        assert "bold" in result
        assert "*bold*" not in result

    def test_snake_case_identifier_untouched(self):
        c = self._c()
        result = clean_text_field("use snake_case_identifier here", c)
        assert "snake_case_identifier" in result

    def test_italic_underscore_untouched(self):
        c = self._c()
        result = clean_text_field("See _italic_ style", c)
        assert "_italic_" in result

    # --- null inputs ---

    def test_float_nan_returns_empty_string(self):
        assert clean_text_field(float("nan"), self._c()) == ""

    def test_none_returns_empty_string(self):
        assert clean_text_field(None, self._c()) == ""

    # --- idempotence ---

    def test_idempotent_on_code_block(self):
        raw = "{code:java}\nint x = 1;\n{code}"
        once = clean_text_field(raw, self._c())
        assert clean_text_field(once, self._c()) == once

    def test_idempotent_on_url(self):
        raw = "See https://example.com for details"
        once = clean_text_field(raw, self._c())
        assert clean_text_field(once, self._c()) == once

    def test_idempotent_on_html(self):
        raw = "&lt;p&gt;hello&lt;/p&gt; <b>world</b>"
        once = clean_text_field(raw, self._c())
        assert clean_text_field(once, self._c()) == once

    def test_idempotent_on_issue_ref(self):
        raw = "Related to PROJ-123"
        once = clean_text_field(raw, self._c())
        assert clean_text_field(once, self._c()) == once

    def test_idempotent_on_bullet_list(self):
        raw = "Steps:\n* first\n* second"
        once = clean_text_field(raw, self._c())
        assert clean_text_field(once, self._c()) == once

    def test_idempotent_on_wiki_heading(self):
        raw = "h2. Steps to reproduce"
        once = clean_text_field(raw, self._c())
        assert clean_text_field(once, self._c()) == once


# ---------------------------------------------------------------------------
# normalize_priority
# ---------------------------------------------------------------------------

class TestNormalizePriority:
    def test_major_with_suffix(self):
        assert normalize_priority("Major - P3") == "High"

    def test_critical(self):
        assert normalize_priority("Critical") == "Highest"

    def test_minor(self):
        assert normalize_priority("Minor") == "Low"

    def test_trivial(self):
        assert normalize_priority("Trivial") == "Lowest"

    def test_none_returns_unknown(self):
        assert normalize_priority(None) == "Unknown"

    def test_nan_returns_unknown(self):
        assert normalize_priority(float("nan")) == "Unknown"

    def test_unmapped_value_returns_unknown(self):
        assert normalize_priority("SomethingWeird") == "Unknown"

    def test_all_outputs_in_canonical_priorities(self):
        inputs = [
            "Blocker", "Critical", "Highest", "Urgent",
            "Major", "High",
            "Medium", "Normal",
            "Minor", "Low",
            "Trivial", "Lowest",
            None, float("nan"), "SomethingWeird",
        ]
        for inp in inputs:
            result = normalize_priority(inp)
            assert result in CANONICAL_PRIORITIES, (
                f"normalize_priority({inp!r}) = {result!r} not in CANONICAL_PRIORITIES"
            )


# ---------------------------------------------------------------------------
# preprocess_project_df
# ---------------------------------------------------------------------------

def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Issue_Key": ["P-1", "P-2", "P-3", "P-4"],
        "Title": [
            "Fix the login bug",
            "Add retry logic",
            "x",                           # tiny title
            "Update documentation thoroughly",
        ],
        "Description": [
            "Users cannot log in when session expires",
            None,                          # NaN description — must be KEPT
            "",                            # empty + tiny title -> dropped
            "Rewrite all API docs with examples",
        ],
        "Story_Point": [1.0, 3.0, 5.0, 2.0],
        "Type":        ["Bug", "Story", "Task", "Task"],
        "Priority":    ["High", "Medium", "Low", None],
        "Creation_Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
    })


class TestPreprocessProjectDf:
    def test_nan_description_row_is_kept(self):
        df, _ = preprocess_project_df(_make_sample_df(), "test")
        assert "P-2" in df["Issue_Key"].values, "Row with NaN description must be kept"

    def test_short_combined_text_row_is_dropped(self):
        df, stats = preprocess_project_df(_make_sample_df(), "test")
        assert stats["rows_dropped_short_text"] >= 1
        assert "P-3" not in df["Issue_Key"].values, "Row with <10-char combined text must be dropped"

    def test_story_point_is_int_dtype(self):
        df, _ = preprocess_project_df(_make_sample_df(), "test")
        assert df["Story_Point"].dtype in (int, np.dtype("int64"), np.dtype("int32"))

    def test_stats_contains_required_keys(self):
        _, stats = preprocess_project_df(_make_sample_df(), "test")
        for key in ("rows_in", "rows_out", "substitutions", "class_distribution"):
            assert key in stats, f"stats dict missing key: {key!r}"

    def test_stats_rows_in_matches_input_length(self):
        sample = _make_sample_df()
        _, stats = preprocess_project_df(sample, "test")
        assert stats["rows_in"] == len(sample)

    def test_stats_rows_out_matches_output_length(self):
        df, stats = preprocess_project_df(_make_sample_df(), "test")
        assert stats["rows_out"] == len(df)

    def test_class_distribution_keys_are_ints(self):
        _, stats = preprocess_project_df(_make_sample_df(), "test")
        for key in stats["class_distribution"]:
            assert isinstance(key, int), f"class_distribution key {key!r} is not int"

    def test_none_priority_normalised_to_unknown(self):
        df, _ = preprocess_project_df(_make_sample_df(), "test")
        p4 = df[df["Issue_Key"] == "P-4"]
        assert not p4.empty
        assert p4.iloc[0]["Priority"] == "Unknown"
