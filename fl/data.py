import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Story-point label mapping — single source of truth for the whole codebase.
# ---------------------------------------------------------------------------
STORY_POINT_CLASSES = (1, 2, 3, 5, 8)
LABEL_MAP: Dict[int, int] = {1: 0, 2: 1, 3: 2, 5: 3, 8: 4}
INV_LABEL_MAP: Dict[int, int] = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)


def story_point_to_label(sp: float) -> int:
    """Map a Fibonacci story-point value to a class index (0-4)."""
    return LABEL_MAP[int(sp)]


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss, computed per-client.
    Uses sklearn's balanced formula for present classes; absent classes get weight 0.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    n_present = float((counts > 0).sum())
    n_samples = float(counts.sum())
    if n_present == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(counts > 0, n_samples / (n_present * counts), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


COLUMN_CANDIDATES = {
    "title": ["title", "summary", "issue_title"],
    "description": ["description", "description_text", "issue_description"],
    "type": ["type", "issuetype", "issue_type"],
    "priority": ["priority", "issue_priority"],
    "story_point": ["story_point", "story points", "storypoint", "sp"],
}

INPUT_KEYS = ["title", "description", "type", "priority"]
MISSING_CATEGORY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "unknown"}


@dataclass
class TabularBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    type_to_id: Dict[str, int]
    priority_to_id: Dict[str, int]


def normalize_col_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def clean_text_value(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_category_value(value: object) -> str:
    text = clean_text_value(value).lower()
    if text in MISSING_CATEGORY_TOKENS:
        return "unknown"
    return text


def infer_project_id(file_name: str) -> str:
    match = re.search(r"project[_\- ]?(\d+)", file_name.lower())
    if match:
        return f"project_{match.group(1)}"
    return Path(file_name).stem


def discover_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(sorted(data_dir.rglob(ext)))
    return files


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def validate_cleaned_dataframe(df: pd.DataFrame, source_name: str) -> None:
    """
    Assert that df was produced by export_issues.py and is ready for training.
    Raises ValueError (with file name + pointer to export_issues.py) if any
    raw/uncleaned data is detected. All checks are vectorised; no mutation.
    """
    from export_issues import CANONICAL_PRIORITIES, VALID_STORY_POINTS as _VALID_SP

    def _fail(msg: str) -> None:
        raise ValueError(
            f"[{source_name}] {msg}. Re-export the data via export_issues.py."
        )

    col_lookup = {normalize_col_name(c): c for c in df.columns}
    title_col    = col_lookup.get("title")
    desc_col     = col_lookup.get("description")
    priority_col = col_lookup.get("priority")
    sp_col       = col_lookup.get("story_point")

    # Raw Jira markup that export_issues.py should have replaced with [CODE].
    # Raw/unexported data shows {code}/{noformat} on a few percent of rows
    # (see CLAUDE.md measured noise table), so a small fraction is tolerated
    # as malformed/typo'd macros (e.g. "{noformat)") that the export's regex
    # can't match, rather than evidence the file was never exported.
    for raw_col in filter(None, [title_col, desc_col]):
        series = df[raw_col].dropna().astype(str)
        code_frac = series.str.contains(r"\{code", case=False, regex=True).mean()
        if code_frac > 0.01:
            _fail(f"Column '{raw_col}' has {code_frac:.1%} uncleaned '{{code' Jira markup")
        noformat_frac = series.str.contains(r"\{noformat", case=False, regex=True).mean()
        if noformat_frac > 0.01:
            _fail(f"Column '{raw_col}' has {noformat_frac:.1%} uncleaned '{{noformat' Jira markup")

    # Export-quoting artifact: titles wrapped in literal double quotes.
    # Raw/unexported data shows this on ~100% of titles (see CLAUDE.md measured
    # noise table), so a small fraction is tolerated as legitimate titles that
    # happen to themselves start/end with a quote character (e.g. quoting an
    # error message), rather than leftover export wrapping.
    if title_col is not None:
        titles = df[title_col].dropna().astype(str)
        wrapped_frac = titles.str.match(r'^".*"$').mean()
        if wrapped_frac > 0.01:
            _fail(
                f"Column '{title_col}' has {wrapped_frac:.1%} quote-wrapped values "
                f"(export artifact)"
            )

    # Priority must be one of the canonical values from export_issues.py
    if priority_col is not None:
        bad = ~df[priority_col].isin(CANONICAL_PRIORITIES)
        if bad.any():
            samples = df.loc[bad, priority_col].dropna().unique()[:3].tolist()
            _fail(f"Column '{priority_col}' has non-canonical priorities {samples}")

    # Story_Point must be one of the valid Fibonacci values
    if sp_col is not None:
        sp_series = pd.to_numeric(df[sp_col], errors="coerce").dropna()
        bad_sp = ~sp_series.isin(_VALID_SP)
        if bad_sp.any():
            samples = sp_series[bad_sp].unique()[:3].tolist()
            _fail(f"Column '{sp_col}' has invalid story points {samples}")


def map_input_columns(df: pd.DataFrame) -> Dict[str, str]:
    normalized = {normalize_col_name(col): col for col in df.columns}
    mapped: Dict[str, str] = {}

    for canonical in INPUT_KEYS:
        candidates = COLUMN_CANDIDATES[canonical]
        found = None
        for cand in candidates:
            key = normalize_col_name(cand)
            if key in normalized:
                found = normalized[key]
                break
        if found is None:
            raise ValueError(
                f"Missing required column for '{canonical}'. Available columns: {list(df.columns)}"
            )
        mapped[canonical] = found

    return mapped


def find_story_point_column(df: pd.DataFrame) -> Optional[str]:
    normalized = {normalize_col_name(col): col for col in df.columns}
    for candidate in COLUMN_CANDIDATES["story_point"]:
        key = normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def map_required_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapped = map_input_columns(df)
    story_point_col = find_story_point_column(df)
    if story_point_col is None:
        raise ValueError(
            f"Missing required column for 'story_point'. Available columns: {list(df.columns)}"
        )
    mapped["story_point"] = story_point_col
    return mapped


def load_dataset_by_project(data_dir: Path) -> pd.DataFrame:
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    frames: List[pd.DataFrame] = []
    for file_path in files:
        raw = read_table(file_path)
        validate_cleaned_dataframe(raw, file_path.name)
        col_map = map_required_columns(raw)

        frame = pd.DataFrame(
            {
                "title": raw[col_map["title"]],
                "description": raw[col_map["description"]],
                "type": raw[col_map["type"]],
                "priority": raw[col_map["priority"]],
                "story_point": pd.to_numeric(raw[col_map["story_point"]], errors="coerce"),
                "client_id": infer_project_id(file_path.name),
                "source_file": file_path.name,
            }
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["story_point"]).copy()
    data = data[data["story_point"] >= 0].copy()

    # Filter to valid Fibonacci story points; log drops per project.
    valid_mask = data["story_point"].isin(STORY_POINT_CLASSES)
    if not valid_mask.all():
        dropped_per_project = data[~valid_mask].groupby("client_id").size()
        for project, count in dropped_per_project.items():
            print(
                f"  [Data] {project}: dropped {count} row(s) with Story_Point not in {STORY_POINT_CLASSES}",
                flush=True,
            )
    data = data[valid_mask].copy()

    data["title"] = data["title"].map(clean_text_value)
    data["description"] = data["description"].map(clean_text_value)
    data["type"] = data["type"].map(clean_category_value)
    data["priority"] = data["priority"].map(clean_category_value)

    data["text"] = (data["title"] + " [SEP] " + data["description"]).map(clean_text_value)
    data = data[data["text"].str.len() > 0].reset_index(drop=True)

    return data


def _carve_fraction(
    group: pd.DataFrame,
    fraction: float,
    random_state: int,
    split_mode: str,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one client's rows into (remainder, carved), where `carved` is
    approximately `fraction` of the group.

    split_mode:
      - "random": shuffle the group, then split. If `stratify` is set, the
        carve is stratified by story_point (falling back to unstratified if
        any class has fewer than 2 examples).
      - "temporal": keep row order and carve the latest `fraction` of rows
        (i.e. `remainder` is earlier, `carved` is later).

    Groups with <2 rows are returned unsplit (nothing carved). Groups with
    <5 rows use an index-based slice, keeping at least one row on each side.
    """
    if split_mode == "random":
        group = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        group = group.reset_index(drop=True)

    if len(group) < 2:
        return group, group.iloc[0:0]

    if len(group) < 5:
        carve_count = max(1, int(round(len(group) * fraction)))
        carve_count = min(carve_count, len(group) - 1)
        if split_mode == "temporal":
            remainder = group.iloc[:-carve_count].copy()
            carved = group.iloc[-carve_count:].copy()
        else:
            carved = group.iloc[:carve_count].copy()
            remainder = group.iloc[carve_count:].copy()
        return remainder, carved

    if split_mode == "temporal":
        carve_count = max(1, int(round(len(group) * fraction)))
        split_idx = len(group) - carve_count
        remainder = group.iloc[:split_idx].copy()
        carved = group.iloc[split_idx:].copy()
        return remainder, carved

    stratify_col = group["story_point"] if stratify else None
    if stratify_col is not None and (stratify_col.value_counts() < 2).any():
        stratify_col = None
    try:
        remainder, carved = train_test_split(
            group, test_size=fraction, random_state=random_state, stratify=stratify_col
        )
    except ValueError:
        # Stratification can fail even when every class has >=2 examples, if the
        # carved fraction is smaller than the number of classes. Fall back to an
        # unstratified split rather than dropping the val carve entirely.
        remainder, carved = train_test_split(
            group, test_size=fraction, random_state=random_state, stratify=None
        )
    return remainder, carved


def split_per_client(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
    split_mode: str = "random",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    split_mode:
      - "random": shuffle each client's rows, then split (default).
      - "temporal": keep export row order (export_issues.py orders by
        Creation_Date ascending, and the loading pipeline preserves row
        order) and hold out each client's latest `test_size` fraction as
        test. Train on earlier issues, test on later ones.
    """
    if split_mode not in ("random", "temporal"):
        raise ValueError(f"Unknown split_mode '{split_mode}'. Expected 'random' or 'temporal'.")

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in data.groupby("client_id"):
        remainder, carved = _carve_fraction(group, test_size, random_state, split_mode)
        train_parts.append(remainder)
        if len(carved):
            test_parts.append(carved)

    train_data = pd.concat(train_parts, ignore_index=True)
    test_data = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=data.columns)

    return train_data, test_data


def split_train_val(
    train_data: pd.DataFrame,
    val_size: float,
    random_state: int,
    split_mode: str = "random",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve a per-client validation split out of the train portion returned by
    split_per_client, using the same edge-case handling. In "random" mode the
    carve is stratified by story_point so rare classes still appear in val
    for per-round macro-F1 tracking; in "temporal" mode val is the latest
    `val_size` slice of each client's train rows (chronologically between
    train and test).
    """
    if val_size <= 0:
        return train_data, train_data.iloc[0:0]

    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []

    for _, group in train_data.groupby("client_id"):
        remainder, carved = _carve_fraction(
            group, val_size, random_state, split_mode, stratify=(split_mode == "random")
        )
        train_parts.append(remainder)
        if len(carved):
            val_parts.append(carved)

    train_out = pd.concat(train_parts, ignore_index=True)
    val_out = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=train_data.columns)

    return train_out, val_out


def build_category_map(values: pd.Series) -> Dict[str, int]:
    unique = sorted({clean_category_value(value) for value in values.tolist()})
    if "unknown" not in unique:
        unique.insert(0, "unknown")
    return {value: idx for idx, value in enumerate(unique)}


def prepare_tabular_bundle(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
    split_mode: str = "random",
    val_size: float = 0.0,
) -> TabularBundle:
    train_df, test_df = split_per_client(data, test_size=test_size, random_state=random_state, split_mode=split_mode)
    train_df, val_df = split_train_val(train_df, val_size=val_size, random_state=random_state, split_mode=split_mode)

    type_to_id = build_category_map(train_df["type"])
    priority_to_id = build_category_map(train_df["priority"])

    return TabularBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        type_to_id=type_to_id,
        priority_to_id=priority_to_id,
    )


class IssueDataset(Dataset):
    """Dataset object used by centralized and federated trainers."""

    def __init__(
        self,
        frame: pd.DataFrame,
        type_to_id: Dict[str, int],
        priority_to_id: Dict[str, int],
    ) -> None:
        self.text = [clean_text_value(value) for value in frame["text"].tolist()]

        self.type_ids = [
            type_to_id.get(clean_category_value(value), type_to_id["unknown"]) for value in frame["type"].tolist()
        ]
        self.priority_ids = [
            priority_to_id.get(clean_category_value(value), priority_to_id["unknown"])
            for value in frame["priority"].tolist()
        ]

        self.labels = np.array(
            [story_point_to_label(sp) for sp in frame["story_point"].tolist()],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "text": self.text[idx],
            "type_id": torch.tensor(self.type_ids[idx], dtype=torch.long),
            "priority_id": torch.tensor(self.priority_ids[idx], dtype=torch.long),
            "target": torch.tensor(self.labels[idx], dtype=torch.long),
        }
