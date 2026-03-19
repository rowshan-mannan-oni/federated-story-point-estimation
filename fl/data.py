import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


COLUMN_CANDIDATES = {
    "title": ["title", "summary", "issue_title"],
    "description": ["description", "description_text", "issue_description"],
    "type": ["type", "issuetype", "issue_type"],
    "priority": ["priority", "issue_priority"],
    "story_point": ["story_point", "story points", "storypoint", "sp"],
}

INPUT_KEYS = ["title", "description", "type", "priority"]


@dataclass
class TabularBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    type_to_id: Dict[str, int]
    priority_to_id: Dict[str, int]


def normalize_col_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


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

    data["title"] = data["title"].fillna("").astype(str)
    data["description"] = data["description"].fillna("").astype(str)
    data["type"] = data["type"].fillna("unknown").astype(str)
    data["priority"] = data["priority"].fillna("unknown").astype(str)

    data["text"] = (data["title"] + " [SEP] " + data["description"]).str.strip()
    data = data[data["text"].str.len() > 0].reset_index(drop=True)

    return data


def split_per_client(data: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in data.groupby("client_id"):
        group = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        if len(group) < 5:
            train_parts.append(group)
            continue

        train_df, test_df = train_test_split(group, test_size=test_size, random_state=random_state)
        train_parts.append(train_df)
        test_parts.append(test_df)

    train_data = pd.concat(train_parts, ignore_index=True)
    test_data = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=data.columns)

    return train_data, test_data


def build_category_map(values: pd.Series) -> Dict[str, int]:
    unique = sorted(set(values.astype(str).tolist()))
    if "unknown" not in unique:
        unique.insert(0, "unknown")
    return {value: idx for idx, value in enumerate(unique)}


def prepare_tabular_bundle(data: pd.DataFrame, test_size: float, random_state: int) -> TabularBundle:
    train_df, test_df = split_per_client(data, test_size=test_size, random_state=random_state)

    type_to_id = build_category_map(train_df["type"])
    priority_to_id = build_category_map(train_df["priority"])

    return TabularBundle(
        train_df=train_df,
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
        use_log_target: bool,
    ) -> None:
        self.text = frame["text"].astype(str).tolist()

        self.type_ids = [type_to_id.get(value, type_to_id["unknown"]) for value in frame["type"].astype(str)]
        self.priority_ids = [
            priority_to_id.get(value, priority_to_id["unknown"]) for value in frame["priority"].astype(str)
        ]

        target = frame["story_point"].to_numpy(dtype=np.float32)
        self.target_raw = target
        if use_log_target:
            target = np.log1p(np.clip(target, a_min=0.0, a_max=None)).astype(np.float32)
        self.target_train = target

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "text": self.text[idx],
            "type_id": torch.tensor(self.type_ids[idx], dtype=torch.long),
            "priority_id": torch.tensor(self.priority_ids[idx], dtype=torch.long),
            "target": torch.tensor(self.target_train[idx], dtype=torch.float32),
        }
