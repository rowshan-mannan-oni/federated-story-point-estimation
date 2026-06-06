import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from fl.data import (
    clean_category_value,
    clean_text_value,
    discover_files,
    find_story_point_column,
    infer_project_id,
    map_input_columns,
    read_table,
)
from fl.data import story_point_to_label
from fl.metrics import evaluate_classification, format_metrics
from fl.model import StoryPointRegressor


def choose_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_inference_table(data_dir: Path) -> pd.DataFrame:
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    frames: List[pd.DataFrame] = []

    for file_path in files:
        raw = read_table(file_path)
        required = map_input_columns(raw)
        story_point_col = find_story_point_column(raw)

        frame = pd.DataFrame(
            {
                "title": raw[required["title"]],
                "description": raw[required["description"]],
                "type": raw[required["type"]],
                "priority": raw[required["priority"]],
                "client_id": infer_project_id(file_path.name),
                "source_file": file_path.name,
            }
        )

        frame["title"] = frame["title"].map(clean_text_value)
        frame["description"] = frame["description"].map(clean_text_value)
        frame["type"] = frame["type"].map(clean_category_value)
        frame["priority"] = frame["priority"].map(clean_category_value)

        if story_point_col is not None:
            frame["story_point"] = pd.to_numeric(raw[story_point_col], errors="coerce")
            frame.loc[frame["story_point"] < 0, "story_point"] = np.nan
        else:
            frame["story_point"] = np.nan

        frame["text"] = (frame["title"] + " [SEP] " + frame["description"]).map(clean_text_value)
        frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


class InferenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, type_to_id: Dict[str, int], priority_to_id: Dict[str, int]) -> None:
        self.text = [clean_text_value(value) for value in frame["text"].tolist()]
        self.type_ids = [
            type_to_id.get(clean_category_value(value), type_to_id["unknown"]) for value in frame["type"].tolist()
        ]
        self.priority_ids = [
            priority_to_id.get(clean_category_value(value), priority_to_id["unknown"])
            for value in frame["priority"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int):
        return {
            "text": self.text[idx],
            "type_id": torch.tensor(self.type_ids[idx], dtype=torch.long),
            "priority_id": torch.tensor(self.priority_ids[idx], dtype=torch.long),
        }


def collate_fn_builder(tokenizer: AutoTokenizer, max_length: int):
    def collate(examples):
        texts = [example["text"] for example in examples]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded["type_id"] = torch.stack([example["type_id"] for example in examples])
        encoded["priority_id"] = torch.stack([example["priority_id"] for example in examples])
        return encoded

    return collate


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict story points using saved artifact")
    parser.add_argument("--artifact-dir", type=str, required=True, help="Path to artifacts/federated or artifacts/centralized")
    parser.add_argument("--data-dir", type=str, required=True, help="New dataset folder to score")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")

    type_to_id: Dict[str, int] = metadata["type_to_id"]
    priority_to_id: Dict[str, int] = metadata["priority_to_id"]

    model = StoryPointRegressor(
        model_name=metadata["model_name"],
        num_types=metadata["num_types"],
        num_priorities=metadata["num_priorities"],
        categorical_emb_dim=metadata["categorical_emb_dim"],
        hidden_dim=metadata["hidden_dim"],
        dropout=metadata["dropout"],
        freeze_encoder=metadata["freeze_encoder"],
        num_classes=metadata["num_classes"],
        use_lora=metadata.get("use_lora", False),
        lora_r=metadata.get("lora_r", 8),
        lora_alpha=metadata.get("lora_alpha", 16),
        lora_dropout=metadata.get("lora_dropout", 0.05),
        lora_target_modules=metadata.get("lora_target_modules", ["query", "value"]),
        ffa_lora=metadata.get("ffa_lora", True),
    )

    state_dict = torch.load(artifact_dir / "model_state.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    device = choose_device(args.device)
    model = model.to(device)
    model.eval()

    data = load_inference_table(Path(args.data_dir))
    dataset = InferenceDataset(data, type_to_id=type_to_id, priority_to_id=priority_to_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_builder(tokenizer, metadata["max_length"]),
    )

    inv_label_map = {int(k): int(v) for k, v in metadata["inv_label_map"].items()}

    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch).detach().cpu()
            predictions.append(logits.argmax(dim=1).numpy())

    pred_labels = np.concatenate(predictions, axis=0)
    pred_story_points = np.array([inv_label_map[int(p)] for p in pred_labels], dtype=np.int64)

    output = data.copy()
    output["predicted_class"] = pred_labels
    output["predicted_story_point"] = pred_story_points
    output_path = Path(args.out_csv)
    output.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")

    # Evaluate only when labels exist in the input data.
    labeled = output.dropna(subset=["story_point"]).copy()
    if not labeled.empty:
        y_true = np.array(
            [story_point_to_label(sp) for sp in labeled["story_point"].tolist()], dtype=np.int64
        )
        y_pred = labeled["predicted_class"].to_numpy(dtype=np.int64)
        metrics = evaluate_classification(y_true, y_pred, num_classes=metadata["num_classes"])
        print(format_metrics("Loaded model eval", metrics))
    else:
        print("No story_point labels found in input data. Skipped evaluation.")


if __name__ == "__main__":
    main()
