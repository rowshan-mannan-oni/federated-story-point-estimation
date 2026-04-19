import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from fl.data import discover_files, find_story_point_column, infer_project_id, map_input_columns, read_table
from fl.metrics import evaluate_regression, format_metrics
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
                "title": raw[required["title"]].fillna("").astype(str),
                "description": raw[required["description"]].fillna("").astype(str),
                "type": raw[required["type"]].fillna("unknown").astype(str),
                "priority": raw[required["priority"]].fillna("unknown").astype(str),
                "client_id": infer_project_id(file_path.name),
                "source_file": file_path.name,
            }
        )

        if story_point_col is not None:
            frame["story_point"] = pd.to_numeric(raw[story_point_col], errors="coerce")
        else:
            frame["story_point"] = np.nan

        frame["text"] = (frame["title"] + " [SEP] " + frame["description"]).str.strip()
        frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


class InferenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, type_to_id: Dict[str, int], priority_to_id: Dict[str, int]) -> None:
        self.text = frame["text"].astype(str).tolist()
        self.type_ids = [type_to_id.get(v, type_to_id["unknown"]) for v in frame["type"].astype(str)]
        self.priority_ids = [priority_to_id.get(v, priority_to_id["unknown"]) for v in frame["priority"].astype(str)]

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

    global_stats = metadata["global_stats"]
    client_stats = metadata.get("client_stats", {})

    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch).detach().cpu().numpy()
            predictions.append(pred)

    pred_norm = np.concatenate(predictions, axis=0)
    pred = np.empty_like(pred_norm, dtype=np.float64)
    for i, (p, cid) in enumerate(zip(pred_norm, data["client_id"].tolist())):
        s = client_stats.get(cid, global_stats)
        pred[i] = p * s["std"] + s["mean"]
    pred = np.clip(pred, a_min=0.0, a_max=None)

    output = data.copy()
    output["predicted_story_point"] = pred
    output_path = Path(args.out_csv)
    output.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")

    # Evaluate only when labels exist in the input data.
    labeled = output.dropna(subset=["story_point"]).copy()
    if not labeled.empty:
        metrics = evaluate_regression(
            labeled["story_point"].to_numpy(dtype=np.float64),
            labeled["predicted_story_point"].to_numpy(dtype=np.float64),
        )
        print(format_metrics("Loaded model eval", metrics))
    else:
        print("No story_point labels found in input data. Skipped evaluation.")


if __name__ == "__main__":
    main()
