"""
Leave-one-project-out (LOPO) onboarding experiment (CLAUDE.md gap #14, step 4.2).

Takes a federated artifact trained WITHOUT a held-out project (see
train_federated_dl.py --holdout-project) and measures how quickly that external
project can be onboarded via HEAD-ONLY adaptation on its earliest issues.

For each adaptation budget in {0, 10, 25, 50, 100} (configurable) the head is
reset to its base initialization, fine-tuned (head only; backbone + LoRA-B +
embeddings frozen) on the holdout's earliest `budget` issues, and evaluated on
the holdout's held-out (latest) test split. Budget 0 = zero-shot.

Head initialization:
  - shared-head artifact: the shared head that was trained federated.
  - personalized artifact: the server-side generic head (--head-init generic,
    default) or a fresh random head (--head-init random).

Output: <out-dir>/lopo_<project>.json mapping budget -> metrics (accuracy,
macro_f1, mae, cohen_kappa, per_class_f1), plus run metadata.

Example:
    python run_lopo.py --artifact-dir artifacts/federated --data-dir data_to_train_on \\
        --holdout-project Moodle --head-init generic --out-dir results
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fl.data import IssueDataset, load_dataset_by_project
from fl.metrics import evaluate_classification, format_metrics, run_prediction
from fl.model import StoryPointClassifier, compute_head_loss, is_head_param


def resolve_project_name(available: List[str], name: str) -> str:
    needle = name.lower()
    exact = [c for c in available if c.lower() == needle]
    if exact:
        return exact[0]
    matches = [c for c in available if needle in c.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Holdout project '{name}' is ambiguous. Matches: {matches}")
    raise ValueError(f"Holdout project '{name}' not found. Available: {available}")


def choose_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_collate(tokenizer: AutoTokenizer, max_length: int):
    def collate(examples):
        texts = [e["text"] for e in examples]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc["type_id"] = torch.stack([e["type_id"] for e in examples])
        enc["priority_id"] = torch.stack([e["priority_id"] for e in examples])
        enc["target"] = torch.stack([e["target"] for e in examples])
        return enc

    return collate


def build_model(metadata: Dict) -> StoryPointClassifier:
    return StoryPointClassifier(
        model_name=metadata["model_name"],
        num_types=metadata["num_types"],
        num_priorities=metadata["num_priorities"],
        categorical_emb_dim=metadata["categorical_emb_dim"],
        hidden_dim=metadata["hidden_dim"],
        dropout=metadata["dropout"],
        freeze_encoder=metadata["freeze_encoder"],
        num_classes=metadata["num_classes"],
        use_lora=metadata.get("use_lora", True),
        lora_r=metadata.get("lora_r", 8),
        lora_alpha=metadata.get("lora_alpha", 16),
        lora_dropout=metadata.get("lora_dropout", 0.05),
        lora_target_modules=metadata.get("lora_target_modules", ["query", "value"]),
        ffa_lora=metadata.get("ffa_lora", True),
        head_type=metadata.get("head_type", "ce"),
    )


def head_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if is_head_param(k)}


def load_shared_and_base_head(artifact_dir: Path, metadata: Dict, model: nn.Module, head_init: str):
    """Load the shared representation into `model` and return the base head state to
    reset to before each budget's adaptation."""
    layout = metadata.get("artifact_layout")
    if layout == "personalized":
        shared = torch.load(artifact_dir / "shared_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(shared)
        if head_init == "generic":
            gh = artifact_dir / "generic_head.pt"
            if not gh.exists():
                raise FileNotFoundError(
                    f"--head-init generic but {gh} not found (retrain with --generic-head)."
                )
            base_head = torch.load(gh, map_location="cpu", weights_only=False)
            print(f"[LOPO] Base head: server generic head ({gh.name}).", flush=True)
        else:  # random
            base_head = head_state(build_model(metadata))  # fresh random head
            print("[LOPO] Base head: fresh random head.", flush=True)
    else:
        state = torch.load(artifact_dir / "model_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        base_head = head_state(model)  # the shared head that was trained
        print("[LOPO] Base head: shared trained head.", flush=True)
    return base_head


def adapt_head(model, adapt_frame, type_to_id, priority_to_id, collate, device,
               epochs, lr, head_type, num_classes, seed):
    """Head-only fine-tuning on `adapt_frame` (backbone + LoRA-B + embeddings frozen)."""
    for name, param in model.named_parameters():
        param.requires_grad_(is_head_param(name))
    if len(adapt_frame) == 0 or epochs <= 0:
        return  # zero-shot

    torch.manual_seed(seed)
    dataset = IssueDataset(adapt_frame.reset_index(drop=True), type_to_id, priority_to_id)
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True, collate_fn=collate)
    head_params = [p for n, p in model.named_parameters() if is_head_param(n)]
    optimizer = AdamW(head_params, lr=lr)
    criterion = nn.CrossEntropyLoss()  # plain CE on tiny onboarding sets; CORN uses corn_loss

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = compute_head_loss(head_type, model(batch), batch["target"], num_classes, criterion)
            loss.backward()
            optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-project-out onboarding experiment (gap #14)")
    parser.add_argument("--artifact-dir", type=str, required=True, help="Path to a trained federated artifact")
    parser.add_argument("--data-dir", type=str, required=True, help="Full dataset dir (holdout project must be present)")
    parser.add_argument("--holdout-project", type=str, required=True, help="Project to onboard (must have been held out of training)")
    parser.add_argument("--head-init", type=str, default="generic", choices=["generic", "random"],
                        help="Personalized artifact only: initialize the head from the server generic head or randomly")
    parser.add_argument("--budgets", type=str, default="0,10,25,50,100", help="Comma-separated adaptation budgets")
    parser.add_argument("--test-size", type=float, default=0.2, help="Latest fraction of the holdout reserved as test")
    parser.add_argument("--adapt-epochs", type=int, default=30, help="Head-only adaptation epochs per budget")
    parser.add_argument("--adapt-lr", type=float, default=1e-3, help="Head-only adaptation learning rate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    artifact_dir = Path(args.artifact_dir)
    with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    head_type = metadata.get("head_type", "ce")
    num_classes = metadata["num_classes"]
    type_to_id = metadata["type_to_id"]
    priority_to_id = metadata["priority_to_id"]

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")
    collate = build_collate(tokenizer, metadata["max_length"])
    device = choose_device(args.device)

    # Load the holdout project's rows (already in Creation_Date order: earliest first).
    data = load_dataset_by_project(Path(args.data_dir))
    available = sorted(data["client_id"].unique())
    holdout = resolve_project_name(available, args.holdout_project)
    frame = data[data["client_id"] == holdout].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Holdout project '{holdout}' has no rows.")

    # Reserve the latest test_size fraction as the fixed test set; the earliest rows are
    # the adaptation pool (mirrors the temporal onboarding scenario).
    n = len(frame)
    n_test = max(1, int(round(n * args.test_size)))
    test_frame = frame.tail(n_test).reset_index(drop=True)
    pool = frame.head(n - n_test).reset_index(drop=True)
    print(f"[LOPO] Holdout '{holdout}': {n} rows -> adapt pool {len(pool)}, test {len(test_frame)}", flush=True)

    test_ds = IssueDataset(test_frame, type_to_id, priority_to_id)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_labels = test_ds.labels

    model = build_model(metadata).to(device)
    base_head = load_shared_and_base_head(artifact_dir, metadata, model, args.head_init)

    budgets = [int(b) for b in args.budgets.split(",") if b.strip() != ""]
    condition = "personalized" if metadata.get("artifact_layout") == "personalized" else "shared"
    per_budget: Dict[str, Dict] = {}
    for budget in budgets:
        # Reset the head to its base init so each budget adapts independently.
        model.load_state_dict(base_head, strict=False)
        n_adapt = min(budget, len(pool))
        adapt_frame = pool.head(n_adapt) if n_adapt > 0 else pool.iloc[0:0]
        adapt_head(
            model, adapt_frame, type_to_id, priority_to_id, collate, device,
            args.adapt_epochs, args.adapt_lr, head_type, num_classes, args.seed,
        )
        pred = run_prediction(model, test_loader, device)
        metrics = evaluate_classification(test_labels, pred, num_classes)
        metrics["budget"] = budget
        metrics["n_adapt"] = int(n_adapt)
        metrics["n_test"] = int(len(test_labels))
        per_budget[str(budget)] = metrics
        print(format_metrics(f"budget={budget} (n_adapt={n_adapt})", metrics), flush=True)

    out = {
        "holdout_project": holdout,
        "artifact_dir": str(artifact_dir),
        "condition": condition,
        "head_init": args.head_init if condition == "personalized" else "shared",
        "head_type": head_type,
        "adapt_epochs": args.adapt_epochs,
        "adapt_lr": args.adapt_lr,
        "test_size": args.test_size,
        "n_pool": int(len(pool)),
        "n_test": int(len(test_labels)),
        "budgets": budgets,
        "results": per_budget,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lopo_{holdout}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[LOPO] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
