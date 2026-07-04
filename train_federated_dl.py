import argparse
import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

# Must be set before CUDA is initialized (i.e. before torch is imported) to take
# effect. Reduces allocator fragmentation across the many client trainings in a
# federated run, which can otherwise cause spurious OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fl import checkpoint as ck
from fl.client import FederatedClient
from fl.config import FLConfig
from sklearn.model_selection import train_test_split

from fl.data import IssueDataset, compute_class_weights, load_dataset_by_project, prepare_tabular_bundle, LABEL_MAP, INV_LABEL_MAP
from fl.metrics import compute_communication_cost, evaluate_classification, format_metrics, run_prediction
from fl.model import StoryPointClassifier, compute_head_loss, log_trainable_params
from fl.server import FedProxServer


def choose_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def federated_condition_name(prox_mu: float, personalized_head: bool = False) -> str:
    """
    Human-readable label for the federated condition, derived from prox_mu.
    prox_mu=0 degrades FedProx to plain FedAvg (CLAUDE.md gap #2) — name it
    explicitly so results/config.json and printed output don't require the
    reader to know this equivalence. In personalized-head mode (gap #11) a
    " + P-head" suffix is appended (e.g. "FedProx (mu=0.01) + P-head").
    """
    base = "FedAvg" if prox_mu == 0.0 else f"FedProx (mu={prox_mu:g})"
    if personalized_head:
        base += " + P-head"
    return base


def collate_fn_builder(tokenizer: AutoTokenizer, max_length: int):
    def collate(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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
        encoded["target"] = torch.stack([example["target"] for example in examples])
        return encoded

    return collate


def train_centralized(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    log_every: int = 1,
    ckpt_root: Optional[Path] = None,
    checkpoint_every: int = 0,
    checkpoint_keep: int = 2,
    checkpoint_config: Optional[FLConfig] = None,
    resume_payload: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # CE head only; CORN ignores weights
    head_type = getattr(model, "head_type", "ce")
    num_classes = getattr(model, "num_classes", class_weights.numel())

    # Trainable-only keys — the frozen backbone is never checkpointed (gap #13).
    trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]

    # Resume: restore trainable params, optimizer state, epoch index, and RNG.
    start_epoch = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"], strict=False)
        optimizer.load_state_dict(resume_payload["optimizer"])
        for opt_state in optimizer.state.values():
            for sk, sv in opt_state.items():
                if isinstance(sv, torch.Tensor):
                    opt_state[sk] = sv.to(device)
        start_epoch = int(resume_payload["epoch"])
        ck.restore_rng_state(resume_payload["rng"])
        print(f"[Centralized] Resumed from epoch {start_epoch}; continuing to {epochs}.", flush=True)

    print(
        f"[Centralized] Starting training: epochs={epochs}, batches_per_epoch={len(train_loader)}",
        flush=True,
    )

    train_start = time.perf_counter()

    for epoch_idx in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = compute_head_loss(head_type, pred, batch["target"], num_classes, criterion)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

        done = epoch_idx + 1
        if done % max(log_every, 1) == 0:
            avg_epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
            epoch_elapsed = time.perf_counter() - epoch_start
            total_elapsed = time.perf_counter() - train_start
            avg_epoch_time = total_elapsed / max(done, 1)
            eta_seconds = avg_epoch_time * max(epochs - done, 0)

            print(
                f"[Centralized][Epoch {done}/{epochs}] "
                f"avg_loss={avg_epoch_loss:.6f} "
                f"epoch_time={epoch_elapsed:.1f}s "
                f"elapsed={total_elapsed:.1f}s "
                f"eta={eta_seconds:.1f}s",
                flush=True,
            )

        # Per-epoch checkpoint (gap #13): trainable state + optimizer + epoch + RNG.
        if checkpoint_every > 0 and ckpt_root is not None and (
            done % checkpoint_every == 0 or done == epochs
        ):
            full_state = model.state_dict()
            payload: Dict[str, Any] = {
                "epoch": done,
                "model_state": {k: full_state[k].detach().cpu().clone() for k in trainable_keys},
                "optimizer": optimizer.state_dict(),
                "rng": ck.capture_rng_state(),
            }
            ck.save_checkpoint(
                ckpt_root, done, payload, checkpoint_config,
                is_best=False, keep=checkpoint_keep, step_prefix="epoch",
            )
            print(f"[Centralized][Checkpoint] saved epoch {done} to {ckpt_root}", flush=True)

    return model


def train_warmstart(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_labels: np.ndarray,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    num_classes: int,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Pretrain on a single project with early stopping on val macro-F1.
    Returns (best_state_dict, best_val_macro_f1).
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # CE head only; CORN ignores weights
    head_type = getattr(model, "head_type", "ce")

    best_macro_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    epochs_no_improve = 0

    print(
        f"[Warmstart] Starting: max_epochs={max_epochs}, patience={patience}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}",
        flush=True,
    )
    ws_start = time.perf_counter()

    for epoch_idx in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = compute_head_loss(head_type, model(batch), batch["target"], num_classes, criterion)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        val_pred = run_prediction(model, val_loader, device)
        val_metrics = evaluate_classification(val_labels, val_pred, num_classes)
        macro_f1 = val_metrics["macro_f1"]
        improved = macro_f1 > best_macro_f1

        print(
            f"[Warmstart][Epoch {epoch_idx + 1}/{max_epochs}] "
            f"loss={epoch_loss / max(n_batches, 1):.4f}  "
            f"val_macro_f1={macro_f1:.4f}  "
            f"elapsed={time.perf_counter() - ws_start:.1f}s"
            + ("  *best*" if improved else f"  (no improve {epochs_no_improve + 1}/{patience})"),
            flush=True,
        )

        if improved:
            best_macro_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Warmstart] Early stopping at epoch {epoch_idx + 1}.", flush=True)
                break

    model.load_state_dict(best_state)
    return best_state, best_macro_f1


def _artifact_metadata(
    config: FLConfig,
    type_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
) -> Dict[str, Any]:
    """Model-reconstruction metadata shared by shared-head and personalized artifacts."""
    return {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "num_classes": config.num_classes,
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "inv_label_map": {str(k): v for k, v in INV_LABEL_MAP.items()},
        "categorical_emb_dim": config.categorical_emb_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "freeze_encoder": config.freeze_encoder,
        "use_lora": config.use_lora,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": list(config.lora_target_modules),
        "ffa_lora": config.ffa_lora,
        "head_type": config.head_type,
        "personalized_head": config.personalized_head,
        "prox_mu": config.prox_mu,
        "num_types": len(type_to_id),
        "num_priorities": len(priority_to_id),
        "type_to_id": type_to_id,
        "priority_to_id": priority_to_id,
    }


def save_model_artifact(
    save_root: Path,
    artifact_name: str,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    config: FLConfig,
    type_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
) -> Path:
    artifact_dir = save_root / artifact_name
    tokenizer_dir = artifact_dir / "tokenizer"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), artifact_dir / "model_state.pt")
    tokenizer.save_pretrained(tokenizer_dir)

    metadata = _artifact_metadata(config, type_to_id, priority_to_id)

    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return artifact_dir


def compute_generic_head(
    client_heads: Dict[str, Dict[str, torch.Tensor]],
    client_sizes: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """One-way weighted average of the per-client heads (weight = client train size).

    Used ONLY to initialize new/external clients (LOPO onboarding); never pushed back
    to participants. Weights come from each client's own example count.
    """
    total = float(sum(client_sizes.get(cid, 0) for cid in client_heads)) or 1.0
    generic: Dict[str, torch.Tensor] = {}
    for cid, head in client_heads.items():
        weight = client_sizes.get(cid, 0) / total
        for key, tensor in head.items():
            contribution = tensor.detach().cpu() * weight
            generic[key] = contribution if key not in generic else generic[key] + contribution
    return generic


def save_personalized_federated_artifact(
    save_root: Path,
    artifact_name: str,
    shared_state: Dict[str, torch.Tensor],
    client_heads: Dict[str, Dict[str, torch.Tensor]],
    tokenizer: AutoTokenizer,
    config: FLConfig,
    type_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
    generic_head: Optional[Dict[str, torch.Tensor]] = None,
) -> Path:
    """
    Personalized-head artifact layout (gap #11):
        <artifact_name>/shared_state.pt      full loadable state (shared repr + placeholder head)
        <artifact_name>/heads/<project>.pt   per-client head params, overlaid at inference
        <artifact_name>/generic_head.pt      optional one-way head average (onboarding only)
        <artifact_name>/{tokenizer, metadata.json}
    """
    artifact_dir = save_root / artifact_name
    heads_dir = artifact_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    torch.save(shared_state, artifact_dir / "shared_state.pt")
    for cid, head in client_heads.items():
        torch.save(head, heads_dir / f"{cid}.pt")
    if generic_head is not None:
        torch.save(generic_head, artifact_dir / "generic_head.pt")
    tokenizer.save_pretrained(artifact_dir / "tokenizer")

    metadata = _artifact_metadata(config, type_to_id, priority_to_id)
    metadata["artifact_layout"] = "personalized"
    metadata["client_ids"] = sorted(client_heads)
    metadata["has_generic_head"] = generic_head is not None
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return artifact_dir


def evaluate_per_project(
    model: nn.Module,
    test_df: pd.DataFrame,
    type_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
    collate_fn,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs inference per client and returns a dict keyed by client_id plus "global".
    Each value is evaluate_classification output + n_test.
    """
    results: Dict[str, Dict[str, Any]] = {}
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for client_id, group in test_df.groupby("client_id"):
        dataset = IssueDataset(group.reset_index(drop=True), type_to_id, priority_to_id)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        pred = run_prediction(model, loader, device)
        true = dataset.labels

        metrics = evaluate_classification(true, pred, num_classes)
        metrics["n_test"] = int(len(true))
        results[str(client_id)] = metrics
        all_true.append(true)
        all_pred.append(pred)

    global_true = np.concatenate(all_true)
    global_pred = np.concatenate(all_pred)
    global_metrics = evaluate_classification(global_true, global_pred, num_classes)
    global_metrics["n_test"] = int(len(global_true))
    results["global"] = global_metrics
    return results


def evaluate_per_project_personalized(
    shared_state: Dict[str, torch.Tensor],
    client_heads: Dict[str, Dict[str, torch.Tensor]],
    model_factory,
    test_df: pd.DataFrame,
    type_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
    collate_fn,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Personalized-head test evaluation: each client is evaluated with the shared
    representation plus ITS OWN head on ITS OWN test split. There is NO pooled
    "global" entry — a pooled metric is undefined when every client uses a
    different head (report per-project + mean/median instead).
    """
    results: Dict[str, Dict[str, Any]] = {}
    for client_id, group in test_df.groupby("client_id"):
        cid = str(client_id)
        model = model_factory().to(device)
        state = dict(shared_state)
        if cid in client_heads:
            state.update(client_heads[cid])
        model.load_state_dict(state)
        model.eval()

        dataset = IssueDataset(group.reset_index(drop=True), type_to_id, priority_to_id)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        pred = run_prediction(model, loader, device)

        metrics = evaluate_classification(dataset.labels, pred, num_classes)
        metrics["n_test"] = int(len(dataset.labels))
        results[cid] = metrics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def mean_metrics_across_projects(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-project metrics into an unweighted mean across projects (used
    for printouts/comparisons in personalized mode, where no pooled-global entry
    exists). Excludes any "global" key if present.
    """
    pids = [k for k in results if k != "global"]
    per_class = np.mean([results[k]["per_class_f1"] for k in pids], axis=0).tolist()
    return {
        "accuracy": float(np.mean([results[k]["accuracy"] for k in pids])),
        "macro_f1": float(np.mean([results[k]["macro_f1"] for k in pids])),
        "per_class_f1": per_class,
        "mae": float(np.mean([results[k]["mae"] for k in pids])),
        "cohen_kappa": float(np.mean([results[k]["cohen_kappa"] for k in pids])),
        "n_test": int(sum(results[k]["n_test"] for k in pids)),
    }


def train_local_only(
    clients: List[FederatedClient],
    model_factory,
    initial_state: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    random_state: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Train each client independently on its own data only — no server, no aggregation.
    Every client starts from the same `initial_state` (the warm-start checkpoint, same
    as the federated run) so the only difference vs. federated is the absence of
    cross-client aggregation. Returns client_id -> full state dict.
    """
    local_states: Dict[str, Dict[str, torch.Tensor]] = {}

    print(f"\n[LocalOnly] Training {len(clients)} clients independently for {epochs} epoch(s) each ...", flush=True)
    start = time.perf_counter()

    for i, client in enumerate(clients):
        result = client.train_local(
            global_state=initial_state,
            model_factory=model_factory,
            device=device,
            epochs=epochs,
            sample_ratio_per_epoch=1.0,
            sample_with_replacement=False,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            prox_mu=0.0,
            seed=random_state + i,
        )
        # Trainable params come from local training; frozen backbone is shared with initial_state.
        local_states[str(client.client_id)] = {**initial_state, **result.state_dict}
        print(
            f"  - client={client.client_id} examples={result.num_examples} final_loss={result.loss:.6f}",
            flush=True,
        )

    print(f"[LocalOnly] Done in {time.perf_counter() - start:.1f}s", flush=True)
    return local_states


def evaluate_local_only(
    local_states: Dict[str, Dict[str, torch.Tensor]],
    model_factory,
    test_df: pd.DataFrame,
    type_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
    collate_fn,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Each client's locally-trained model is evaluated on that client's own test split.
    Returns per-client metrics (evaluate_classification output + n_test) plus a
    "global" entry pooling every client's predictions on its own test set.
    """
    results: Dict[str, Dict[str, Any]] = {}
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for client_id, group in test_df.groupby("client_id"):
        client_id_str = str(client_id)
        model = model_factory().to(device)
        model.load_state_dict(local_states[client_id_str])
        model.eval()

        dataset = IssueDataset(group.reset_index(drop=True), type_to_id, priority_to_id)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        pred = run_prediction(model, loader, device)
        true = dataset.labels

        metrics = evaluate_classification(true, pred, num_classes)
        metrics["n_test"] = int(len(true))
        results[client_id_str] = metrics
        all_true.append(true)
        all_pred.append(pred)

        del model

    global_true = np.concatenate(all_true)
    global_pred = np.concatenate(all_pred)
    global_metrics = evaluate_classification(global_true, global_pred, num_classes)
    global_metrics["n_test"] = int(len(global_true))
    results["global"] = global_metrics
    return results


def build_summary_df(
    fed_results: Dict[str, Dict[str, Any]],
    central_results: Optional[Dict[str, Dict[str, Any]]],
    local_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    sp_labels = [1, 2, 3, 5, 8]
    # Personalized mode has no pooled "global" entry; append it only if present.
    project_ids = sorted(k for k in fed_results if k != "global")
    if "global" in fed_results:
        project_ids = project_ids + ["global"]
    rows = []
    for pid in project_ids:
        fed = fed_results[pid]
        row: Dict[str, Any] = {
            "project": pid,
            "n_test": fed["n_test"],
            "fed_acc": round(fed["accuracy"], 4),
            "fed_macro_f1": round(fed["macro_f1"], 4),
        }
        for i, sp in enumerate(sp_labels):
            row[f"fed_f1_sp{sp}"] = round(fed["per_class_f1"][i], 4)
        if central_results is not None:
            cen = central_results[pid]
            row["cen_acc"] = round(cen["accuracy"], 4)
            row["cen_macro_f1"] = round(cen["macro_f1"], 4)
            for i, sp in enumerate(sp_labels):
                row[f"cen_f1_sp{sp}"] = round(cen["per_class_f1"][i], 4)
        if local_results is not None:
            loc = local_results[pid]
            row["local_acc"] = round(loc["accuracy"], 4)
            row["local_macro_f1"] = round(loc["macro_f1"], 4)
            for i, sp in enumerate(sp_labels):
                row[f"local_f1_sp{sp}"] = round(loc["per_class_f1"][i], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def save_evaluation_results(
    results_dir: Path,
    fed_results: Dict[str, Dict[str, Any]],
    central_results: Optional[Dict[str, Dict[str, Any]]],
    summary_df: pd.DataFrame,
    config: FLConfig,
    local_results: Optional[Dict[str, Dict[str, Any]]] = None,
    fed_history: Optional[List[Dict[str, Any]]] = None,
    communication_cost: Optional[Dict[str, Any]] = None,
    federated_condition: Optional[str] = None,
    test_df: Optional[pd.DataFrame] = None,
    no_warmstart_results: Optional[Dict[str, Dict[str, Any]]] = None,
    no_warmstart_history: Optional[List[Dict[str, Any]]] = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    if test_df is not None:
        test_df.to_csv(results_dir / "test_split.csv", index=False)

    with (results_dir / "federated_per_project.json").open("w", encoding="utf-8") as fh:
        json.dump(fed_results, fh, indent=2)
    if central_results is not None:
        with (results_dir / "centralized_per_project.json").open("w", encoding="utf-8") as fh:
            json.dump(central_results, fh, indent=2)
    if local_results is not None:
        with (results_dir / "local_only_per_project.json").open("w", encoding="utf-8") as fh:
            json.dump(local_results, fh, indent=2)
    if fed_history is not None:
        with (results_dir / "federated_round_history.json").open("w", encoding="utf-8") as fh:
            json.dump(fed_history, fh, indent=2)
    if communication_cost is not None:
        with (results_dir / "communication_cost.json").open("w", encoding="utf-8") as fh:
            json.dump(communication_cost, fh, indent=2)
    if no_warmstart_results is not None:
        with (results_dir / "federated_no_warmstart_per_project.json").open("w", encoding="utf-8") as fh:
            json.dump(no_warmstart_results, fh, indent=2)
    if no_warmstart_history is not None:
        with (results_dir / "federated_no_warmstart_round_history.json").open("w", encoding="utf-8") as fh:
            json.dump(no_warmstart_history, fh, indent=2)

    summary_df.to_csv(results_dir / "summary.csv", index=False)
    summary_md = _df_to_markdown(summary_df)
    if federated_condition is not None:
        summary_md = f"**Federated condition:** {federated_condition} (prox_mu={config.prox_mu:g})\n\n{summary_md}"
    (results_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    config_dict = dataclasses.asdict(config)
    config_dict["data_dir"] = str(config_dict["data_dir"])
    if federated_condition is not None:
        config_dict["federated_condition"] = federated_condition
    with (results_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_dict, fh, indent=2)

    print(f"[Eval] Results saved to {results_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Deep Learning for Story Point Estimation")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--split-mode", type=str, default="random", choices=["random", "temporal"])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prox-mu", type=float, default=1e-2)
    parser.add_argument("--head-type", type=str, default="ce", choices=["ce", "corn"],
                        help="Classification head/loss: 'ce' (softmax + CrossEntropy) or 'corn' (ordinal CORN head + loss)")
    parser.add_argument("--personalized-head", action="store_true",
                        help="Keep the classification head local per client (FedSP-PEFT-P); only LoRA-B + embeddings are aggregated")
    parser.add_argument("--generic-head", action="store_true",
                        help="Personalized mode only: also save a one-way weighted average of client heads as generic_head.pt (for new-client onboarding); never pushed to participants")
    parser.add_argument("--clients-per-round-fraction", "--fraction", dest="clients_per_round_fraction", type=float, default=1.0)
    parser.add_argument("--local-sample-ratio-per-epoch", type=float, default=1.0)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--no-ffa-lora", dest="ffa_lora", action="store_false")
    parser.set_defaults(use_lora=True, ffa_lora=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["query", "value"])
    parser.add_argument("--warmstart-project", type=str, default="lsstcorp")
    parser.add_argument("--warmstart-val-size", type=float, default=0.15)
    parser.add_argument("--warmstart-epochs", type=int, default=10)
    parser.add_argument("--warmstart-patience", type=int, default=3)
    parser.add_argument("--warmstart-lr", type=float, default=2e-5)
    parser.add_argument("--skip-centralized", action="store_true")
    parser.add_argument("--skip-local-only", action="store_true")
    parser.add_argument("--run-no-warmstart-fl", action="store_true")
    parser.add_argument("--central-log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Checkpoint frequency (0=off). Unit: epochs for centralized/local-only, global rounds for federated")
    parser.add_argument("--checkpoint-keep", type=int, default=2,
                        help="Number of numbered checkpoints to retain (latest/ and best/ are always kept)")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-detect and resume from the latest checkpoint under <save-dir>/checkpoints/")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from an explicit checkpoint dir (overrides --resume auto-detection)")
    parser.add_argument("--save-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if not (0.0 < args.clients_per_round_fraction <= 1.0):
        raise ValueError("--clients-per-round-fraction must be in (0, 1].")
    if not (0.0 < args.local_sample_ratio_per_epoch <= 1.0):
        raise ValueError("--local-sample-ratio-per-epoch must be in (0, 1].")

    config = FLConfig(
        data_dir=Path(args.data_dir),
        random_state=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        split_mode=args.split_mode,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        rounds=args.rounds,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        prox_mu=args.prox_mu,
        head_type=args.head_type,
        personalized_head=args.personalized_head,
        generic_head=args.generic_head,
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        resume=args.resume,
        resume_from=args.resume_from,
        clients_per_round_fraction=args.clients_per_round_fraction,
        local_sample_ratio_per_epoch=args.local_sample_ratio_per_epoch,
        sample_with_replacement=args.sample_with_replacement,
        freeze_encoder=args.freeze_encoder,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        ffa_lora=args.ffa_lora,
        skip_centralized=args.skip_centralized,
        skip_local_only=args.skip_local_only,
        run_no_warmstart_fl=args.run_no_warmstart_fl,
        warmstart_project=args.warmstart_project,
        warmstart_val_size=args.warmstart_val_size,
        warmstart_epochs=args.warmstart_epochs,
        warmstart_patience=args.warmstart_patience,
        warmstart_lr=args.warmstart_lr,
        device=args.device,
    )
    fl_condition = federated_condition_name(config.prox_mu, config.personalized_head)
    save_dir = Path(args.save_dir)

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    device = choose_device(config.device)

    data = load_dataset_by_project(config.data_dir)

    # Split warmstart project out before building FL bundle. Resolve the
    # configured name against client_id by exact match first, falling back
    # to a case-insensitive substring match (e.g. "lsstcorp" ->
    # "Lsstcorp_Data_management") so users don't need the full client_id.
    available = sorted(data["client_id"].unique())
    needle = config.warmstart_project.lower()
    exact = [c for c in available if c.lower() == needle]
    if exact:
        resolved_project = exact[0]
    else:
        substring_matches = [c for c in available if needle in c.lower()]
        if len(substring_matches) == 1:
            resolved_project = substring_matches[0]
        elif len(substring_matches) > 1:
            raise ValueError(
                f"Warmstart project '{config.warmstart_project}' is ambiguous. "
                f"Matching client IDs: {substring_matches}"
            )
        else:
            raise ValueError(
                f"Warmstart project '{config.warmstart_project}' not found in data. "
                f"Available client IDs: {available}"
            )

    warmstart_mask = data["client_id"] == resolved_project
    warmstart_data = data[warmstart_mask].reset_index(drop=True)
    fl_data = data[~warmstart_mask].reset_index(drop=True)
    config.warmstart_project = resolved_project
    print(
        f"[Data] Warmstart '{config.warmstart_project}': {len(warmstart_data)} rows | "
        f"FL projects: {fl_data['client_id'].nunique()} | {len(fl_data)} rows",
        flush=True,
    )

    # Category maps and train/val/test split built from FL data only.
    bundle = prepare_tabular_bundle(
        fl_data, test_size=config.test_size, random_state=config.random_state,
        split_mode=config.split_mode, val_size=config.val_size,
    )
    print(f"[Data] Split mode: {config.split_mode}", flush=True)

    if bundle.test_df.empty:
        raise ValueError("No test split generated. Increase data volume or reduce project fragmentation.")
    if bundle.val_df.empty:
        raise ValueError("No FL validation split generated. Increase data volume or reduce project fragmentation.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collate_fn = collate_fn_builder(tokenizer, config.max_length)

    # FL datasets and loaders.
    train_dataset = IssueDataset(
        frame=bundle.train_df,
        type_to_id=bundle.type_to_id,
        priority_to_id=bundle.priority_to_id,
    )
    fl_val_dataset = IssueDataset(
        frame=bundle.val_df,
        type_to_id=bundle.type_to_id,
        priority_to_id=bundle.priority_to_id,
    )
    test_dataset = IssueDataset(
        frame=bundle.test_df,
        type_to_id=bundle.type_to_id,
        priority_to_id=bundle.priority_to_id,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    fl_val_loader = DataLoader(fl_val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    print(
        f"[Data] FL split: train={len(bundle.train_df)}, val={len(bundle.val_df)}, test={len(bundle.test_df)}",
        flush=True,
    )

    # Warmstart train/val split — stratified so rare classes appear in val.
    ws_train_df, ws_val_df = train_test_split(
        warmstart_data,
        test_size=config.warmstart_val_size,
        random_state=config.random_state,
        stratify=warmstart_data["story_point"],
    )
    ws_train_dataset = IssueDataset(ws_train_df.reset_index(drop=True), bundle.type_to_id, bundle.priority_to_id)
    ws_val_dataset = IssueDataset(ws_val_df.reset_index(drop=True), bundle.type_to_id, bundle.priority_to_id)
    ws_train_loader = DataLoader(ws_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    ws_val_loader = DataLoader(ws_val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    print(
        f"[Data] Warmstart split: train={len(ws_train_df)}, val={len(ws_val_df)}",
        flush=True,
    )

    y_test_labels = test_dataset.labels  # integer class labels for evaluation

    def model_factory() -> StoryPointClassifier:
        return StoryPointClassifier(
            model_name=config.model_name,
            num_types=len(bundle.type_to_id),
            num_priorities=len(bundle.priority_to_id),
            categorical_emb_dim=config.categorical_emb_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            freeze_encoder=config.freeze_encoder,
            num_classes=config.num_classes,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
            ffa_lora=config.ffa_lora,
            head_type=config.head_type,
        )

    _probe = model_factory()
    log_trainable_params(_probe)
    probe_trainable_params = sum(p.numel() for p in _probe.parameters() if p.requires_grad)
    probe_total_params = sum(p.numel() for p in _probe.parameters())
    del _probe

    # Baseline: majority-class prediction.
    majority_class = int(np.bincount(train_dataset.labels).argmax())
    baseline_pred = np.full(len(y_test_labels), fill_value=majority_class, dtype=np.int64)
    baseline_metrics = evaluate_classification(y_test_labels, baseline_pred, config.num_classes)

    # Warm-start: pretrain LoRA + head on Lsstcorp with early stopping on val macro-F1.
    ws_class_weights = compute_class_weights(ws_train_dataset.labels, config.num_classes)
    warmstart_model = model_factory().to(device)
    warmstart_state, warmstart_val_f1 = train_warmstart(
        model=warmstart_model,
        train_loader=ws_train_loader,
        val_loader=ws_val_loader,
        val_labels=ws_val_dataset.labels,
        device=device,
        max_epochs=config.warmstart_epochs,
        patience=config.warmstart_patience,
        learning_rate=config.warmstart_lr,
        weight_decay=config.weight_decay,
        class_weights=ws_class_weights,
        num_classes=config.num_classes,
    )
    print(f"[Warmstart] Best val macro-F1: {warmstart_val_f1:.4f}", flush=True)

    warmstart_artifact = save_model_artifact(
        save_root=save_dir,
        artifact_name="warmstart",
        model=warmstart_model,
        tokenizer=tokenizer,
        config=config,
        type_to_id=bundle.type_to_id,
        priority_to_id=bundle.priority_to_id,
    )

    # Centralized deep model reference on FL data (fair comparison — same 18 projects as FL).
    centralized_model = None
    centralized_artifact = None
    if config.skip_centralized:
        print("[Centralized] Skipped (--skip-centralized).", flush=True)
    else:
        global_class_weights = compute_class_weights(train_dataset.labels, config.num_classes)
        centralized_model = model_factory().to(device)

        # Checkpoint / resume wiring (gap #13) for the centralized phase.
        central_ckpt_root = ck.centralized_ckpt_root(save_dir)
        central_resume_payload = None
        if config.resume and config.resume_from is None:
            latest = ck.latest_checkpoint_dir(central_ckpt_root)
            if latest is not None:
                print(f"[Centralized] Auto-resuming from latest checkpoint: {latest}", flush=True)
                central_resume_payload, _ = ck.load_checkpoint(latest, config)

        centralized_model = train_centralized(
            model=centralized_model,
            train_loader=train_loader,
            device=device,
            epochs=config.rounds,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            class_weights=global_class_weights,
            log_every=max(1, args.central_log_every),
            ckpt_root=central_ckpt_root,
            checkpoint_every=config.checkpoint_every,
            checkpoint_keep=config.checkpoint_keep,
            checkpoint_config=config,
            resume_payload=central_resume_payload,
        )
        centralized_artifact = save_model_artifact(
            save_root=save_dir,
            artifact_name="centralized",
            model=centralized_model,
            tokenizer=tokenizer,
            config=config,
            type_to_id=bundle.type_to_id,
            priority_to_id=bundle.priority_to_id,
        )

    # Federated deep model.
    clients: List[FederatedClient] = []
    for client_id, frame in bundle.train_df.groupby("client_id"):
        clients.append(
            FederatedClient(
                client_id=client_id,
                client_df=frame.reset_index(drop=True),
                tokenizer=tokenizer,
                type_to_id=bundle.type_to_id,
                priority_to_id=bundle.priority_to_id,
                num_classes=config.num_classes,
                max_length=config.max_length,
                batch_size=config.batch_size,
            )
        )

    # Personalized-head mode needs each client's own val split (shared repr + own head is
    # evaluated on it per round). Built once here and reused across warmstart/no-warmstart runs.
    client_val: Optional[Dict[str, Any]] = None
    if config.personalized_head:
        client_val = {}
        for client_id, group in bundle.val_df.groupby("client_id"):
            if group.empty:
                continue
            ds = IssueDataset(group.reset_index(drop=True), bundle.type_to_id, bundle.priority_to_id)
            loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
            client_val[str(client_id)] = (loader, ds.labels)
        print(f"[FedProx] Personalized-head per-client val splits: {len(client_val)} clients.", flush=True)

    communication_cost = compute_communication_cost(
        trainable_params=probe_trainable_params,
        total_params=probe_total_params,
        rounds=config.rounds,
        num_clients=len(clients),
        head_type=config.head_type,
    )

    # Local-only baseline: each client trains independently from the same warm-start
    # checkpoint as federated, with no cross-client aggregation. Total local epochs
    # match federated's per-client exposure (rounds x local_epochs) for a fair comparison.
    local_states = None
    if not config.skip_local_only:
        local_states = train_local_only(
            clients=clients,
            model_factory=model_factory,
            initial_state=warmstart_state,
            device=device,
            epochs=config.rounds * config.local_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            random_state=config.random_state,
        )
    else:
        print("[LocalOnly] Skipped (--skip-local-only).", flush=True)

    # Checkpoint / resume wiring (gap #13) for the primary federated run.
    fed_ckpt_root = ck.federated_ckpt_root(save_dir)
    fed_resume_payload = None
    if config.resume_from:
        resume_dir = Path(config.resume_from)
        print(f"[FedProx] Resuming from explicit checkpoint: {resume_dir}", flush=True)
        fed_resume_payload, _ = ck.load_checkpoint(resume_dir, config)
    elif config.resume:
        latest = ck.latest_checkpoint_dir(fed_ckpt_root)
        if latest is not None:
            print(f"[FedProx] Auto-resuming from latest checkpoint: {latest}", flush=True)
            fed_resume_payload, _ = ck.load_checkpoint(latest, config)
        else:
            print("[FedProx] --resume set but no federated checkpoint found; starting fresh.", flush=True)

    server = FedProxServer(model_factory=model_factory, clients=clients, random_state=config.random_state)
    fed_state, fed_history, fed_client_heads = server.train(
        rounds=config.rounds,
        clients_per_round_fraction=config.clients_per_round_fraction,
        local_epochs=config.local_epochs,
        local_sample_ratio_per_epoch=config.local_sample_ratio_per_epoch,
        sample_with_replacement=config.sample_with_replacement,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        prox_mu=config.prox_mu,
        device=device,
        initial_state=warmstart_state,
        val_loader=fl_val_loader,
        val_labels=fl_val_dataset.labels,
        num_classes=config.num_classes,
        personalized_head=config.personalized_head,
        client_val=client_val,
        checkpoint_every=config.checkpoint_every,
        checkpoint_keep=config.checkpoint_keep,
        ckpt_root=fed_ckpt_root,
        checkpoint_config=config,
        resume_payload=fed_resume_payload,
    )

    federated_model = model_factory().to(device)
    federated_model.load_state_dict(fed_state)

    if config.personalized_head:
        # Personalized layout: shared_state.pt + per-project heads (+ optional generic head).
        client_sizes = {str(c.client_id): len(c.dataset) for c in clients}
        generic_head_state = (
            compute_generic_head(fed_client_heads, client_sizes) if config.generic_head else None
        )
        federated_artifact = save_personalized_federated_artifact(
            save_root=save_dir,
            artifact_name="federated",
            shared_state=fed_state,
            client_heads=fed_client_heads,
            tokenizer=tokenizer,
            config=config,
            type_to_id=bundle.type_to_id,
            priority_to_id=bundle.priority_to_id,
            generic_head=generic_head_state,
        )
    else:
        federated_artifact = save_model_artifact(
            save_root=save_dir,
            artifact_name="federated",
            model=federated_model,
            tokenizer=tokenizer,
            config=config,
            type_to_id=bundle.type_to_id,
            priority_to_id=bundle.priority_to_id,
        )

    # Warm-start ablation (gap #7): also run FL from random init, using a fresh
    # server with the same random_state so both runs select the same per-round
    # clients — isolating the effect of the warm-start checkpoint.
    fed_results_nw = None
    fed_history_nw = None
    federated_artifact_nw = None
    if config.run_no_warmstart_fl:
        print("\n[FedProx] Running no-warmstart federated training (random init) ...", flush=True)
        server_nw = FedProxServer(model_factory=model_factory, clients=clients, random_state=config.random_state)
        fed_state_nw, fed_history_nw, fed_client_heads_nw = server_nw.train(
            rounds=config.rounds,
            clients_per_round_fraction=config.clients_per_round_fraction,
            local_epochs=config.local_epochs,
            local_sample_ratio_per_epoch=config.local_sample_ratio_per_epoch,
            sample_with_replacement=config.sample_with_replacement,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            prox_mu=config.prox_mu,
            device=device,
            initial_state=None,
            val_loader=fl_val_loader,
            val_labels=fl_val_dataset.labels,
            num_classes=config.num_classes,
            personalized_head=config.personalized_head,
            client_val=client_val,
        )

        federated_model_nw = model_factory().to(device)
        federated_model_nw.load_state_dict(fed_state_nw)

        if config.personalized_head:
            client_sizes = {str(c.client_id): len(c.dataset) for c in clients}
            generic_head_state_nw = (
                compute_generic_head(fed_client_heads_nw, client_sizes) if config.generic_head else None
            )
            federated_artifact_nw = save_personalized_federated_artifact(
                save_root=save_dir,
                artifact_name="federated_no_warmstart",
                shared_state=fed_state_nw,
                client_heads=fed_client_heads_nw,
                tokenizer=tokenizer,
                config=config,
                type_to_id=bundle.type_to_id,
                priority_to_id=bundle.priority_to_id,
                generic_head=generic_head_state_nw,
            )
        else:
            federated_artifact_nw = save_model_artifact(
                save_root=save_dir,
                artifact_name="federated_no_warmstart",
                model=federated_model_nw,
                tokenizer=tokenizer,
                config=config,
                type_to_id=bundle.type_to_id,
                priority_to_id=bundle.priority_to_id,
            )

        if config.personalized_head:
            fed_results_nw = evaluate_per_project_personalized(
                fed_state_nw, fed_client_heads_nw, model_factory, bundle.test_df,
                bundle.type_to_id, bundle.priority_to_id,
                collate_fn, device, config.batch_size, config.num_classes,
            )
        else:
            fed_results_nw = evaluate_per_project(
                federated_model_nw, bundle.test_df,
                bundle.type_to_id, bundle.priority_to_id,
                collate_fn, device, config.batch_size, config.num_classes,
            )
        del federated_model_nw

    # Per-project evaluation — both models evaluated on each FL client's test split.
    print("\n[Eval] Running per-project evaluation ...", flush=True)
    if config.personalized_head:
        # Personalized mode: each client uses its own head; no pooled-global entry.
        fed_results = evaluate_per_project_personalized(
            fed_state, fed_client_heads, model_factory, bundle.test_df,
            bundle.type_to_id, bundle.priority_to_id,
            collate_fn, device, config.batch_size, config.num_classes,
        )
    else:
        fed_results = evaluate_per_project(
            federated_model, bundle.test_df,
            bundle.type_to_id, bundle.priority_to_id,
            collate_fn, device, config.batch_size, config.num_classes,
        )
    central_results = (
        evaluate_per_project(
            centralized_model, bundle.test_df,
            bundle.type_to_id, bundle.priority_to_id,
            collate_fn, device, config.batch_size, config.num_classes,
        )
        if centralized_model is not None else None
    )
    local_results = (
        evaluate_local_only(
            local_states, model_factory, bundle.test_df,
            bundle.type_to_id, bundle.priority_to_id,
            collate_fn, device, config.batch_size, config.num_classes,
        )
        if local_states is not None else None
    )

    # In personalized mode there is no pooled-global entry; use the mean across projects
    # for the aggregate printouts and improvement comparisons.
    if config.personalized_head:
        federated_metrics = mean_metrics_across_projects(fed_results)
    else:
        federated_metrics = fed_results["global"]
    centralized_metrics = central_results["global"] if central_results is not None else None
    local_metrics = local_results["global"] if local_results is not None else None

    summary_df = build_summary_df(fed_results, central_results, local_results)
    results_dir = save_dir / "results"
    save_evaluation_results(
        results_dir, fed_results, central_results, summary_df, config,
        local_results, fed_history, communication_cost, fl_condition,
        bundle.test_df, fed_results_nw, fed_history_nw,
    )

    print("\nDataset summary")
    print(f"  Warmstart '{config.warmstart_project}': {len(warmstart_data)} rows (train={len(ws_train_df)}, val={len(ws_val_df)})")
    print(f"  FL projects: {fl_data['client_id'].nunique()} | Train: {len(bundle.train_df)} | Val: {len(bundle.val_df)} | Test: {len(bundle.test_df)} | Clients: {len(clients)}")
    print(f"  Model: {config.model_name} | Device: {device}")
    print(f"  Federated condition: {fl_condition} (prox_mu={config.prox_mu:g})")

    agg_label = "per-project mean" if config.personalized_head else "global pooled"
    print(f"\nClassification metrics — {agg_label} (primary: Macro-F1)")
    print(format_metrics("Baseline (Majority)", baseline_metrics))
    if local_metrics is not None:
        print(format_metrics("Local-only DL", local_metrics))
    if centralized_metrics is not None:
        print(format_metrics("Centralized DL", centralized_metrics))
    print(format_metrics(fl_condition, federated_metrics))
    if fed_results_nw is not None:
        nw_metrics = (
            mean_metrics_across_projects(fed_results_nw)
            if config.personalized_head else fed_results_nw["global"]
        )
        print(format_metrics(f"{fl_condition} (no warmstart)", nw_metrics))
    print(f"  Warmstart best val macro-F1: {warmstart_val_f1:.4f}")

    print(f"\nPer-project summary ({fl_condition} vs Centralized vs Local-only):")
    print(_df_to_markdown(summary_df))

    if fed_history:
        print(
            f"{fl_condition} loss history: first={fed_history[0]['mean_local_loss']:.6f}, "
            f"last={fed_history[-1]['mean_local_loss']:.6f}, rounds={len(fed_history)}"
        )
        # Shared mode records "val_macro_f1"; personalized mode records the weighted mean
        # across clients as "weighted_val_macro_f1". Select on whichever is present.
        val_key = "weighted_val_macro_f1" if config.personalized_head else "val_macro_f1"
        val_f1_history = [entry[val_key] for entry in fed_history if val_key in entry]
        if val_f1_history:
            best_round = max(fed_history, key=lambda e: e.get(val_key, -1.0))["round"]
            print(
                f"{fl_condition} val Macro-F1 history ({val_key}): first={val_f1_history[0]:.4f}, "
                f"last={val_f1_history[-1]:.4f}, best={max(val_f1_history):.4f} (round {best_round})"
            )

    if fed_history_nw:
        val_key_nw = "weighted_val_macro_f1" if config.personalized_head else "val_macro_f1"
        val_f1_history_nw = [entry[val_key_nw] for entry in fed_history_nw if val_key_nw in entry]
        if val_f1_history_nw:
            best_round_nw = max(fed_history_nw, key=lambda e: e.get(val_key_nw, -1.0))["round"]
            print(
                f"{fl_condition} (no warmstart) val Macro-F1 history: first={val_f1_history_nw[0]:.4f}, "
                f"last={val_f1_history_nw[-1]:.4f}, best={max(val_f1_history_nw):.4f} (round {best_round_nw})"
            )

    print("\nCommunication cost (per round, per client, float32)")
    print(
        f"  Trainable: {communication_cost['trainable_params']:,} / "
        f"{communication_cost['total_params']:,} params "
        f"({communication_cost['bytes_per_client_per_round']:,} bytes vs. "
        f"{communication_cost['bytes_per_client_per_round_full_finetune']:,} bytes full fine-tune)"
    )
    print(
        f"  Total upload over {communication_cost['rounds']} rounds x "
        f"{communication_cost['num_clients']} clients: "
        f"{communication_cost['total_upload_bytes']:,} bytes "
        f"({communication_cost['reduction_factor']:.1f}x smaller than full fine-tuning)"
    )

    print("\nSaved artifacts")
    print(f"  Warmstart:   {warmstart_artifact}")
    if centralized_artifact:
        print(f"  Centralized: {centralized_artifact}")
    print(f"  {fl_condition}: {federated_artifact}")
    if federated_artifact_nw:
        print(f"  {fl_condition} (no warmstart): {federated_artifact_nw}")

    f1_improvement = 100.0 * (federated_metrics["macro_f1"] - baseline_metrics["macro_f1"]) / max(
        baseline_metrics["macro_f1"], 1e-9
    )
    print(f"\n{fl_condition} Macro-F1 improvement vs baseline: {f1_improvement:.2f}%")

    if local_metrics is not None:
        f1_vs_local = 100.0 * (federated_metrics["macro_f1"] - local_metrics["macro_f1"]) / max(
            local_metrics["macro_f1"], 1e-9
        )
        print(f"{fl_condition} Macro-F1 improvement vs local-only: {f1_vs_local:.2f}%")

    if fed_results_nw is not None:
        nw_agg = (
            mean_metrics_across_projects(fed_results_nw)
            if config.personalized_head else fed_results_nw["global"]
        )
        no_ws_macro_f1 = nw_agg["macro_f1"]
        f1_warmstart_gain = 100.0 * (federated_metrics["macro_f1"] - no_ws_macro_f1) / max(no_ws_macro_f1, 1e-9)
        print(f"{fl_condition} Macro-F1 improvement from warm-start: {f1_warmstart_gain:.2f}%")


if __name__ == "__main__":
    main()
