import argparse
import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fl.client import FederatedClient
from fl.config import FLConfig
from fl.data import ClientNormStats, IssueDataset, load_dataset_by_project, prepare_tabular_bundle
from fl.metrics import evaluate_regression, format_metrics
from fl.model import StoryPointRegressor
from fl.server import FedAvgServer


def choose_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def run_prediction(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(batch).detach().cpu().numpy()
            preds.append(output)

    return np.concatenate(preds, axis=0)


def denormalize_per_client(
    predictions: np.ndarray,
    client_ids: List[str],
    client_stats: Dict[str, ClientNormStats],
    global_stats: ClientNormStats,
) -> np.ndarray:
    result = np.empty_like(predictions, dtype=np.float64)
    for i, (pred, cid) in enumerate(zip(predictions, client_ids)):
        stats = client_stats.get(cid, global_stats)
        result[i] = stats.denormalize(np.array([pred]))[0]
    return result


def train_centralized(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    log_every: int = 1,
) -> nn.Module:
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    print(
        f"[Centralized] Starting training: epochs={epochs}, batches_per_epoch={len(train_loader)}",
        flush=True,
    )

    train_start = time.perf_counter()

    for epoch_idx in range(epochs):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = criterion(pred, batch["target"])
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

    return model


def save_model_artifact(
    save_root: Path,
    artifact_name: str,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    config: FLConfig,
    type_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
    global_stats: ClientNormStats,
    client_stats: Dict[str, ClientNormStats],
) -> Path:
    artifact_dir = save_root / artifact_name
    tokenizer_dir = artifact_dir / "tokenizer"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), artifact_dir / "model_state.pt")
    tokenizer.save_pretrained(tokenizer_dir)

    metadata = {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "categorical_emb_dim": config.categorical_emb_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "freeze_encoder": config.freeze_encoder,
        "num_types": len(type_to_id),
        "num_priorities": len(priority_to_id),
        "type_to_id": type_to_id,
        "priority_to_id": priority_to_id,
        "global_stats": {"mean": global_stats.mean, "std": global_stats.std},
        "client_stats": {cid: {"mean": s.mean, "std": s.std} for cid, s in client_stats.items()},
    }

    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Deep Learning for Story Point Estimation")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--skip-centralized", action="store_true")
    parser.add_argument("--central-log-every", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    config = FLConfig(
        data_dir=Path(args.data_dir),
        random_state=args.seed,
        test_size=args.test_size,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        rounds=args.rounds,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        clients_per_round_fraction=args.fraction,
        freeze_encoder=args.freeze_encoder,
        device=args.device,
    )

    save_dir = Path(args.save_dir)

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    device = choose_device(config.device)

    data = load_dataset_by_project(config.data_dir)
    bundle = prepare_tabular_bundle(data, test_size=config.test_size, random_state=config.random_state)

    if bundle.test_df.empty:
        raise ValueError("No test split generated. Increase data volume or reduce project fragmentation.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = IssueDataset(
        frame=bundle.train_df,
        type_to_id=bundle.type_to_id,
        priority_to_id=bundle.priority_to_id,
        norm_mean=bundle.global_stats.mean,
        norm_std=bundle.global_stats.std,
    )
    test_dataset = IssueDataset(
        frame=bundle.test_df,
        type_to_id=bundle.type_to_id,
        priority_to_id=bundle.priority_to_id,
        norm_mean=bundle.global_stats.mean,
        norm_std=bundle.global_stats.std,
    )

    collate_fn = collate_fn_builder(tokenizer, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    y_test_raw = test_dataset.target_raw.astype(np.float64)

    def model_factory() -> StoryPointRegressor:
        return StoryPointRegressor(
            model_name=config.model_name,
            num_types=len(bundle.type_to_id),
            num_priorities=len(bundle.priority_to_id),
            categorical_emb_dim=config.categorical_emb_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            freeze_encoder=config.freeze_encoder,
        )

    # Baseline prediction: global mean on raw target values.
    mean_pred = np.full_like(y_test_raw, fill_value=float(np.mean(train_dataset.target_raw)), dtype=np.float64)
    baseline_metrics = evaluate_regression(y_test_raw, mean_pred)

    # Centralized deep model reference (skipped when --skip-centralized is set).
    centralized_metrics = None
    centralized_artifact = None
    if not args.skip_centralized:
        centralized_model = model_factory().to(device)
        centralized_model = train_centralized(
            model=centralized_model,
            train_loader=train_loader,
            device=device,
            epochs=config.rounds,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            log_every=max(1, args.central_log_every),
        )
        centralized_pred_norm = run_prediction(centralized_model, test_loader, device)
        centralized_pred = bundle.global_stats.denormalize(centralized_pred_norm)
        centralized_pred = np.clip(centralized_pred, a_min=0.0, a_max=None)
        centralized_metrics = evaluate_regression(y_test_raw, centralized_pred)

        centralized_artifact = save_model_artifact(
            save_root=save_dir,
            artifact_name="centralized",
            model=centralized_model,
            tokenizer=tokenizer,
            config=config,
            type_to_id=bundle.type_to_id,
            priority_to_id=bundle.priority_to_id,
            global_stats=bundle.global_stats,
            client_stats=bundle.client_stats,
        )

    # Federated deep model.
    clients: List[FederatedClient] = []
    for client_id, frame in bundle.train_df.groupby("client_id"):
        stats = bundle.client_stats[client_id]
        clients.append(
            FederatedClient(
                client_id=client_id,
                client_df=frame.reset_index(drop=True),
                tokenizer=tokenizer,
                type_to_id=bundle.type_to_id,
                priority_to_id=bundle.priority_to_id,
                norm_mean=stats.mean,
                norm_std=stats.std,
                max_length=config.max_length,
                batch_size=config.batch_size,
            )
        )

    server = FedAvgServer(model_factory=model_factory, clients=clients, random_state=config.random_state)
    fed_state, fed_history = server.train(
        rounds=config.rounds,
        clients_per_round_fraction=config.clients_per_round_fraction,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )

    federated_model = model_factory().to(device)
    federated_model.load_state_dict(fed_state)

    federated_pred_norm = run_prediction(federated_model, test_loader, device)
    federated_pred = denormalize_per_client(
        predictions=federated_pred_norm,
        client_ids=bundle.test_df["client_id"].tolist(),
        client_stats=bundle.client_stats,
        global_stats=bundle.global_stats,
    )
    federated_pred = np.clip(federated_pred, a_min=0.0, a_max=None)
    federated_metrics = evaluate_regression(y_test_raw, federated_pred)

    federated_artifact = save_model_artifact(
        save_root=save_dir,
        artifact_name="federated",
        model=federated_model,
        tokenizer=tokenizer,
        config=config,
        type_to_id=bundle.type_to_id,
        priority_to_id=bundle.priority_to_id,
        global_stats=bundle.global_stats,
        client_stats=bundle.client_stats,
    )

    print("\nDataset summary")
    print(
        f"Rows: {len(data)} | Train: {len(bundle.train_df)} | Test: {len(bundle.test_df)} | Clients: {len(clients)}"
    )
    print(f"Model: {config.model_name} | Device: {device}")

    print("\nRegression metrics (lower MAE/RMSE/MAPE is better, higher R2 is better)")
    print(format_metrics("Baseline (Mean)", baseline_metrics))
    if centralized_metrics is not None:
        print(format_metrics("Centralized DL", centralized_metrics))
    print(format_metrics("Federated DL", federated_metrics))

    if fed_history:
        print(
            f"Federated loss history: first={fed_history[0]:.6f}, "
            f"last={fed_history[-1]:.6f}, rounds={len(fed_history)}"
        )

    print("\nSaved artifacts")
    if centralized_artifact is not None:
        print(f"Centralized: {centralized_artifact}")
    print(f"Federated: {federated_artifact}")

    mae_improvement = 100.0 * (baseline_metrics["mae"] - federated_metrics["mae"]) / max(
        baseline_metrics["mae"], 1e-9
    )
    print(f"\nFederated MAE improvement vs baseline: {mae_improvement:.2f}%")


if __name__ == "__main__":
    main()
