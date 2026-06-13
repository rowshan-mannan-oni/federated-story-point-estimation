from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl.client import FederatedClient
from fl.metrics import evaluate_classification, run_prediction


class FedProxServer:
    """FedProx coordinator for deep learning models."""

    def __init__(self, model_factory, clients: List[FederatedClient], random_state: int) -> None:
        self.model_factory = model_factory
        self.clients = clients
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def choose_client_indices(total_clients: int, per_round: int, rng: np.random.Generator) -> np.ndarray:
        all_indices = np.arange(total_clients)
        return rng.choice(all_indices, size=per_round, replace=False)

    def train(
        self,
        rounds: int,
        clients_per_round_fraction: float,
        local_epochs: int,
        local_sample_ratio_per_epoch: float,
        sample_with_replacement: bool,
        learning_rate: float,
        weight_decay: float,
        prox_mu: float,
        device: torch.device,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
        val_loader: Optional[DataLoader] = None,
        val_labels: Optional[np.ndarray] = None,
        num_classes: int = 5,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        global_model = self.model_factory().to(device)
        if initial_state is not None:
            global_model.load_state_dict(initial_state)
            print("[FedProx] Initialized from warm-start checkpoint.", flush=True)
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        # Derive which keys to aggregate from trainable params — single source of truth.
        # Frozen backbone and (when FFA-LoRA is on) frozen A matrices are excluded.
        aggregatable_keys = frozenset(
            name for name, param in global_model.named_parameters() if param.requires_grad
        )
        n_agg = len(aggregatable_keys)
        n_total = sum(1 for _ in global_model.parameters())
        print(f"[FedProx] Aggregating {n_agg}/{n_total} param tensors per round.", flush=True)
        del global_model

        track_val = val_loader is not None and val_labels is not None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_macro_f1 = -1.0
        best_round = 0

        history: List[Dict[str, Any]] = []

        per_round = max(1, int(round(len(self.clients) * clients_per_round_fraction)))

        print(
            f"[FedProx] Starting training: rounds={rounds}, total_clients={len(self.clients)}, "
            f"clients_per_round={per_round}, prox_mu={prox_mu}, "
            f"local_sample_ratio_per_epoch={local_sample_ratio_per_epoch}, "
            f"sample_with_replacement={sample_with_replacement}",
            flush=True,
        )

        train_start = time.perf_counter()

        for round_idx in range(rounds):
            round_start = time.perf_counter()
            picked = self.choose_client_indices(total_clients=len(self.clients), per_round=per_round, rng=self.rng)
            selected_client_ids = [self.clients[int(idx)].client_id for idx in picked]

            print(
                f"[FedProx][Round {round_idx + 1}/{rounds}] selected_clients={selected_client_ids}",
                flush=True,
            )

            # Running weighted sum over aggregatable params, accumulated as each client
            # finishes so only one client's (tiny, trainable-only) state dict exists at a time.
            accum: Dict[str, torch.Tensor] = {}
            total_weight = 0.0
            client_weights: List[int] = []
            client_losses: List[float] = []

            for idx in picked:
                client_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
                result = self.clients[int(idx)].train_local(
                    global_state=global_state,
                    model_factory=self.model_factory,
                    device=device,
                    epochs=local_epochs,
                    sample_ratio_per_epoch=local_sample_ratio_per_epoch,
                    sample_with_replacement=sample_with_replacement,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    prox_mu=prox_mu,
                    seed=client_seed,
                )

                weight = float(result.num_examples)
                for key, tensor in result.state_dict.items():
                    contribution = tensor * weight
                    if key in accum:
                        accum[key] += contribution
                    else:
                        accum[key] = contribution
                total_weight += weight

                client_weights.append(result.num_examples)
                client_losses.append(result.loss)

                print(
                    f"  - client={self.clients[int(idx)].client_id} "
                    f"examples={result.num_examples} local_loss={result.loss:.6f}",
                    flush=True,
                )
                del result

            # Apply the aggregated update to the global state in place.
            for key, summed in accum.items():
                global_state[key] = summed / total_weight
            del accum
            round_mean_loss = float(np.mean(client_losses))
            weighted_round_loss = float(np.average(client_losses, weights=client_weights))

            round_entry: Dict[str, Any] = {
                "round": round_idx + 1,
                "mean_local_loss": round_mean_loss,
                "weighted_local_loss": weighted_round_loss,
            }

            val_msg = ""
            if track_val:
                eval_model = self.model_factory().to(device)
                eval_model.load_state_dict(global_state)
                val_pred = run_prediction(eval_model, val_loader, device)
                val_metrics = evaluate_classification(val_labels, val_pred, num_classes)
                del eval_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                val_macro_f1 = val_metrics["macro_f1"]
                round_entry["val_macro_f1"] = val_macro_f1
                round_entry["val_accuracy"] = val_metrics["accuracy"]

                is_best = val_macro_f1 > best_val_macro_f1
                if is_best:
                    best_val_macro_f1 = val_macro_f1
                    best_round = round_idx + 1
                    best_state = {k: v.clone() for k, v in global_state.items()}

                val_msg = (
                    f" val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_macro_f1:.4f}"
                    f"{' (best)' if is_best else ''}"
                )

            history.append(round_entry)

            completed_rounds = round_idx + 1
            round_elapsed = time.perf_counter() - round_start
            total_elapsed = time.perf_counter() - train_start
            avg_round_time = total_elapsed / completed_rounds
            remaining_rounds = max(rounds - completed_rounds, 0)
            eta_seconds = avg_round_time * remaining_rounds

            bar_width = 20
            filled = int((completed_rounds / max(rounds, 1)) * bar_width)
            bar = "#" * filled + "." * (bar_width - filled)

            print(
                f"[FedProx][Round {round_idx + 1}/{rounds}] "
                f"mean_local_loss={round_mean_loss:.6f} weighted_local_loss={weighted_round_loss:.6f}"
                f"{val_msg}",
                flush=True,
            )
            print(
                f"[FedProx][Progress] [{bar}] {completed_rounds}/{rounds} "
                f"round_time={round_elapsed:.1f}s elapsed={total_elapsed:.1f}s eta={eta_seconds:.1f}s",
                flush=True,
            )

        if track_val and best_state is not None:
            print(
                f"[FedProx] Selected global model from round {best_round} "
                f"(best val macro-F1={best_val_macro_f1:.4f}).",
                flush=True,
            )
            return best_state, history

        return global_state, history
