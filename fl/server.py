from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl import checkpoint as ck
from fl.client import FederatedClient
from fl.metrics import evaluate_classification, run_prediction
from fl.model import is_head_param


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
        personalized_head: bool = False,
        client_val: Optional[Dict[str, Tuple[DataLoader, np.ndarray]]] = None,
        checkpoint_every: int = 0,
        checkpoint_keep: int = 2,
        ckpt_root: Optional[Path] = None,
        checkpoint_config: Optional[Any] = None,
        resume_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]], Optional[Dict[str, Dict[str, torch.Tensor]]]]:
        global_model = self.model_factory().to(device)
        if initial_state is not None:
            global_model.load_state_dict(initial_state)
            print("[FedProx] Initialized from warm-start checkpoint.", flush=True)
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        # Derive which keys to aggregate from trainable params — single source of truth.
        # Frozen backbone and (when FFA-LoRA is on) frozen A matrices are excluded.
        trainable_keys = frozenset(
            name for name, param in global_model.named_parameters() if param.requires_grad
        )
        # Personalized-head mode (gap #11): the classification head stays local per client
        # and is never aggregated — only LoRA-B + embeddings are federated.
        head_keys = frozenset(k for k in trainable_keys if is_head_param(k))
        if personalized_head:
            aggregatable_keys = trainable_keys - head_keys
            print(
                f"[FedProx] Personalized-head mode: excluding {len(head_keys)} head param "
                f"tensor(s) from aggregation: {sorted(head_keys)}",
                flush=True,
            )
        else:
            aggregatable_keys = trainable_keys
        n_agg = len(aggregatable_keys)
        n_total = sum(1 for _ in global_model.parameters())
        print(f"[FedProx] Aggregating {n_agg}/{n_total} param tensors per round.", flush=True)
        del global_model

        # Personalized-head mode: every client keeps its OWN head across rounds. All clients
        # start from the same initial head (from the warm-start checkpoint / random init).
        # Clients not selected in a round retain their head; selected clients update it.
        client_heads: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        best_client_heads: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if personalized_head:
            init_head = {k: v.clone() for k, v in global_state.items() if is_head_param(k)}
            client_heads = {
                str(c.client_id): {k: v.clone() for k, v in init_head.items()}
                for c in self.clients
            }

        # Shared mode tracks val on the single pooled global model; personalized mode tracks
        # val per-client (own head on own val split) and selects on the weighted mean.
        track_val_shared = (not personalized_head) and val_loader is not None and val_labels is not None
        track_val_personalized = personalized_head and bool(client_val)
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_macro_f1 = -1.0
        best_round = 0

        history: List[Dict[str, Any]] = []

        # Resume (gap #13): overlay the checkpoint's trainable-only states onto the
        # freshly-constructed global_state (so the frozen backbone + LoRA-A come from the
        # normal init path and only B/embeddings/head are restored), then restore per-client
        # heads, history, best-on-val bookkeeping, the round index, and all RNG state. The
        # loop then continues at round start_round+1 reproducing an uninterrupted run.
        start_round = 0
        if resume_payload is not None:
            for key, tensor in resume_payload["shared_state"].items():
                global_state[key] = tensor.clone()
            if personalized_head and resume_payload.get("client_heads") is not None:
                client_heads = {
                    hcid: {k: v.clone() for k, v in head.items()}
                    for hcid, head in resume_payload["client_heads"].items()
                }
            history = list(resume_payload.get("history", []))
            best_val_macro_f1 = resume_payload.get("best_val_macro_f1", -1.0)
            best_round = resume_payload.get("best_round", 0)
            if resume_payload.get("best_state") is not None:
                # best_state is stored trainable-only; rebuild it on the current backbone.
                best_state = {k: v.clone() for k, v in global_state.items()}
                for key, tensor in resume_payload["best_state"].items():
                    best_state[key] = tensor.clone()
            if personalized_head and resume_payload.get("best_client_heads") is not None:
                best_client_heads = {
                    hcid: {k: v.clone() for k, v in head.items()}
                    for hcid, head in resume_payload["best_client_heads"].items()
                }
            start_round = int(resume_payload["round"])
            ck.restore_rng_state(resume_payload["rng"], {"server": self.rng})
            print(
                f"[FedProx] Resumed from round {start_round} "
                f"(best val macro-F1={best_val_macro_f1:.4f} at round {best_round}); "
                f"continuing to round {rounds}.",
                flush=True,
            )

        per_round = max(1, int(round(len(self.clients) * clients_per_round_fraction)))

        print(
            f"[FedProx] Starting training: rounds={rounds}, total_clients={len(self.clients)}, "
            f"clients_per_round={per_round}, prox_mu={prox_mu}, "
            f"local_sample_ratio_per_epoch={local_sample_ratio_per_epoch}, "
            f"sample_with_replacement={sample_with_replacement}",
            flush=True,
        )

        train_start = time.perf_counter()

        for round_idx in range(start_round, rounds):
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
                cid = str(self.clients[int(idx)].client_id)
                client_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
                # In personalized mode the client trains the shared representation plus ITS
                # OWN head — inject that head into the state it loads (otherwise it would
                # restart from the stale global head every round).
                if personalized_head:
                    round_client_state = dict(global_state)
                    round_client_state.update(client_heads[cid])
                else:
                    round_client_state = global_state
                result = self.clients[int(idx)].train_local(
                    global_state=round_client_state,
                    model_factory=self.model_factory,
                    device=device,
                    epochs=local_epochs,
                    sample_ratio_per_epoch=local_sample_ratio_per_epoch,
                    sample_with_replacement=sample_with_replacement,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    prox_mu=prox_mu,
                    seed=client_seed,
                    personalized_head=personalized_head,
                )

                weight = float(result.num_examples)
                for key, tensor in result.state_dict.items():
                    # Skip non-aggregatable keys (e.g. the local head in personalized mode).
                    if key not in aggregatable_keys:
                        continue
                    contribution = tensor * weight
                    if key in accum:
                        accum[key] += contribution
                    else:
                        accum[key] = contribution
                total_weight += weight

                # Persist this client's freshly-trained head for next time it's selected
                # and for per-client validation / test evaluation.
                if personalized_head:
                    client_heads[cid] = {
                        k: v.clone() for k, v in result.state_dict.items() if is_head_param(k)
                    }

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
            is_best = False  # whether this round set a new best-on-val (drives best/ checkpoint mirror)
            if track_val_shared:
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
            elif track_val_personalized:
                # Each client evaluates the shared representation + its OWN head on its OWN
                # val split. Selection is on the weighted mean val macro-F1 (weight = client
                # val examples); the best round snapshots ALL client heads.
                macro_f1s: List[float] = []
                accuracies: List[float] = []
                val_ns: List[int] = []
                for eval_cid, (eval_loader, eval_labels) in client_val.items():
                    eval_model = self.model_factory().to(device)
                    eval_state = dict(global_state)
                    eval_state.update(client_heads[eval_cid])
                    eval_model.load_state_dict(eval_state)
                    eval_pred = run_prediction(eval_model, eval_loader, device)
                    eval_metrics = evaluate_classification(eval_labels, eval_pred, num_classes)
                    macro_f1s.append(eval_metrics["macro_f1"])
                    accuracies.append(eval_metrics["accuracy"])
                    val_ns.append(int(len(eval_labels)))
                    del eval_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                mean_val_macro_f1 = float(np.mean(macro_f1s))
                weighted_val_macro_f1 = float(np.average(macro_f1s, weights=val_ns))
                mean_val_accuracy = float(np.mean(accuracies))
                round_entry["mean_val_macro_f1"] = mean_val_macro_f1
                round_entry["weighted_val_macro_f1"] = weighted_val_macro_f1
                round_entry["mean_val_accuracy"] = mean_val_accuracy

                is_best = weighted_val_macro_f1 > best_val_macro_f1
                if is_best:
                    best_val_macro_f1 = weighted_val_macro_f1
                    best_round = round_idx + 1
                    best_state = {k: v.clone() for k, v in global_state.items()}
                    best_client_heads = {
                        hcid: {k: v.clone() for k, v in head.items()}
                        for hcid, head in client_heads.items()
                    }

                val_msg = (
                    f" mean_val_macro_f1={mean_val_macro_f1:.4f} "
                    f"weighted_val_macro_f1={weighted_val_macro_f1:.4f}"
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

            # Checkpoint after per-round val (gap #13). Trainable-only states keep it small;
            # RNG is captured AFTER this round's selection + client seeds so a resume
            # reproduces the next round's client selection exactly.
            if checkpoint_every > 0 and ckpt_root is not None and (
                completed_rounds % checkpoint_every == 0 or completed_rounds == rounds
            ):
                payload: Dict[str, Any] = {
                    "round": completed_rounds,
                    "shared_state": {k: global_state[k].clone() for k in trainable_keys},
                    "best_state": (
                        {k: best_state[k].clone() for k in trainable_keys}
                        if best_state is not None else None
                    ),
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_round": best_round,
                    "history": history,
                    "rng": ck.capture_rng_state({"server": self.rng}),
                }
                if personalized_head:
                    payload["client_heads"] = {
                        hcid: {k: v.clone() for k, v in head.items()}
                        for hcid, head in client_heads.items()
                    }
                    payload["best_client_heads"] = (
                        {
                            hcid: {k: v.clone() for k, v in head.items()}
                            for hcid, head in best_client_heads.items()
                        }
                        if best_client_heads is not None else None
                    )
                ck.save_checkpoint(
                    ckpt_root, completed_rounds, payload, checkpoint_config,
                    is_best=is_best, keep=checkpoint_keep, step_prefix="round",
                )
                print(
                    f"[FedProx][Checkpoint] saved round {completed_rounds} "
                    f"(is_best={is_best}) to {ckpt_root}",
                    flush=True,
                )

        if personalized_head:
            # Return best-on-val shared state + the per-client heads snapshotted at that
            # round. Falls back to the final round if val tracking was disabled.
            if track_val_personalized and best_state is not None:
                print(
                    f"[FedProx] Selected shared state from round {best_round} "
                    f"(best weighted val macro-F1={best_val_macro_f1:.4f}); "
                    f"snapshotted {len(best_client_heads)} client heads.",
                    flush=True,
                )
                return best_state, history, best_client_heads
            return global_state, history, client_heads

        if track_val_shared and best_state is not None:
            print(
                f"[FedProx] Selected global model from round {best_round} "
                f"(best val macro-F1={best_val_macro_f1:.4f}).",
                flush=True,
            )
            return best_state, history, None

        return global_state, history, None
