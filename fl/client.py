from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from fl.data import IssueDataset, compute_class_weights
from fl.metrics import evaluate_classification, preserved_rng, run_prediction
from fl.model import compute_head_loss, is_head_param


@dataclass
class ClientOutput:
    state_dict: Dict[str, torch.Tensor]
    num_examples: int
    loss: float
    # Set only when train_local was given a val split and selected a best epoch
    # (local-only condition). None in federated rounds, where the SERVER does the
    # best-on-val selection across rounds instead.
    best_val_macro_f1: Optional[float] = None
    best_epoch: Optional[int] = None


class FederatedClient:
    """Single client trainer used by the federated server."""

    def __init__(
        self,
        client_id: str,
        client_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        type_to_id: Dict[str, int],
        priority_to_id: Dict[str, int],
        num_classes: int,
        max_length: int,
        batch_size: int,
    ) -> None:
        self.client_id = client_id
        self.num_classes = num_classes
        self.dataset = IssueDataset(
            frame=client_df,
            type_to_id=type_to_id,
            priority_to_id=priority_to_id,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def _collate(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        texts = [example["text"] for example in examples]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        type_id = torch.stack([example["type_id"] for example in examples])
        priority_id = torch.stack([example["priority_id"] for example in examples])
        target = torch.stack([example["target"] for example in examples])

        encoded["type_id"] = type_id
        encoded["priority_id"] = priority_id
        encoded["target"] = target

        return encoded

    @staticmethod
    def sample_epoch_indices(
        num_examples: int,
        sample_ratio_per_epoch: float,
        sample_with_replacement: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if num_examples <= 0:
            return np.array([], dtype=np.int64)

        if sample_ratio_per_epoch >= 1.0:
            if sample_with_replacement:
                return rng.choice(num_examples, size=num_examples, replace=True).astype(np.int64)
            return rng.permutation(num_examples).astype(np.int64)

        sample_size = max(1, int(round(num_examples * sample_ratio_per_epoch)))
        return rng.choice(num_examples, size=sample_size, replace=sample_with_replacement).astype(np.int64)

    def train_local(
        self,
        global_state: Dict[str, torch.Tensor],
        model_factory,
        device: torch.device,
        epochs: int,
        sample_ratio_per_epoch: float,
        sample_with_replacement: bool,
        learning_rate: float,
        weight_decay: float,
        prox_mu: float,
        seed: int,
        personalized_head: bool = False,
        val_loader: Optional[DataLoader] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> ClientOutput:
        """
        Train this client's model for `epochs` local epochs and return its trainable params.

        `val_loader`/`val_labels` are for the LOCAL-ONLY condition, where a client trains
        for the full budget in one call and nothing else selects a checkpoint for it: when
        given, the epoch with the best val macro-F1 is returned instead of the final epoch.
        Federated rounds leave them None — there the server does best-on-val selection
        across rounds, so selecting per round as well would double-select.
        """
        rng = np.random.default_rng(seed)

        model = model_factory().to(device)
        model.load_state_dict(global_state)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Per-client inverse-frequency weights — computed once from full local data.
        # (Used only by the CE head; CORN ignores class weights.)
        class_weights = compute_class_weights(self.dataset.labels, self.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        head_type = getattr(model, "head_type", "ce")

        # The FedProx proximal term is computed over AGGREGATABLE params only — the params
        # that have a global reference point on the server. This restriction is applied
        # unconditionally: in shared-head mode the head IS aggregated, so it's included and
        # behavior is unchanged; in personalized-head mode the local head has no global
        # counterpart (never aggregated), so anchoring it to the global state would be
        # meaningless and would undo personalization — it's excluded.
        def is_aggregatable(name: str, param) -> bool:
            return param.requires_grad and not (personalized_head and is_head_param(name))

        # Only load aggregatable params for the proximal term — avoids double-loading the
        # frozen backbone (~264MB for DistilBERT) that is already on device via model.to(device).
        global_params = {
            name: global_state[name].to(device)
            for name, param in model.named_parameters()
            if is_aggregatable(name, param)
        }

        running_loss = 0.0
        num_batches = 0
        sampled_examples_total = 0

        select_on_val = val_loader is not None and val_labels is not None
        best_val_macro_f1 = -1.0
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        def trainable_snapshot() -> Dict[str, torch.Tensor]:
            # Frozen params are identical to global_state and never change, so a
            # trainable-only snapshot fully captures one epoch's model.
            return {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

        for epoch_idx in range(epochs):
            # run_prediction leaves the model in eval mode, so re-enter train mode every
            # epoch. Harmless no-op when no val pass runs.
            model.train()
            sampled_indices = self.sample_epoch_indices(
                num_examples=len(self.dataset),
                sample_ratio_per_epoch=sample_ratio_per_epoch,
                sample_with_replacement=sample_with_replacement,
                rng=rng,
            )
            sampled_examples_total += int(len(sampled_indices))
            epoch_subset = Subset(self.dataset, sampled_indices.tolist())
            loader = DataLoader(epoch_subset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate)

            preview = sampled_indices[: min(10, len(sampled_indices))].tolist()
            print(
                f"    [Client {self.client_id}][Epoch {epoch_idx + 1}/{epochs}] "
                f"sampled_examples={len(sampled_indices)} total_client_examples={len(self.dataset)} first_indices={preview}",
                flush=True,
            )

            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
                loss = compute_head_loss(head_type, pred, batch["target"], self.num_classes, criterion)

                if prox_mu > 0.0:
                    prox_term = torch.zeros((), device=device)
                    for name, param in model.named_parameters():
                        if is_aggregatable(name, param):  # aggregatable only (see note above)
                            prox_term += torch.sum((param - global_params[name]) ** 2)
                    loss = loss + 0.5 * prox_mu * prox_term

                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                num_batches += 1

                if (batch_idx + 1) % 100 == 0:
                    print(
                        f"      [Client {self.client_id}][Epoch {epoch_idx + 1}] "
                        f"batch {batch_idx + 1}/{len(loader)} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            # Local-only best-on-val selection, wrapped in preserved_rng: iterating the val
            # DataLoader draws from the global torch RNG, which would leak into everything
            # downstream (later clients' inits, and on CPU the dropout masks of the
            # federated phase that runs after local-only). Keeps this purely observational.
            if select_on_val:
                with preserved_rng():
                    val_pred = run_prediction(model, val_loader, device)
                val_macro_f1 = evaluate_classification(val_labels, val_pred, self.num_classes)["macro_f1"]
                improved = val_macro_f1 > best_val_macro_f1
                if improved:
                    best_val_macro_f1 = val_macro_f1
                    best_epoch = epoch_idx + 1
                    best_state = trainable_snapshot()
                print(
                    f"    [Client {self.client_id}][Epoch {epoch_idx + 1}/{epochs}] "
                    f"val_macro_f1={val_macro_f1:.4f}" + ("  *best*" if improved else ""),
                    flush=True,
                )

        avg_loss = running_loss / max(num_batches, 1)
        # Only trainable params are aggregated server-side — frozen backbone weights
        # are identical to global_state and would otherwise be cloned/transferred for nothing
        # (~500MB/client for a 125M-param encoder, vs ~1MB for LoRA-B + embeddings + head).
        if select_on_val and best_state is not None:
            state = best_state
            print(
                f"    [Client {self.client_id}] Selected epoch {best_epoch}/{epochs} "
                f"(best val macro-F1={best_val_macro_f1:.4f}).",
                flush=True,
            )
        else:
            state = trainable_snapshot()

        # Explicitly release GPU tensors so the CUDA allocator can defragment before
        # the next client's model is loaded; without this, sequential client training
        # accumulates fragmentation and causes OOM mid-round. The optimizer holds its
        # own GPU-resident state (momentum/variance buffers) independent of the model,
        # so it must be deleted too or empty_cache() can't reclaim that memory.
        del model, optimizer, global_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        effective_examples = max(sampled_examples_total, 1)
        return ClientOutput(
            state_dict=state,
            num_examples=effective_examples,
            loss=avg_loss,
            best_val_macro_f1=best_val_macro_f1 if select_on_val else None,
            best_epoch=best_epoch if select_on_val else None,
        )
