from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from fl.data import IssueDataset, compute_class_weights


@dataclass
class ClientOutput:
    state_dict: Dict[str, torch.Tensor]
    num_examples: int
    loss: float


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
    ) -> ClientOutput:
        rng = np.random.default_rng(seed)

        model = model_factory().to(device)
        model.load_state_dict(global_state)
        model.train()

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Per-client inverse-frequency weights — computed once from full local data.
        class_weights = compute_class_weights(self.dataset.labels, self.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Only load trainable params for the proximal term — avoids double-loading the
        # frozen backbone (~264MB for DistilBERT) that is already on device via model.to(device).
        global_params = {
            name: global_state[name].to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        running_loss = 0.0
        num_batches = 0
        sampled_examples_total = 0

        for epoch_idx in range(epochs):
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
                loss = criterion(pred, batch["target"])

                if prox_mu > 0.0:
                    prox_term = torch.zeros((), device=device)
                    for name, param in model.named_parameters():
                        if param.requires_grad:  # B + head only when FFA-LoRA on
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

        avg_loss = running_loss / max(num_batches, 1)
        # Only trainable params are aggregated server-side — frozen backbone weights
        # are identical to global_state and would otherwise be cloned/transferred for nothing
        # (~500MB/client for a 125M-param encoder, vs ~1MB for LoRA-B + embeddings + head).
        state = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Explicitly release GPU tensors so the CUDA allocator can defragment before
        # the next client's model is loaded; without this, sequential client training
        # accumulates fragmentation and causes OOM mid-round. The optimizer holds its
        # own GPU-resident state (momentum/variance buffers) independent of the model,
        # so it must be deleted too or empty_cache() can't reclaim that memory.
        del model, optimizer, global_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        effective_examples = max(sampled_examples_total, 1)
        return ClientOutput(state_dict=state, num_examples=effective_examples, loss=avg_loss)
