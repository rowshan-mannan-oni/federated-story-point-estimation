from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fl.data import IssueDataset


@dataclass
class ClientOutput:
    state_dict: Dict[str, torch.Tensor]
    num_examples: int
    loss: float


class FederatedClient:
    """Single client trainer used by the FedAvg server."""

    def __init__(
        self,
        client_id: str,
        client_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        type_to_id: Dict[str, int],
        priority_to_id: Dict[str, int],
        use_log_target: bool,
        max_length: int,
        batch_size: int,
    ) -> None:
        self.client_id = client_id
        self.dataset = IssueDataset(
            frame=client_df,
            type_to_id=type_to_id,
            priority_to_id=priority_to_id,
            use_log_target=use_log_target,
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

    def train_local(
        self,
        global_state: Dict[str, torch.Tensor],
        model_factory,
        device: torch.device,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
    ) -> ClientOutput:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate)

        model = model_factory().to(device)
        model.load_state_dict(global_state)
        model.train()

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        running_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
                loss = criterion(pred, batch["target"])
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        return ClientOutput(state_dict=state, num_examples=len(self.dataset), loss=avg_loss)
