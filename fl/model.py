from typing import Dict

import torch
from torch import nn
from transformers import AutoModel


class StoryPointRegressor(nn.Module):
    """Hybrid regressor using transformer text encoding and categorical embeddings."""

    def __init__(
        self,
        model_name: str,
        num_types: int,
        num_priorities: int,
        categorical_emb_dim: int,
        hidden_dim: int,
        dropout: float,
        freeze_encoder: bool,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        text_dim = self.encoder.config.hidden_size
        self.type_emb = nn.Embedding(num_types, categorical_emb_dim)
        self.priority_emb = nn.Embedding(num_priorities, categorical_emb_dim)

        fusion_dim = text_dim + categorical_emb_dim * 2

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = output.last_hidden_state

        # Mean pooling over valid tokens.
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)

        return summed / counts

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_vector = self.encode_text(batch["input_ids"], batch["attention_mask"])
        type_vector = self.type_emb(batch["type_id"])
        priority_vector = self.priority_emb(batch["priority_id"])

        fused = torch.cat([text_vector, type_vector, priority_vector], dim=1)
        prediction = self.head(fused).squeeze(1)

        return prediction
