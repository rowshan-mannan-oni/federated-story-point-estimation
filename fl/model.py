from typing import Dict, Sequence

import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class StoryPointClassifier(nn.Module):
    """Transformer encoder + categorical embeddings + classification head."""

    def __init__(
        self,
        model_name: str,
        num_types: int,
        num_priorities: int,
        categorical_emb_dim: int,
        hidden_dim: int,
        dropout: float,
        freeze_encoder: bool,
        num_classes: int = 5,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("query", "value"),
        ffa_lora: bool = True,
        head_type: str = "ce",
    ) -> None:
        super().__init__()
        if head_type not in ("ce", "corn"):
            raise ValueError(f"head_type must be 'ce' or 'corn', got {head_type!r}")
        self.head_type = head_type
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_name)

        # Capture hidden_size before peft potentially rewraps the encoder.
        text_dim = self.encoder.config.hidden_size

        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            )
            # Wrap only the encoder so the head + embeddings stay trainable.
            self.encoder = get_peft_model(self.encoder, lora_cfg)
            if ffa_lora:
                # FFA-LoRA: freeze A so only B is trained and aggregated.
                for name, param in self.encoder.named_parameters():
                    if "lora_A" in name:
                        param.requires_grad_(False)
        elif freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.type_emb = nn.Embedding(num_types, categorical_emb_dim)
        self.priority_emb = nn.Embedding(num_priorities, categorical_emb_dim)

        fusion_dim = text_dim + categorical_emb_dim * 2

        # CORN ordinal head emits num_classes-1 threshold logits; CE head emits num_classes.
        head_out_dim = num_classes - 1 if head_type == "corn" else num_classes

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_out_dim),
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
        # CE: (batch, num_classes) logits. CORN: (batch, num_classes-1) threshold logits.
        return self.head(fused)


def compute_head_loss(
    head_type: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ce_criterion: nn.Module,
) -> torch.Tensor:
    """Branch the training loss on head type.

    CE: class-weighted CrossEntropy over `num_classes` logits.
    CORN: rank-consistent ordinal loss over `num_classes-1` threshold logits
    (class weights do not apply — CORN handles imbalance via its threshold structure).
    """
    if head_type == "corn":
        from coral_pytorch.losses import corn_loss
        return corn_loss(logits, targets, num_classes=num_classes)
    return ce_criterion(logits, targets)


def log_trainable_params(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    print(f"[Model] Trainable: {trainable:,} / {total:,} params ({pct:.2f}%)", flush=True)
