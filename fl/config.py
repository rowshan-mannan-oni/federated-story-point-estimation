from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FLConfig:
    data_dir: Path

    # Reproducibility and split controls.
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1   # per-client validation fraction carved from the train split (FL convergence tracking)
    split_mode: str = "random"  # "random" or "temporal" (train on earlier issues, test on later ones)

    # Model controls.
    model_name: str = "prajjwal1/bert-tiny"
    max_length: int = 128
    categorical_emb_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.2
    freeze_encoder: bool = False  # only used when use_lora=False

    # LoRA config.
    use_lora: bool = True
    lora_r: int = 8                # rank; r=8 is standard for NLP classification fine-tuning
    lora_alpha: int = 16           # scaling = alpha/r = 2.0; mild amplification of adapter update
    lora_dropout: float = 0.05     # light regularization on adapter weights
    lora_target_modules: tuple = field(default_factory=lambda: ("query", "value"))  # BERT Q+V; use ("q_lin","v_lin") for DistilBERT
    ffa_lora: bool = True  # FFA-LoRA: freeze A, train+aggregate B only — principled for non-IID FL

    # Warm-start (centralized pre-training on one large project before FL).
    warmstart_project: str = "lsstcorp"   # matched case-insensitively against client_id
    warmstart_val_size: float = 0.15      # stratified val fraction for early-stopping signal
    warmstart_epochs: int = 10            # max pre-training epochs
    warmstart_patience: int = 3           # stop after this many epochs without val macro-F1 gain
    warmstart_lr: float = 2e-5            # pre-training LR; independent of FL learning_rate

    # Training toggles.
    skip_centralized: bool = False  # skip centralized baseline training to save time
    skip_local_only: bool = False  # skip the local-only (no-federation) baseline condition
    run_no_warmstart_fl: bool = False  # also run federated training from random init (warm-start ablation, gap #7)

    # Optimization controls.
    batch_size: int = 16
    local_epochs: int = 1
    rounds: int = 8
    clients_per_round_fraction: float = 1.0
    local_sample_ratio_per_epoch: float = 1.0
    sample_with_replacement: bool = False
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    prox_mu: float = 1e-2

    num_classes: int = 5  # fixed: Fibonacci {1,2,3,5,8} → labels {0..4}

    # Head / loss type (gap #12). "ce" = 5-logit softmax + class-weighted CrossEntropy
    # (baseline/ablation, default for backward compatibility). "corn" = 4-logit CORN
    # ordinal head + corn_loss (default for final experiments; pass --head-type corn).
    head_type: str = "ce"

    # Personalized head (gap #11, FedSP-PEFT-P). When True, the classification head
    # stays LOCAL per client (never aggregated); only LoRA-B + embeddings are federated.
    # Targets the project-specific story-point calibration problem (FedPer/FedRep-style).
    personalized_head: bool = False

    # Device.
    device: str = "cuda"

    def __post_init__(self):
        if self.head_type not in ("ce", "corn"):
            raise ValueError(
                f"head_type must be 'ce' or 'corn', got {self.head_type!r}"
            )
