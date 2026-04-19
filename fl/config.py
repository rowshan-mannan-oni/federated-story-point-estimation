from dataclasses import dataclass
from pathlib import Path


@dataclass
class FLConfig:
    data_dir: Path

    # Reproducibility and split controls.
    random_state: int = 42
    test_size: float = 0.2

    # Model controls.
    model_name: str = "prajjwal1/bert-tiny"
    max_length: int = 128
    categorical_emb_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.2
    freeze_encoder: bool = False

    # Optimization controls.
    batch_size: int = 16
    local_epochs: int = 1
    rounds: int = 8
    clients_per_round_fraction: float = 0.25
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4

    # Device.
    device: str = "cuda"
