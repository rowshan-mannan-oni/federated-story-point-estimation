import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fl import checkpoint as ck
from fl.client import ClientOutput, FederatedClient
from fl.config import FLConfig
from fl.server import FedProxServer


def check_isolated_rng() -> bool:
    """Original checks: client selection and per-epoch sampling are reproducible/varied."""
    seed = 42
    rng = np.random.default_rng(seed)

    total_clients = 5
    per_round = 3
    rounds = 4

    selected_each_round = []
    for round_idx in range(rounds):
        picked = FedProxServer.choose_client_indices(total_clients=total_clients, per_round=per_round, rng=rng)
        picked_list = [int(x) for x in picked.tolist()]
        selected_each_round.append(picked_list)
        print(f"Round {round_idx + 1}: selected_clients={picked_list}")

    unique_round_patterns = {tuple(item) for item in selected_each_round}
    print(f"Client selection differs across rounds: {len(unique_round_patterns) > 1}")

    epoch_rng = np.random.default_rng(seed)
    num_examples = 12
    local_epochs = 4
    epoch_indices = []
    for epoch_idx in range(local_epochs):
        sampled = FederatedClient.sample_epoch_indices(
            num_examples=num_examples,
            sample_ratio_per_epoch=0.5,
            sample_with_replacement=False,
            rng=epoch_rng,
        )
        sampled_list = [int(x) for x in sampled.tolist()]
        epoch_indices.append(sampled_list)
        print(f"Epoch {epoch_idx + 1}: sampled_count={len(sampled_list)} indices={sampled_list}")

    unique_epoch_patterns = {tuple(item) for item in epoch_indices}
    print(f"Epoch sampling differs across epochs: {len(unique_epoch_patterns) > 1}")
    return len(unique_round_patterns) > 1 and len(unique_epoch_patterns) > 1


# --------------------------------------------------------------------------------------
# Resume-reproducibility check (gap #13, step 3.6)
# --------------------------------------------------------------------------------------
class _TinyModel(nn.Module):
    """Minimal all-trainable stand-in for the real model — lets us exercise the server's
    checkpoint/resume RNG plumbing on CPU without loading a transformer."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4))
        self.b = nn.Parameter(torch.zeros(3))


class _FakeClient:
    """Client whose update depends on BOTH its per-client seed AND the global torch RNG,
    so the reproducibility check only passes if the server's client-selection Generator
    AND the global torch RNG are both captured and restored on resume."""

    def __init__(self, client_id: str, num_examples: int) -> None:
        self.client_id = client_id
        self.num_examples = num_examples

    def train_local(self, global_state, model_factory, device, epochs,
                    sample_ratio_per_epoch, sample_with_replacement,
                    learning_rate, weight_decay, prox_mu, seed, personalized_head=False):
        g = np.random.default_rng(seed)
        state = {}
        for key, tensor in global_state.items():
            seed_noise = torch.from_numpy(g.standard_normal(size=tuple(tensor.shape)).astype("float32"))
            torch_noise = torch.randn(tensor.shape)  # consumes the GLOBAL torch RNG
            state[key] = (tensor + 0.01 * seed_noise + 0.01 * torch_noise).clone()
        return ClientOutput(state_dict=state, num_examples=self.num_examples, loss=float(g.random()))


def _run_federated(rounds, seed, ckpt_root, checkpoint_every, resume_payload=None):
    # Seed the globals identically before each run; on resume the server overwrites these
    # from the checkpoint, so the resumed rounds pick up exactly where the straight run was.
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    clients = [_FakeClient(f"c{i}", num_examples=10 + i) for i in range(5)]
    initial_state = {k: v.detach().clone() for k, v in _TinyModel().state_dict().items()}
    cfg = FLConfig(data_dir=Path("."))

    server = FedProxServer(model_factory=_TinyModel, clients=clients, random_state=seed)
    fed_state, history, _ = server.train(
        rounds=rounds,
        clients_per_round_fraction=0.6,
        local_epochs=1,
        local_sample_ratio_per_epoch=1.0,
        sample_with_replacement=False,
        learning_rate=0.0,
        weight_decay=0.0,
        prox_mu=0.0,
        device=torch.device("cpu"),
        initial_state=initial_state,
        val_loader=None,
        val_labels=None,
        num_classes=5,
        checkpoint_every=checkpoint_every,
        checkpoint_keep=2,
        ckpt_root=ckpt_root,
        checkpoint_config=cfg,
        resume_payload=resume_payload,
    )
    return fed_state, history


def check_resume_reproducibility() -> bool:
    """A 4-round straight run must equal a 2-round + resume + 2-round run, in both the
    client-selection sequence and the final aggregated state."""
    tmp = Path(tempfile.mkdtemp())
    try:
        print("\n[Resume] Running 4 rounds straight ...")
        straight_state, straight_hist = _run_federated(4, seed=7, ckpt_root=tmp / "straight", checkpoint_every=2)

        print("[Resume] Running 2 rounds, then resuming for 2 more ...")
        _run_federated(2, seed=7, ckpt_root=tmp / "interrupt", checkpoint_every=1)
        payload, _ = ck.load_checkpoint(ck.latest_checkpoint_dir(tmp / "interrupt"))
        resumed_state, resumed_hist = _run_federated(
            4, seed=7, ckpt_root=tmp / "interrupt", checkpoint_every=1, resume_payload=payload
        )

        straight_sel = [h["selected_clients"] for h in straight_hist]
        resumed_sel = [h["selected_clients"] for h in resumed_hist]
        sel_match = straight_sel == resumed_sel
        state_match = all(torch.allclose(straight_state[k], resumed_state[k]) for k in straight_state)

        print(f"Straight selection: {straight_sel}")
        print(f"Resumed  selection: {resumed_sel}")
        print(f"Client-selection sequences match (4 straight vs 2+resume+2): {sel_match}")
        print(f"Final aggregated states match: {state_match}")
        return sel_match and state_match
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    print("=== Isolated RNG checks ===")
    ok_isolated = check_isolated_rng()

    print("\n=== Resume reproducibility check ===")
    ok_resume = check_resume_reproducibility()

    print(f"\nAll checks passed: {ok_isolated and ok_resume}")
    if not (ok_isolated and ok_resume):
        sys.exit(1)


if __name__ == "__main__":
    main()
