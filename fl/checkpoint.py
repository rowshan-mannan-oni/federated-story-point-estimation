"""Checkpoint / resume for federated, centralized, and local-only training (gap #13).

Design contract (see CLAUDE.md gap #13):
- Saves TRAINABLE-ONLY states — never the frozen backbone. On resume the backbone is
  reconstructed by re-instantiating the model (pretrained weights + frozen LoRA-A), so
  checkpoints stay small (same trick as per-client transfer).
- Each checkpoint dir holds `state.pt` (torch-pickled payload: trainable states, RNG
  blob, step index, round/epoch history, best-on-val bookkeeping) and `meta.json`
  (human-readable: format version, step, and the full FLConfig).
- Rotation keeps at most `keep` numbered checkpoints, and ALWAYS keeps `latest/` and
  `best/` (mirrors of the newest and best-scoring checkpoints respectively).
- `load_checkpoint` refuses to resume across an incompatible config (head_type, encoder,
  personalized_head, and other shape-critical params) but allows `rounds` to change
  (raising it to train longer is legal).

The RNG capture/restore covers python `random`, numpy global, torch (+cuda), and any
named numpy Generators passed in (e.g. the server's client-selection Generator) so a
resumed run reproduces client selection and data sampling bit-for-bit.
"""
from __future__ import annotations

import dataclasses
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fl.config import FLConfig

CHECKPOINT_FORMAT_VERSION = 1

# Config fields that MUST match to resume: they change parameter shapes or training
# semantics, so resuming across them would corrupt or fail the state load. `rounds` is
# deliberately NOT here — raising it to continue training longer is allowed.
COMPAT_KEYS = (
    "model_name",          # the encoder ("encoder" in the spec)
    "head_type",
    "personalized_head",
    "num_classes",
    "categorical_emb_dim",
    "hidden_dim",
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_target_modules",
    "ffa_lora",
    "freeze_encoder",
)


# --------------------------------------------------------------------------------------
# Path helpers (layout: <save-dir>/checkpoints/{federated|centralized|local_only/<proj>})
# --------------------------------------------------------------------------------------
def checkpoints_root(save_dir: Path) -> Path:
    return Path(save_dir) / "checkpoints"


def federated_ckpt_root(save_dir: Path) -> Path:
    return checkpoints_root(save_dir) / "federated"


def centralized_ckpt_root(save_dir: Path) -> Path:
    return checkpoints_root(save_dir) / "centralized"


def local_only_ckpt_root(save_dir: Path, project: str) -> Path:
    return checkpoints_root(save_dir) / "local_only" / project


# --------------------------------------------------------------------------------------
# RNG capture / restore
# --------------------------------------------------------------------------------------
def capture_rng_state(
    generators: Optional[Dict[str, np.random.Generator]] = None,
) -> Dict[str, Any]:
    """Snapshot python/numpy/torch(+cuda) global RNG plus any named numpy Generators."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if generators:
        state["generators"] = {
            name: gen.bit_generator.state for name, gen in generators.items()
        }
    return state


def _as_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(torch.uint8).cpu()
    return torch.tensor(value, dtype=torch.uint8)


def restore_rng_state(
    state: Dict[str, Any],
    generators: Optional[Dict[str, np.random.Generator]] = None,
) -> None:
    """Restore RNG state captured by `capture_rng_state`, in place."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_byte_tensor(s) for s in state["torch_cuda"]]
        # Only restore if device count matches what was captured; otherwise skip safely.
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
    if generators and "generators" in state:
        for name, gen in generators.items():
            saved = state["generators"].get(name)
            if saved is not None:
                gen.bit_generator.state = saved


# --------------------------------------------------------------------------------------
# Config serialization + compatibility
# --------------------------------------------------------------------------------------
def config_to_dict(config: FLConfig) -> Dict[str, Any]:
    """JSON-serializable snapshot of an FLConfig (paths -> str, tuples -> lists)."""
    d = dataclasses.asdict(config)
    d["data_dir"] = str(d["data_dir"])
    if isinstance(d.get("lora_target_modules"), tuple):
        d["lora_target_modules"] = list(d["lora_target_modules"])
    return d


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def assert_config_compatible(saved_config: Dict[str, Any], config: FLConfig) -> None:
    """Raise ValueError if `config` is incompatible with a checkpoint's saved config.

    Compares the shape/semantics-critical COMPAT_KEYS only; `rounds` and other tuning
    knobs may differ.
    """
    mismatches: List[str] = []
    for key in COMPAT_KEYS:
        old = _normalize(saved_config.get(key))
        new = _normalize(getattr(config, key, None))
        if old != new:
            mismatches.append(f"  - {key}: checkpoint={old!r} != current={new!r}")
    if mismatches:
        raise ValueError(
            "Cannot resume: the current config is incompatible with the checkpoint on "
            "these fields (they change parameter shapes/semantics):\n"
            + "\n".join(mismatches)
        )


# --------------------------------------------------------------------------------------
# Save / load / rotate
# --------------------------------------------------------------------------------------
def _write_one(ckpt_dir: Path, payload: Dict[str, Any], config: FLConfig, step: int, step_prefix: str) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_dir / "state.pt")
    meta = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "step_prefix": step_prefix,
        "config": config_to_dict(config),
    }
    with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def _mirror(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "state.pt", dst / "state.pt")
    shutil.copy2(src / "meta.json", dst / "meta.json")


def _numbered_dirs(ckpt_root: Path, step_prefix: str) -> List[Path]:
    pattern = re.compile(rf"^{re.escape(step_prefix)}_(\d+)$")
    dirs = []
    for child in ckpt_root.iterdir() if ckpt_root.exists() else []:
        if child.is_dir():
            m = pattern.match(child.name)
            if m:
                dirs.append((int(m.group(1)), child))
    dirs.sort(key=lambda t: t[0])
    return [d for _, d in dirs]


def _rotate(ckpt_root: Path, keep: int, step_prefix: str) -> None:
    """Keep the newest `keep` numbered checkpoints; latest/ and best/ are never rotated."""
    dirs = _numbered_dirs(ckpt_root, step_prefix)
    excess = len(dirs) - max(keep, 0)
    for old in dirs[:max(excess, 0)]:
        shutil.rmtree(old, ignore_errors=True)


def save_checkpoint(
    ckpt_root: Path,
    step: int,
    payload: Dict[str, Any],
    config: FLConfig,
    *,
    is_best: bool = False,
    keep: int = 2,
    step_prefix: str = "round",
) -> Path:
    """Write a checkpoint into `ckpt_root/<step_prefix>_<step>/`, mirror to latest/ (and
    best/ if `is_best`), then rotate numbered checkpoints down to `keep`.

    Returns the numbered checkpoint dir.
    """
    ckpt_root = Path(ckpt_root)
    step_dir = ckpt_root / f"{step_prefix}_{step}"
    _write_one(step_dir, payload, config, step, step_prefix)
    _mirror(step_dir, ckpt_root / "latest")
    if is_best:
        _mirror(step_dir, ckpt_root / "best")
    _rotate(ckpt_root, keep=keep, step_prefix=step_prefix)
    return step_dir


def load_checkpoint(
    ckpt_dir: Path,
    config: Optional[FLConfig] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load (payload, meta) from a checkpoint dir. If `config` is given, verify
    compatibility first (raises ValueError on mismatch)."""
    ckpt_dir = Path(ckpt_dir)
    with (ckpt_dir / "meta.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if config is not None:
        assert_config_compatible(meta.get("config", {}), config)
    payload = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
    return payload, meta


def latest_checkpoint_dir(ckpt_root: Path) -> Optional[Path]:
    """Return `ckpt_root/latest` if it holds a checkpoint, else None."""
    latest = Path(ckpt_root) / "latest"
    if (latest / "state.pt").exists() and (latest / "meta.json").exists():
        return latest
    return None
