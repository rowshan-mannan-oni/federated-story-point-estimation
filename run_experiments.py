"""
Multi-seed experiment runner (CLAUDE.md gap #4).

For each seed, runs train_federated_dl.py up to twice:

  1. "fedprox" condition: the configured --prox-mu (default 0.01, i.e.
     FedProx), including the majority/local-only/centralized baselines.
  2. "fedavg" condition: prox_mu=0 (FedAvg), with --skip-centralized
     --skip-local-only since those baselines don't depend on prox_mu and
     are already produced by run 1 for the same seed.

Results land in <results-root>/seed_<N>/{fedprox,fedavg}/results/, exactly
the per-project JSON files train_federated_dl.py already writes. Feed
<results-root> to compute_statistics.py (gap #5) for the paired statistical
tests across seeds.

Any arguments not recognized here are forwarded verbatim to
train_federated_dl.py for BOTH runs (e.g. --data-dir, --model-name,
--rounds, --max-length, --lora-*, --warmstart-*, --device, ...). Do not pass
--seed, --prox-mu, --save-dir, --skip-centralized, or --skip-local-only as
passthrough args — the runner sets these itself for each run.

Checkpoint/resume flags (gap #13) are passthrough too — forwarding
--checkpoint-every / --checkpoint-keep / --resume / --resume-from applies them
to every per-seed run, and since each run gets its own per-condition
--save-dir, each resumes only from its OWN checkpoints. Resume is two-layered:
the per-seed skip below (a seed/condition whose federated_per_project.json
already exists is skipped entirely) is the COARSE layer; the within-run
checkpoints under each save-dir are the FINE layer that resumes a run that was
interrupted before it wrote its results JSON.

Example:
    python run_experiments.py --data-dir data_to_train_on --seeds 42-51 \\
        --results-root experiments --model-name microsoft/codebert-base \\
        --max-length 384 --rounds 20 --batch-size 8 \\
        --checkpoint-every 5 --resume
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_federated_dl.py"


def parse_seeds(seeds_arg: str) -> List[int]:
    """Parse '42-51', '42,43,44', or '42' into a list of ints."""
    seeds: List[int] = []
    for part in seeds_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_condition(
    python_exe: str,
    passthrough_args: List[str],
    extra_args: List[str],
    save_dir: Path,
    log_path: Path,
    dry_run: bool,
) -> bool:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, str(TRAIN_SCRIPT), *passthrough_args, *extra_args, "--save-dir", str(save_dir)]
    print(f"[run_experiments] $ {' '.join(cmd)}", flush=True)

    if dry_run:
        return True

    # Force the subprocess to emit UTF-8 on stdout/stderr (Windows defaults to
    # the console codepage, e.g. cp1252, which mismatches the utf-8 decoding
    # below and produces U+FFFD replacement chars that cp1252 can't re-encode).
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()

    return process.returncode == 0


def main() -> None:
    # Avoid UnicodeEncodeError on Windows consoles (cp1252) when relaying
    # subprocess output that was decoded as UTF-8 with errors="replace".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run train_federated_dl.py across multiple seeds for FedProx and FedAvg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seeds", type=str, default="42-51", help="Seed list/range, e.g. '42-51' or '42,43,44'.")
    parser.add_argument("--prox-mu", type=float, default=1e-2, help="FedProx mu for the 'fedprox' run (default 0.01).")
    parser.add_argument("--results-root", type=str, default="experiments", help="Output root directory.")
    parser.add_argument("--skip-fedavg", action="store_true", help="Only run the FedProx (+baselines) condition.")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Also skip centralized + local-only in the fedprox condition (e.g. the personalized RQ4 run, which reuses the shared run's seed-matched baselines).")
    parser.add_argument("--force", action="store_true", help="Re-run even if results already exist for a seed/condition.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands that would run without executing them.")
    args, passthrough = parser.parse_known_args()

    seeds = parse_seeds(args.seeds)
    results_root = Path(args.results_root)
    python_exe = sys.executable

    manifest = {
        "seeds": seeds,
        "prox_mu": args.prox_mu,
        "skip_fedavg": args.skip_fedavg,
        "skip_baselines": args.skip_baselines,
        "passthrough_args": passthrough,
        "runs": [],
    }

    overall_start = time.perf_counter()
    for seed in seeds:
        seed_dir = results_root / f"seed_{seed}"

        fedprox_extra = ["--seed", str(seed), "--prox-mu", str(args.prox_mu)]
        if args.skip_baselines:
            fedprox_extra += ["--skip-centralized", "--skip-local-only"]
        conditions = [
            ("fedprox", fedprox_extra),
        ]
        if not args.skip_fedavg:
            conditions.append(
                ("fedavg", ["--seed", str(seed), "--prox-mu", "0", "--skip-centralized", "--skip-local-only"])
            )

        for condition_name, extra_args in conditions:
            condition_dir = seed_dir / condition_name
            marker = condition_dir / "results" / "federated_per_project.json"

            if marker.exists() and not args.force and not args.dry_run:
                print(f"[run_experiments] seed={seed} {condition_name}: results exist, skipping (use --force to re-run)", flush=True)
                status = "skipped"
            else:
                start = time.perf_counter()
                ok = run_condition(
                    python_exe, passthrough, extra_args,
                    condition_dir, seed_dir / f"{condition_name}.log", args.dry_run,
                )
                elapsed = time.perf_counter() - start
                status = "dry_run" if args.dry_run else ("ok" if ok else "failed")
                print(f"[run_experiments] seed={seed} {condition_name}: {status} ({elapsed:.1f}s)", flush=True)

            manifest["runs"].append({
                "seed": seed, "condition": condition_name,
                "save_dir": str(condition_dir), "status": status,
            })

    manifest["total_elapsed_seconds"] = time.perf_counter() - overall_start

    if not args.dry_run:
        results_root.mkdir(parents=True, exist_ok=True)
        with (results_root / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\n[run_experiments] Done. Manifest written to {results_root / 'manifest.json'}", flush=True)

    failed = [r for r in manifest["runs"] if r["status"] == "failed"]
    if failed:
        print(f"\n[run_experiments] {len(failed)} run(s) failed:", flush=True)
        for r in failed:
            print(f"  seed={r['seed']} condition={r['condition']} -> {r['save_dir']}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
