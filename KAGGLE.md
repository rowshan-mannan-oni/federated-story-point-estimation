# Running on Kaggle

The code lives on GitHub, but `data_to_train_on/` is **not** in the repo (it's
gitignored, 19 MB). So on Kaggle you **clone the code** and **add the data as a
Kaggle Dataset** separately. GPU: free **T4 x2 or P100 (16 GB)** — plenty for
distilroberta at batch 16 (peak ~4 GB).

There is a ready notebook at [`kaggle_run.ipynb`](kaggle_run.ipynb) — on Kaggle do
**File → Import Notebook → Upload** and pick it, or just paste the cells below.

---

## 1. Upload your data as a Kaggle Dataset

1. Kaggle → **Datasets → New Dataset**.
2. Upload all 19 CSVs from `data_to_train_on/` (drag the folder, or a zip — Kaggle
   unzips it). `preprocessing_report.json` is optional.
3. Title it e.g. **`tawos-story-points`** and Create.

It will mount read-only at `/kaggle/input/tawos-story-points/`. (The notebook
auto-detects the path, so the exact name doesn't matter.)

## 2. Create the notebook

1. Kaggle → **Code → New Notebook**.
2. Right sidebar **Settings**:
   - **Accelerator:** **GPU T4 x2** — do **NOT** use **P100**. The P100 is Pascal
     (sm_60) and Kaggle's current PyTorch no longer ships kernels for it, so any CUDA
     op fails with `CUDA error: no kernel image is available for execution on the
     device`. T4 (sm_75) works. If Kaggle hands you a P100 anyway, restart the session
     until you get a T4.
   - **Internet:** **ON** (required — pip installs + Hugging Face model download).
3. **Add Input →** your `tawos-story-points` dataset.

## 3. Cells

**Clone the code** (feature branch has CORN / personalized / checkpointing / LOPO):

```python
import os
REPO = "/kaggle/working/repo"
if not os.path.exists(REPO):
    !git clone --branch feat/personalization-corn-checkpointing \
        https://github.com/rowshan-mannan-oni/federated-story-point-estimation.git {REPO}
%cd {REPO}
```

**Install only the missing deps** (Kaggle already has torch/transformers — do **not**
`pip install -r requirements.txt`, it would reinstall torch). Also remove Kaggle's
old `torchao`, which `peft>=0.19` rejects during LoRA setup (we don't use torchao):

```python
!pip install -q coral-pytorch "peft>=0.19"
!pip uninstall -q -y torchao
```

> If you already hit `ImportError: Found an incompatible version of torchao`, run the
> `pip uninstall -q -y torchao` line and then **Run → Restart & Run All** (the restart
> clears the stale import).

**Auto-detect the data folder** among your inputs:

```python
import glob, os
counts = {}
for p in glob.glob("/kaggle/input/**/*.csv", recursive=True):
    d = os.path.dirname(p); counts[d] = counts.get(d, 0) + 1
DATA = max(counts, key=counts.get)
print("DATA =", DATA, f"({counts[DATA]} csv files)")
```

**Restore checkpoints from a previous session** (see §4 for the cross-session setup;
harmless on the first run):

```python
import glob, shutil, os
SAVE = "/kaggle/working/artifacts_fedsp_peft_p"
prev = [p for p in glob.glob("/kaggle/input/**/artifacts_fedsp_peft_p", recursive=True)]
if prev and not os.path.exists(SAVE):
    shutil.copytree(prev[0], SAVE)
    print("Restored checkpoints from", prev[0])
else:
    print("Fresh start (no previous checkpoint mounted).")
```

**Smoke test** (bert-tiny, a few minutes — verify it runs before a long job):

```python
!python train_federated_dl.py --data-dir {DATA} \
    --model-name prajjwal1/bert-tiny --rounds 2 --local-epochs 1 --warmstart-epochs 1 \
    --skip-centralized --skip-local-only --save-dir /kaggle/working/smoke --seed 42
```

**Real run** — distilroberta, CORN + personalized head, checkpointing to
`/kaggle/working` so it survives as notebook output. Re-running with `--resume`
continues from the last checkpoint:

```python
!python train_federated_dl.py --data-dir {DATA} \
    --model-name distilroberta-base --max-length 256 \
    --rounds 20 --local-epochs 1 --batch-size 16 --lr 3e-5 --warmstart-lr 3e-5 \
    --prox-mu 1e-2 --head-type corn --personalized-head --generic-head \
    --checkpoint-every 2 --resume \
    --save-dir {SAVE} --seed 42
```

Results land in `{SAVE}/results/` (per-project JSON, `summary.md`, confusion
matrices, `communication_cost.json`) and appear under the notebook's **Output** tab.

## 4. Resume across sessions (Kaggle wipes everything between sessions)

Kaggle only keeps `/kaggle/working` if you **save it as output**, and it's read-only
next time. The loop:

1. Run the notebook; when your session is about to end, **Save Version** (Quick Save)
   — this stores `/kaggle/working` (including `artifacts_fedsp_peft_p/checkpoints/`).
2. Next session: **Add Input → Notebooks →** your own notebook, add its latest
   output. It mounts under `/kaggle/input/<your-notebook>/`.
3. Run all cells. The **restore** cell copies the saved checkpoints into
   `/kaggle/working`, and `--resume` continues from the last round. Warm-start is
   loaded from cache, so no time is re-paid.

(Alternative: use `--resume-from` pointing directly at a `/kaggle/input/.../round_<k>`
dir.)

## 5. Download results

Output tab → download `artifacts_fedsp_peft_p/` (or zip it: `!cd /kaggle/working && zip -qr results.zip artifacts_fedsp_peft_p/results`). Feed `results/` to `dashboard/` locally, or `compute_statistics.py` for multi-seed stats.

## Honest note on the GPU quota

Free Kaggle GPU is **~30 h/week**, max ~12 h/session. A single distilroberta **seed**
(FedProx incl. baselines + FedAvg) is roughly **5–8 h** on a T4, so ~1–2 seeds per
session. The **full 10-seed sweep will span multiple weeks** on the quota. Practical
options:

- Use Kaggle for the **smoke test, a few seeds, and the LOPO onboarding experiment**;
  run the rest on your 24 GB lab PC (see `commands.txt` §9).
- Or split seeds across the two machines and merge the `experiments/seed_<N>/` folders
  before `compute_statistics.py`.