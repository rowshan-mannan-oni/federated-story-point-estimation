# Experiment Workflow

## Standard sequence

1. Prepare and validate cleaned project CSV files.
2. Warm-start on the largest project when the run configuration enables it.
3. Train local-only and centralized comparison conditions.
4. Train federated conditions with fixed seeds.
5. Evaluate each condition per project.
6. Compute statistics and communication costs.

## Final defaults

- Encoder: `microsoft/codebert-base`
- Head: CORN
- Federated optimizer: FedProx with `prox_mu=0.01`
- LoRA: FFA-LoRA
- Seeds: 42, 43, and 44
- Splits: temporal primary, random robustness check

## Development smoke run

Use `prajjwal1/bert-tiny` only to verify plumbing. Its metrics must not be interpreted as thesis results.

```bash
python train_federated_dl.py --data-dir data_to_train_on \\
  --model-name prajjwal1/bert-tiny --max-length 64 \\
  --rounds 2 --local-epochs 1 --warmstart-epochs 1 \\
  --skip-centralized --skip-local-only
```

## Reporting

Report MAE and quadratic-weighted Cohen's kappa as primary metrics, with macro-F1, accuracy, per-class F1, confusion matrices, and communication cost as supporting evidence. Use per-project results and avoid pooled metrics for personalized or per-project baselines.
