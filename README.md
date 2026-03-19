# Federated Story Point Estimation (Deep Learning)

This project implements a modular federated learning system for story point regression using TAWOS-style project files.

Input features:
- title
- description
- type
- priority (nullable, mapped to unknown)

Target:
- story_point

## Project structure

- fl/config.py: hyperparameters and runtime config
- fl/data.py: data loading, schema mapping, split, dataset
- fl/model.py: transformer + categorical embedding regression model
- fl/client.py: local client training logic
- fl/server.py: FedAvg orchestration
- fl/metrics.py: regression metrics and formatting
- train_federated_dl.py: end-to-end training script

## Deep model

Default Hugging Face encoder:
- prajjwal1/bert-tiny

This is intentionally lightweight and suitable for federated experiments on a single machine.

## Install

```powershell
pip install -r requirements.txt
```

## Run with all clients in a folder

```powershell
python train_federated_dl.py --data-dir "D:\2026 bachelor Thesis FL\Issues by project dataset"
```

Artifacts are automatically saved after training under:

```text
artifacts/
  centralized/
    model_state.pt
    metadata.json
    tokenizer/
  federated/
    model_state.pt
    metadata.json
    tokenizer/
```

You can override output location:

```powershell
python train_federated_dl.py --data-dir "D:\Federated Learning using DL\data_3_workers" --save-dir "D:\Federated Learning using DL\saved_models"
```

## Run with only 3 workers

```powershell
$src = "D:\2026 bachelor Thesis FL\Issues by project dataset"
$dst = "D:\Federated Learning using DL\data_3_workers"

New-Item -ItemType Directory -Force -Path $dst | Out-Null
Get-ChildItem -Path $src -File |
  Where-Object { $_.Extension -in ".xlsx", ".xls", ".csv" } |
  Sort-Object Name |
  Select-Object -First 3 |
  Copy-Item -Destination $dst -Force

python train_federated_dl.py --data-dir "$dst"
```

## Useful options

```powershell
python train_federated_dl.py ^
  --data-dir "D:\Federated Learning using DL\data_3_workers" ^
  --model-name "prajjwal1/bert-tiny" ^
  --rounds 8 ^
  --local-epochs 1 ^
  --batch-size 16 ^
  --max-length 128 ^
  --fraction 1.0 ^
  --lr 2e-5 ^
  --weight-decay 1e-4 ^
  --seed 42
```

## Outputs

The script reports:
- MAE
- RMSE
- R2
- MAPE
- federated MAE improvement over mean baseline

It also compares:
- mean baseline
- centralized deep model
- federated deep model

## Test a saved model on different data

Use the saved artifact (for example federated) to generate predictions on a new folder.

```powershell
python predict_saved_model.py ^
  --artifact-dir "D:\Federated Learning using DL\artifacts\federated" ^
  --data-dir "D:\2026 bachelor Thesis FL\Issues by project dataset" ^
  --device cuda ^
  --out-csv "D:\Federated Learning using DL\predictions_federated.csv"
```

If the input data contains `story_point`, evaluation metrics are printed.
If labels are missing, the script only saves predictions.

### Example: test 20-epoch federated model

```powershell
python predict_saved_model.py ^
  --artifact-dir "D:\Federated Learning using DL\saved_models_20_global_epochs\federated" ^
  --data-dir "D:\Federated Learning using DL\data_to_test_on" ^
  --device cuda ^
  --out-csv "D:\Federated Learning using DL\predictions_federated_20_epochs.csv"
```
