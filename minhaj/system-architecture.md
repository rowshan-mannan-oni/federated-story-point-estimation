# System Architecture

## Client

Each client is one software project with private training, validation, and test data.

The model combines:

- A frozen CodeBERT encoder for final experiments.
- FFA-LoRA adapters, where matrix A is frozen and matrix B is trainable.
- Trainable issue-type and priority embeddings.
- Either a shared classification head or a local personalized head.

## Server

The server aggregates only trainable shared parameters using sample-count-weighted averaging. In personalized mode, client heads stay local and are never pushed back to participants.

## Training objectives

- FedAvg is represented by `--prox-mu 0`.
- FedProx adds a proximal penalty over shared parameters.
- CORN models story points as ordered thresholds.
- Cross-entropy remains available as an ablation.

## Privacy scope

Raw issue data does not leave its project. Adapter and embedding updates are still transmitted, so differential privacy and secure aggregation remain possible future protections.
