# Privacy and Federation

Federated learning keeps raw JIRA issue data inside the project that owns it. Clients train locally and send selected model updates to the server.

## What stays local

- Issue titles and descriptions.
- Story point labels.
- Issue type and priority records.
- Training, validation, and test rows.
- Local activations and gradients.
- Personalized prediction heads.

## What is transmitted

The server receives only trainable parameters selected for aggregation:

- LoRA-B matrices.
- Categorical embeddings.
- The classification head in shared-head mode.

The frozen encoder and frozen LoRA-A matrices are not transmitted.

## Aggregation

The server computes a sample-count-weighted average of client parameters:

```text
new_parameter = sum(client_examples * client_parameter) / total_examples
```

This allows larger clients to contribute according to the number of training examples used in their update.

## FedAvg and FedProx

FedAvg is represented by `--prox-mu 0`. Clients optimize their local objective and the server averages the resulting updates.

FedProx adds a proximal penalty that discourages excessive movement away from the global shared model:

```text
local_loss + (mu / 2) * ||local_parameters - global_parameters||^2
```

The penalty applies only to shared aggregatable parameters. In personalized mode, it does not apply to a local head because there is no meaningful global head reference.

## FFA-LoRA

Standard LoRA has an A matrix and a B matrix. Averaging both matrices separately does not generally equal averaging the resulting product.

FFA-LoRA freezes A as a shared random projection and trains only B. Since A is identical across clients, averaging B gives an exact average of the adapter update:

```text
average(B_i * A) = average(B_i) * A
```

This reduces communication and avoids the aggregation error caused by independently averaging A and B.

## Personalized heads

With `--personalized-head`, each project keeps a local classification head. The shared LoRA representation learns general signals about issue complexity, while each head learns the project's local story point calibration.

This is appropriate because story points are unitless and team-calibrated. An 8-point issue in one project may represent a different amount of work than an 8-point issue in another project.

Personalized mode has several consequences:

- There is no single pooled global test metric.
- Per-project metrics are the primary evaluation unit.
- New projects require head initialization and local adaptation.
- Communication is reduced because heads are not aggregated.

## Privacy scope

The system prevents raw issue data from being exchanged during federation. It does not provide a formal guarantee against every model-based privacy attack.

Potential risks include:

- Gradient inversion.
- Membership inference.
- Model inversion.
- Inference from update magnitude or participation patterns.
- A compromised client or aggregation server.

These risks should be acknowledged rather than hidden.

## Secure aggregation

Secure aggregation could prevent the server from seeing individual client updates. Instead, the server would receive only an aggregate protected by cryptographic masking.

This is a possible future extension. It adds protocol complexity and coordination overhead but would reduce the risk of a curious server inspecting one project's update.

## Differential privacy

Differential privacy could provide a formal privacy budget. A typical implementation clips client gradients or updates and adds calibrated noise before aggregation.

The privacy-utility trade-off must be measured experimentally. Stronger privacy generally requires more noise and can reduce MAE, kappa, and macro-F1.

## Warm-start consideration

Warm-start training uses a large project to initialize the model. This is the main stage where one project's data may be processed centrally. The thesis must state this exception clearly and distinguish it from the subsequent federated training of the remaining projects.

A no-warm-start experiment can measure the utility cost of removing this initialization step.

## Non-IID data

Projects differ in label distributions, terminology, issue length, and domain. This non-IID behavior is central to the privacy-preserving design rather than an accidental nuisance.

FedProx limits client drift, while personalized heads allow local calibration without sharing project-specific head parameters.

## Practical safeguards

Recommended deployment safeguards include:

1. Encrypt client-server connections with TLS.
2. Authenticate every participating project.
3. Avoid logging raw text or labels.
4. Restrict access to checkpoints and update files.
5. Rotate credentials and encryption keys.
6. Keep a clear model and data retention policy.
7. Monitor unusual update sizes or participation patterns.
8. Consider secure aggregation for a production deployment.

## Honest privacy claim

The appropriate claim is that the system is privacy-preserving with respect to raw issue data: raw project data remains local while model updates are exchanged.

The system should not claim perfect privacy, differential privacy, or immunity to gradient attacks unless those protections are separately implemented and evaluated.
