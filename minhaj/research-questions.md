# Research Questions

## RQ1: Feasibility and privacy cost

How does FedSP-PEFT compare with centralized pooling and per-project local training for story point estimation?

## RQ2: Personalization

Does estimation improve when each project keeps its own prediction head instead of federating the head with the shared representation?

## RQ3: Parameter efficiency

How does federating only LoRA adapters compare with fully fine-tuning and federating the whole encoder in estimation quality and communication cost?

## Case study

Leave-one-project-out onboarding measures how much labeled history a new project needs before head-only adaptation becomes useful. This is reported as a case study rather than a separate research question.
