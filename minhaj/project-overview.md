# FedSP-PEFT Project Overview

FedSP-PEFT is a privacy-preserving federated learning system for story point estimation from JIRA issue text.

## Goal

Each software project acts as a federated client. Raw issue data remains local while the clients collaboratively train a model that predicts one of the story point classes `{1, 2, 3, 5, 8}`.

## Dataset

The project uses the TAWOS dataset, which contains cleaned JIRA issues from open-source projects. Each project is represented by its own CSV file.

## Main contribution

The thesis studies the gap between local-only, federated, and centralized training under a privacy constraint. It also evaluates personalized prediction heads and communication cost.

## Important framing

Story points are ordinal and team-calibrated. An absolute accuracy score is not the only signal: MAE, quadratic-weighted Cohen's kappa, confusion-matrix adjacency, personalization, and communication efficiency are central to the evaluation.
