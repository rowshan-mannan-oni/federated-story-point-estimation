# Dataset Notes

The project uses the TAWOS dataset, a collection of JIRA issues from open-source software projects.

## Data shape

- One CSV file represents one software project.
- Each project is a federated learning client.
- Important fields include `Issue_Key`, `Title`, `Description`, `Story_Point`, `Type`, `Priority`, and `Creation_Date`.
- Missing descriptions are retained as title-only examples.

## Labels

The task uses five ordinal story point classes:

```text
{1, 2, 3, 5, 8}
```

Rows with unsupported story point values are filtered during preparation and recorded in the preprocessing report.

## Preprocessing

Cleaning is performed during export rather than repeatedly during training. The pipeline handles HTML entities, JIRA markup, URLs, issue references, whitespace, missing categories, and very short records.

The following transformations are applied:

- URLs become `[URL]`.
- Issue references become `[ISSUE_REF]`.
- Code and noformat blocks become `[CODE]`.
- HTML tags are removed after entities are decoded.
- Priority values are normalized to a canonical vocabulary.
- Missing type and priority values become `Unknown`.
- Titles and descriptions are preserved in their original case.

The training loader validates that files are already cleaned. It does not silently re-clean raw data.

## Text input

Title and description are combined before tokenization. A separator marks the boundary between the two fields:

```text
title [SEP] description
```

The encoder receives a fixed maximum sequence length. Longer issues are truncated and shorter issues are padded.

## Splitting

Random stratified splitting preserves the local label distribution across train, validation, and test sets. Temporal splitting sorts by `Creation_Date` and trains on earlier issues while evaluating on later issues.

Temporal evaluation is useful because it resembles deployment: the model learns from historical issues and predicts effort for future issues.

## Project heterogeneity

Projects differ in size, vocabulary, issue type, priority usage, and story point calibration. Some projects contain mostly small issues, while others use the upper part of the Fibonacci scale more often.

This non-IID structure motivates federated learning and personalized heads. A shared representation can learn general issue-complexity signals while a local head can preserve project-specific calibration.

## Data quality checks

Before training, verify:

- Required columns are present.
- Story point values belong to the supported set.
- Issue keys are not duplicated.
- Titles are not empty.
- Dates are parseable for temporal experiments.
- Raw JIRA markup is not present at scale.
- Category values use the expected normalized vocabulary.

## Reporting

The preprocessing report records row counts before and after filtering, per-project distributions, and the number of replacements made by each cleaning rule. This report supports reproducibility and should be cited in the thesis Data chapter.
