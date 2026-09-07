# Evaluation Metrics

Results are evaluated per project because project-specific story point conventions make pooled metrics potentially misleading.

## Primary metrics

### Mean Absolute Error

MAE is computed after mapping class indices back to their story point values:

```text
MAE = mean(abs(predicted_story_points - true_story_points))
```

MAE is the main bridge to regression-based story point estimation literature. It is directly interpretable: predicting 3 for a true value of 5 produces an absolute error of 2.

Lower MAE is better.

### Quadratic-weighted Cohen's kappa

Quadratic-weighted kappa measures chance-corrected agreement while penalizing distant ordinal errors more heavily than adjacent errors.

A prediction of 2 for a true value of 3 is less serious than a prediction of 8 for a true value of 3. Kappa captures this difference through quadratic weights.

A value near 0 indicates chance-level agreement. Higher values indicate stronger agreement.

## Supporting metrics

- Accuracy
- Macro-F1
- Weighted F1
- Per-class F1
- Confusion matrix
- Communication cost

### Accuracy

Accuracy is the proportion of exact class matches. It is easy to interpret but does not distinguish between adjacent and distant mistakes.

### Macro-F1

Macro-F1 is the unweighted mean of the F1 score for the five classes. It gives rare classes equal importance and is useful under class imbalance.

### Weighted F1

Weighted F1 weights each class by its observed frequency. It reflects common-class performance more strongly than macro-F1.

### Per-class F1

Per-class F1 shows whether a model performs well on SP 1, SP 2, SP 3, SP 5, and SP 8 individually. It can expose mode collapse or poor performance on rare classes.

### Confusion matrix

The confusion matrix should be inspected for ordinal adjacency. Errors concentrated near the diagonal suggest the model learned the ordering even when exact accuracy is modest.

```text
              Predicted
             1   2   3   5   8
True 1       .   .   .   .   .
True 2       .   .   .   .   .
True 3       .   .   .   .   .
True 5       .   .   .   .   .
True 8       .   .   .   .   .
```

## Communication cost

Communication cost measures the bytes uploaded by clients during federated training. LoRA transmits only a small trainable portion of the encoder, while full fine-tuning transmits the complete model.

The cost should be reported per client, per round, and for the complete experiment. The reduction factor is important for RQ3.

## Per-project reporting

Metrics must be calculated per project and then summarized using means and medians. Avoid pooled metrics for personalized heads, TF-IDF baselines, and other per-project conditions.

Pooling can reward a model for encoding project identity rather than learning transferable estimation behavior. This is especially problematic for kappa and macro-F1.

## Statistical testing

Use the same project splits and seeds for all conditions. Recommended analyses include:

- Wilcoxon signed-rank tests for paired comparisons.
- Friedman test for an omnibus comparison.
- Nemenyi post-hoc comparisons.
- Vargha-Delaney A12 or Cliff's delta for effect size.

Report effect sizes alongside p-values. Statistical significance without practical effect is insufficient.

## Interpretation guidance

Accuracy on story point estimation is inherently limited. Compare the federated result with the centralized ceiling, local-only condition, and simple baselines before interpreting a low score.

MAE and kappa should be emphasized when CORN improves ordinal behavior without producing a large macro-F1 improvement.
