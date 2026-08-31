# Evaluation Metrics

Results are evaluated **per project** because project-specific story point conventions make pooled metrics potentially misleading. A team's "8" is not the same as another team's "8"; averaging across projects conflates this calibration difference with actual prediction error.

## Primary metrics

Two metrics are reported as primary because they address the two key concerns in story point estimation: absolute effort prediction and ordinal structure.

### Mean Absolute Error (MAE)

**Definition**: Average absolute error after mapping class predictions back to story point values.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|$$

where $\hat{y}_i$ and $y_i$ are the predicted and true story point values.

**Why primary**: This metric bridges to the regression-based SPE literature (Deep-SE, GPT2SP, etc.) and directly measures effort prediction accuracy in business terms. A model that predicts "3" when the true value is "5" has MAE = 2, directly interpretable as a 2-point underestimate.

**Interpretation**: Lower MAE is better. Because the story point deck is {1, 2, 3, 5, 8}, the maximum possible MAE on a single issue is 7 (predicting 1 when the true value is 8). On an imbalanced dataset, a naive baseline (always predict the most frequent class) produces a non-zero MAE that serves as a floor.

**Computation in the codebase**: Predictions are mapped through `INV_LABEL_MAP` (the inverse of the label encoding) to recover story point values, then MAE is computed.

### Quadratic-weighted Cohen's Kappa

**Definition**: A chance-corrected agreement metric that weights disagreement by the square of the distance between classes.

$$\kappa = 1 - \frac{p_e - p_o}{1 - p_e}$$

where $p_o$ is the observed agreement and $p_e$ is the expected agreement by chance. The quadratic weighting means a miss from 8-when-truth-is-3 (distance 5) incurs a penalty of $5^2 = 25$, while 3-when-truth-is-2 (distance 1) incurs only $1^2 = 1$.

**Why primary**: Kappa is chance-corrected, so a model that encodes project identity (always outputting the modal class for each project) still scores poorly. Quadratic weighting reflects the ordinal structure: distant misses are genuinely worse for planning (a team estimated 8 sprints but got 3 = bad surprise; 3 estimated, got 2 = acceptable).

**Interpretation**: Kappa ranges from -1 (perfect disagreement) to 1 (perfect agreement); 0 means chance agreement. Typical benchmarks:
- κ < 0.2: slight agreement
- 0.2–0.4: fair agreement
- 0.4–0.6: moderate agreement
- 0.6–0.8: substantial agreement
- 0.8+: near-perfect agreement

On TAWOS, even strong deep models achieve κ in the 0.3–0.5 range, underscoring the inherent difficulty of the task.

**Computation in the codebase**: Uses `sklearn.metrics.cohen_kappa_score(..., weights="quadratic")` with class indices mapped to ordinal values.

## Supporting metrics

### Accuracy

Simple per-project accuracy: proportion of correct class predictions.

$$\text{Accuracy} = \frac{\text{# correct}}{\text{# total}}$$

**Limitation**: Treats all misses equally. A model that always predicts class 3 (the most frequent class) achieves ~18% accuracy on TAWOS; this is a low floor but not negligible.

**Use**: Included for completeness and comparison with published SPE results, but secondary to MAE and κ because it ignores ordinal structure and class imbalance.

### Macro-F1

Macro-averaged F1 score: unweighted mean of per-class F1 scores.

$$\text{Macro-F1} = \frac{1}{5} \sum_{c \in \{1,2,3,5,8\}} F1_c$$

where $F1_c$ is the F1 score for class $c$.

**Use**: Robust to class imbalance (weights all classes equally regardless of frequency). Useful for spotting which classes are consistently misclassified.

**Limitation**: Blind to ordinal adjacency. A model that predicts 2 when the truth is 3 scores the same F1 impact as predicting 8 when the truth is 3, which is wrong.

### Weighted F1

F1 score weighted by the true class frequency:

$$\text{Weighted F1} = \sum_{c} P(\text{class} = c) \cdot F1_c$$

**Use**: Reflects the class distribution and is biased toward the most frequent classes (SP 1 and 2).

### Per-class F1 and Recall

Breakdown of F1 and recall by individual story point class. Essential for understanding which classes are easy (usually SP 1, the most frequent) and which are hard (usually SP 8, the rarest).

**Use**: Identify if the model has mode collapse (predicting mostly one class) or if specific classes are systematically misclassified.

### Confusion Matrix

A 5×5 matrix showing the distribution of predictions vs. true labels:

```
           Pred:1  Pred:2  Pred:3  Pred:5  Pred:8
True:1      ...     ...     ...     ...     ...
True:2      ...     ...     ...     ...     ...
True:3      ...     ...     ...     ...     ...
True:5      ...     ...     ...     ...     ...
True:8      ...     ...     ...     ...     ...
```

**Use**: Reveals error patterns:
- Diagonal dominance = good (correct predictions)
- Off-diagonal concentration = systematic bias (e.g., the model tends to underestimate)
- Adjacency structure = ordinal behavior (errors concentrated near the diagonal, not far from it)

Even when macro-F1 is flat, a confusion matrix concentrated on the diagonal's neighbors indicates that the model learned the ordinal scale, even if it can't predict the exact class.

## Communication cost

**Definition**: Total bytes transmitted from clients to server per federated round, summed across all rounds and clients.

$$\text{Cost (bytes)} = \text{(model size)} \times \text{(# parameters per client)} \times \text{(# rounds)} \times \text{(# clients)}$$

For LoRA-only: the transmitted model is only the trainable B matrices and embeddings, ~0.5–1% of the full encoder size.

For full fine-tuning: the entire encoder is transmitted, orders of magnitude larger.

**Use**: Justifies the parameter-efficient approach (RQ3). Communication cost is the primary bottleneck in real federated deployments over limited-bandwidth networks.

**Computation in the codebase**: Calculated once per run in `train_federated_dl.py` from a probe model and saved to `results/communication_cost.json`.

## Per-project vs. pooled reporting

### Why not pooled?

Pooling (computing metrics across all projects at once) is misleading for this task because:

1. **Project-specific calibration**: An "8" in one team means 2 weeks; in another team, it means 5 days. Cross-project agreement on the absolute value of a label doesn't make sense.

2. **Metric inflation from project encoding**: A constant baseline that always predicts SP 3 for every project scores:
   - κ = 0.0000 in every individual project (chance agreement, correct)
   - κ = 0.5006 when pooled (because it has encoded project identity implicitly in the test set distribution)

This pooling artifact makes it impossible to compare per-project models (like personalized-head) against cross-project models on a fair scale.

### Correct reporting

Report per-project results for all metrics; then report mean and median across projects:

```
Project         MAE    κ     Macro-F1
Proj-A          1.2    0.35  0.42
Proj-B          0.9    0.48  0.51
Proj-C          1.5    0.28  0.38
...
Mean            1.3    0.38  0.44
Median          1.2    0.36  0.42
```

**Important exclusion**: Conditions whose model is per-project (personalized-head, TF-IDF+SVM, median-SP baseline) emit per-project results only — NO pooled entry. Compute the mean/median of per-project metrics, not the pooled metric.

## Statistical testing

Given 3 seeds and 18 projects per seed, the unit of analysis is (seed, project). With n=3×18=54 paired observations, significance is assessed via:

- **Wilcoxon signed-rank test**: pairwise comparison of two conditions (e.g., FedProx vs. FedAvg) using the 54 paired differences.
- **Friedman test**: tests whether all conditions come from the same distribution (omnibus test across 6+ conditions).
- **Nemenyi post-hoc**: pairwise comparisons after Friedman, controlling for multiple comparisons.
- **Vargha-Delaney Â or Cliff's delta**: effect size to complement p-values.

Required by empirical SE venues; p-values alone are insufficient.

## Accuracy expectations and framing

Tawosi et al. (2023) showed that even state-of-the-art deep models for story point estimation barely beat naive baselines. Absolute accuracy on this task is low for everyone.

**Do not panic if macro-F1 is ~0.45.** Instead:

1. Check the centralized ceiling (what's the pooled model's macro-F1?). If centralized is also ~0.45, the federated arm has not lost ground to privacy.
2. Check the confusion matrix adjacency. If errors cluster on the diagonal's neighbors, the model learned ordinal structure even if classification accuracy is flat.
3. Interpret MAE and κ in context of Tawosi's baselines. If the deep federated model beats TF-IDF+SVM by 0.05 in κ, that is meaningful.

The thesis contribution is not absolute accuracy; it is the federated-vs-centralized gap, the personalization finding, and communication efficiency.

## Detailed metric calculations in code

### MAE computation

```python
def compute_mae(predictions, true_labels, inv_label_map):
    \"\"\"
    predictions: shape (N,) or (N, 5) with class logits/indices
    true_labels: shape (N,) with class indices
    inv_label_map: dict mapping class_index -> story_point_value
    \"\"\"
    # Decode predictions to class indices if needed
    if len(predictions.shape) > 1:
        pred_classes = predictions.argmax(dim=1)
    else:\n        pred_classes = predictions\n\n    # Map to story point values\n    pred_values = torch.tensor([inv_label_map[c.item()] for c in pred_classes])\n    true_values = torch.tensor([inv_label_map[t.item()] for t in true_labels])\n\n    # Compute absolute differences\n    return torch.abs(pred_values - true_values).mean().item()\n```\n\n### Kappa computation\n\nUsing sklearn, which handles the quadratic weighting internally:\n\n```python\nfrom sklearn.metrics import cohen_kappa_score\n\nkappa = cohen_kappa_score(\n    y_true=true_labels.numpy(),\n    y_pred=pred_labels.numpy(),\n    weights='quadratic'\n)\n```\n\nThe quadratic weight matrix $w_{ij}$ for disagreement between classes $i$ and $j$ is:\n\n$$w_{ij} = 1 - \\frac{(i - j)^2}{(k - 1)^2}$$\n\nwhere $k = 5$ is the number of classes.\n\n### Confusion matrix interpretation\n\nFor a 5-class problem, the confusion matrix is 5×5. Key patterns:\n\n- **Diagonal dominance**: correct predictions (desired)\n- **Near-diagonal concentration**: ordinal errors (acceptable)\n- **Off-diagonal corners**: far misses (bad; suggests the model didn't learn the scale)\n\nExample interpretation:\n\n```\nPredicted:     1      2      3      5      8\nTrue 1:        800     90     10      0      0   (mostly correct, some type-II errors)\nTrue 2:         85    750     120     30      0   (good, nearby errors)\nTrue 3:         10    105     650    180     30   (some confusion with 5)\nTrue 5:          0     20     140    620    120   (broader spread, harder class)\nTrue 8:          0      0      25    140    620   (rarest class, more errors)\n```\n\nIf errors were random, the matrix would be nearly uniform. The above pattern (near-diagonal) indicates ordinal behavior.\n\n## Ablation metrics\n\nWhen comparing variants, the changes in metrics reveal what different components contribute:\n\n### CORN vs. CrossEntropy head\n\n**Expected pattern**:\n- CE: higher accuracy on the modal class (SP 1, 2), lower on rare classes\n- CORN: lower accuracy overall, but higher MAE due to ordinal structure; \u03ba should move more than F1\n\nThis asymmetry (metrics move differently) is itself the finding: ordinal modeling helps with adjacent classes even if overall accuracy drops.\n\n### FedProx vs. FedAvg\n\n**Expected pattern**:\n- FedAvg: faster initial convergence, potential mode collapse (predicting modal class for all projects)\n- FedProx: slower convergence, better final accuracy on non-IID data\n\nOn TAWOS with non-IID projects, FedProx typically outperforms FedAvg by 2\u20135% in \u03ba.\n\n### Personalized-head vs. shared-head\n\n**Expected pattern**:\n- Shared-head: lower per-project macro-F1 (the shared head compromises to satisfy all projects)\n- Personalized-head: higher per-project macro-F1 (each head specializes); \u03ba improves more than accuracy\n\nThis is the core RQ2 finding: personalization pays off most on projects with very different label semantics.\n\n## Reporting metrics across conditions\n\n### Standard results table\n\nA complete results table includes:\n\n```\nCondition       MAE    Kappa  Macro-F1  Accuracy  F1-1  F1-2  F1-3  F1-5  F1-8\nMedian-SP      1.8     0.00    0.18      0.30     0.30  0.26  0.15  0.00  0.00\nTF-IDF+SVM     1.4     0.32    0.40      0.48     0.52  0.46  0.35  0.22  0.05\nLocal-only     1.2     0.45    0.52      0.58     0.62  0.58  0.48  0.40  0.28\nCentralized    1.1     0.52    0.58      0.63     0.68  0.62  0.55  0.48  0.35\nFedAvg         1.15    0.48    0.54      0.60     0.64  0.60  0.52  0.44  0.30\nFedProx        1.12    0.51    0.56      0.62     0.66  0.61  0.53  0.46  0.32\nFedProx+P-head 1.10    0.53    0.57      0.63     0.67  0.62  0.54  0.48  0.34\n```\n\nNote: all per-project, then report mean across 18 projects.\n\n### Pairwise significance\n\nWhen comparing two conditions, report:\n\n1. **Paired difference**: (Condition A - Condition B) for each (seed, project)\n2. **Wilcoxon p-value**: is the median difference significantly different from 0?\n3. **Effect size**: Vargha-Delaney \u00c2, interpreted as \"probability that A > B\"\n\nExample output:\n\n```\nFedProx vs. FedAvg on Macro-F1:\n  Mean difference: 0.020 (FedProx higher)\n  Wilcoxon p-value: 0.031 *\n  Vargha-Delaney A: 0.62 (small-to-medium effect)\n```\n\n## Metric limitations and caveats\n\n### Accuracy is not enough\n\nAccuracy treats all errors equally. On TAWOS:\n- Predicting 2 when true is 1: -1 point error\n- Predicting 8 when true is 1: -7 point error\n\nAccuracy rates both as \"1 wrong prediction,\" but the business impact is very different. Use MAE and \u03ba to capture this asymmetry.\n\n### Kappa can mislead on tiny projects\n\nFor very small projects (< 50 test examples), kappa's denominator becomes tiny, and statistical noise dominates. Report confidence intervals or use bootstrap resampling.\n\n### Macro-F1 ignores class imbalance direction\n\nMacro-F1 weights all classes equally, so:\n- High macro-F1 on a balanced dataset means the model is uniformly good\n- High macro-F1 on an imbalanced dataset could mean the model is good on rare classes and mediocre on frequent ones\n\nAlways inspect per-class F1 scores, not macro-F1 alone.\n\n### Communication cost has diminishing returns\n\nReporting bytes transmitted is necessary but incomplete. What matters is:\n- **Latency**: time for aggregation (depends on network, not model size)\n- **Bandwidth**: sustained throughput (LoRA updates are small but numerous)\n- **Energy**: on mobile, transmission is expensive; LoRA reduces this significantly\n\n## Creating figures\n\n### Convergence curves\n\nPlot per-round validation macro-F1 for federated runs:\n\n```\nMacro-F1\n0.60 |     .---FedProx\n0.55 |    /\n0.50 |   /-----FedAvg\n0.45 |  /\n0.40 | /\n     +----+----+----+----+---- Rounds\n     1    5   10   15   20\n```\n\nFedProx should show slower initial convergence but better final accuracy; FedAvg should converge faster but plateau lower.\n\n### Confusion matrices as heatmaps\n\nFor each condition, plot the normalized confusion matrix (rows = true, columns = predicted):\n\n```\nTrue\\Pred   1     2     3     5     8\n  1       0.80  0.12  0.05  0.02  0.01\n  2       0.10  0.75  0.12  0.02  0.01\n  3       0.02  0.10  0.65  0.18  0.05\n  5       0.00  0.02  0.14  0.62  0.22\n  8       0.00  0.00  0.03  0.18  0.79\n```\n\nHigh diagonal + near-diagonal concentration indicates good ordinal learning.\n\n### Per-project breakdowns\n\nBar chart of MAE or \u03ba per project for each condition, sorted by project size or difficulty:\n\nThis reveals which projects are \"easy\" (e.g., Lsstcorp with consistent terminology) and which are \"hard\" (e.g., projects with inconsistent naming or rapid estimation drift).\n\n### Effect size visualization\n\nPlot Vargha-Delaney \u00c2 values with confidence intervals. Values around 0.5 indicate no effect; <0.4 or >0.6 indicate meaningful difference.

## Practical workflow for computing and reporting metrics

### Step 1: Run experiments with 3 seeds

```bash
python run_experiments.py \
  --data-dir data_to_train_on \
  --seeds 42 43 44 \
  --model-name microsoft/codebert-base \
  --head-type corn \
  --prox-mu 0.01
```

Output: `results_root/seed_42/{fedprox,fedavg}/results/*.json` through `results_root/seed_44/`

### Step 2: Compute statistics

```bash
python compute_statistics.py \
  --experiments-root results_root \
  --metric mae
```

Outputs:
- `results_root/statistics/results_long.csv` — long-form data
- `results_root/statistics/pairwise_vs_fedprox.csv` — p-values and effect sizes
- `results_root/statistics/summary_table.tex` — LaTeX table

### Step 3: Visualize and interpret

```python
import pandas as pd
df = pd.read_csv('results_root/statistics/results_long.csv')
df.boxplot(column='mae', by='condition', figsize=(10, 6))
plt.show()
```

## Common pitfalls in metric interpretation

### Pitfall 1: Comparing pooled metrics across conditions

**Wrong**: "Personalized-head achieves κ=0.53 globally, shared-head is κ=0.48."

**Right**: "Personalized-head mean κ=0.53 (across 18 projects); shared-head mean κ=0.48."

### Pitfall 2: Over-interpreting macro-F1 on imbalanced data

**Wrong**: "Macro-F1 improved from 0.50 to 0.52, so 4% better."

**Right**: "Macro-F1 improved from 0.50 to 0.52; F1-8 improved 10 points, F1-1 unchanged."

### Pitfall 3: Assuming low accuracy means failure

On TAWOS, ~60% accuracy is state-of-the-art. Compare federated vs. centralized gap, not absolute accuracy. Report MAE and κ as primary.

### Pitfall 4: Ignoring confidence intervals

**Wrong**: "FedProx κ = 0.51, FedAvg κ = 0.48."

**Right**: Report point estimate ± 95% CI or use bootstrap.

### Pitfall 5: Using accuracy to rank personalized-head conditions

With per-client heads, each has its own optimal head. Pooled accuracy is biased. Always use per-project metrics.

## Advanced optional metrics

### Ordinal-specific metrics

- **Kendall's τ**: rank-based correlation, sensitive to ordinal structure
- **Spearman's ρ**: similar, uses ranks instead of values
- **Mean ordinal distance**: ℙ[|rank(ŷ) - rank(y)|], simpler than MAE but less interpretable

### Communication metrics

- **Compression ratio**: bytes transmitted / full fine-tuning. Typically 0.5–1% for LoRA.
- **Energy cost**: on mobile, transmission uses 100–1000× more energy than computation.
- **Bandwidth utilization**: sustained throughput over federated rounds.

### Population-level fairness

- **Gini index**: inequality in per-project performance (0 = all equal, 1 = one dominates)
- **Standard deviation of per-project MAE**: simple measure of heterogeneity

## Venue-specific metric expectations

**Empirical SE (ESEM, MSR)**: Must include statistical tests. Wilcoxon, Friedman, Nemenyi, and effect sizes required.

**ML venues (NeurIPS)**: Per-project metrics expected. Confidence intervals or std over 3–10 seeds.

**SE conferences (FSE, ICSE)**: Practical impact emphasized. Include confidence intervals. MAE in business units.

**Federated Learning (AISTATS)**: Communication cost and privacy paramount. Report bytes, convergence rounds, threat model.

## Reproducibility

The `fl/metrics.py` module provides reusable functions:

```python
from fl.metrics import evaluate_classification, format_metrics, run_prediction

metrics = evaluate_classification(
    predictions=pred_logits,
    true_labels=test_labels,
    num_classes=5,
    inv_class_map={0: 1, 1: 2, 2: 3, 3: 5, 4: 8}
)
print(f"MAE: {metrics['mae']:.3f}, Kappa: {metrics['cohen_kappa']:.3f}")
```

All statistical tests are in `compute_statistics.py` and can be reused for ad-hoc analysis.


## Handling edge cases in metric computation

### What if a condition doesn't report metrics for all projects?

When a run fails partway through, some projects' metrics may be missing. The pipeline:
1. Loads results_long.csv and checks for missing (seed, project, condition) combinations
2. Warns the user: "FedProx missing metrics for seed_42, Project_X"
3. Drops those rows and continues (Wilcoxon will have n=2 pairs instead of 3)

This is usually acceptable; the power decreases slightly but p-values are still valid. If multiple seeds are missing, consider re-running rather than reporting sparse results.

### What if accuracy is exactly 50% on a 5-class problem?

This suggests the model is predicting uniformly random (one of 5 classes, each 20%, times 2.5 = 50%). This is actually worse than random for a class-imbalanced dataset. Kappa handles this correctly (it is low/negative). Report this as-is; it indicates a failed run.

### What if macro-F1 is reported but per-class F1 is not?

Some implementations simplify and only report macro-F1. For this project, always compute and report per-class F1, because understanding which classes are hard is more informative than a single number.

### What if the test set has only one example of a rare class?

Precision/Recall for that class become unreliable (high variance). Ideally, all test sets have ≥50 examples per class. If not, report a caveat in the results.

### What if two conditions have identical metric values?

This can happen if:
1. Randomness was fixed identically (unlikely unless they used the same seed)
2. Rounding is hiding small differences (check raw precision)
3. Both conditions genuinely converged to the same optimum (rare but possible on simple problems)

Always report raw values with 3–4 decimal places; do not round to 2 decimals before comparison.

## Creating visualizations for the thesis

### Figure 1: Convergence curves

```
Macro-F1 (Validation, per-round average across clients)

0.58 |
0.56 |         ___FedProx
0.54 |        /
0.52 |       /___FedAvg
0.50 |      /
0.48 |     /
0.46 |____/
0.44 |
     +-----+-----+-----+-----+-----+
     0     5     10    15    20    25
              Federated Rounds
```

**Caption**: FedProx with proximal regularization (μ=0.01) converges more slowly than FedAvg (μ=0) but to a higher final macro-F1. Error bars represent ± one standard deviation across 18 projects.

### Figure 2: Per-project MAE heatmap

Create a 18×7 heatmap (18 projects × 7 conditions: Median, TF-IDF+SVM, Local-only, Centralized, FedAvg, FedProx, FedProx+P-head):

```
Project                  Median  TF-IDF  Local  Central  FedAvg  FedProx  P-head
Lsstcorp_Data_mgmt       2.1     1.4     1.2    1.0      1.1     1.0      1.0
Apache_Spark             1.8     1.3     1.1    0.95     1.05    1.0      0.95
Hyperledger_Sawtooth     2.0     1.5     1.3    1.15     1.2     1.15     1.1
OpenStack_Horizon        1.9     1.4     1.2    1.05     1.1     1.05     1.0
...
Mean                     1.85    1.41    1.20   1.07     1.12    1.09     1.05
```

Color scale: red (high MAE) → yellow → green (low MAE). This shows which projects benefit most from federation and personalization.

### Figure 3: Confusion matrices (side-by-side for comparison)

Plot normalized confusion matrices for three conditions (Centralized, FedProx, FedProx+P-head) to show error patterns visually.

**Analysis**: If FedProx and Centralized have nearly identical confusion matrices, federated hasn't lost signal. If FedProx+P-head has more diagonal concentration, personalization is helping.

## Metrics in the thesis structure

### Data chapter

Include:
- Dataset size (42,002 issues, 19 projects)
- Label distribution (table of per-class percentages)
- Text length statistics (p50, p90, p99 token counts)
- Preprocessing impact (how many issues/words removed)

### Methods chapter

Include:
- Definition of each metric (MAE, Kappa, F1)
- Why each metric was chosen (ordinal structure, chance-correction, per-project evaluation)
- Statistical testing approach (Wilcoxon, Friedman, Nemenyi)
- Seed count and reproducibility (3 seeds, RNG fixed)

### Results chapter

Include:
- Main results table (per-project means for each condition)
- Convergence curves (federated rounds)
- Pairwise significance (FedProx vs. each baseline)
- Per-class F1 breakdown (which classes are hard)
- Confusion matrix heatmaps

### Discussion chapter

Interpret:
- Federated-vs-centralized gap (privacy cost in %Δ metrics)
- Personalization benefit (RQ2 finding)
- Communication efficiency (LoRA reduction factor)
- Comparison with related work (Tawosi et al., Deep-SE, GPT2SP)
- Limitations of the evaluation (3 seeds, 18 projects, bert-tiny not reported)

## Metric checklist before publication

- [ ] All metrics are per-project + mean/median (never pooled for per-project models)
- [ ] MAE and Kappa are reported as primary; F1 as supporting
- [ ] Confusion matrices show ordinal structure (near-diagonal concentration)
- [ ] Per-class F1 is included (not macro-F1 alone)
- [ ] Effect sizes (Cliff's δ or Vargha-Delaney Â) are reported alongside p-values
- [ ] Confidence intervals or bootstrap estimates are included
- [ ] Communication cost is reported in bytes and compression ratio
- [ ] Centralized results are reported as upper bound (to contextualize federated gap)
- [ ] Local-only results are reported as lower bound (to show federation benefit)
- [ ] TF-IDF+SVM baseline is included (to contextualize "competitive with what?")
- [ ] Missing data (failed runs, skipped seeds) is disclosed
- [ ] Raw numeric values have ≥3 decimal places (not rounded to 2)
- [ ] Figures are captioned and explained in the text
- [ ] Tables are self-contained (no external references needed to interpret)

