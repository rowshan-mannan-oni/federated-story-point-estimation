# Evaluation Metrics

Results are evaluated per project because project-specific story point conventions make pooled metrics potentially misleading.

## Primary metrics

- **MAE:** absolute error after mapping predictions back to story point values.
- **Quadratic-weighted Cohen's kappa:** chance-corrected agreement that penalizes distant ordinal errors.

## Supporting metrics

- Macro-F1
- Accuracy
- Weighted F1
- Per-class F1
- Confusion matrix
- Communication cost

Errors concentrated on neighboring story point classes indicate better ordinal behavior than accuracy alone may show.
