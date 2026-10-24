# Triple-judge calibration vs human annotators on GradingBench

Raw artifacts for the triple-judge calibration against $N=200$ human-graded examples on IMO-GradingBench (Luong et al., 2025, arXiv:2511.01846). These files allow reviewers to reproduce the calibration metrics reported in the paper:

- Aggregate (mean of 3 judges) Pearson $r = 0.820$ vs human Points
- Cohen $\kappa@5 = 0.677$ (substantial by Landis-Koch)
- MAE $= 1.424$ on a 0-7 scale

## Files

| File | Size | Purpose |
|---|---|---|
| `raw_judge_scores_n200.json` | 34 KB | Per-example raw scores from the 3 judges + human Points reference |
| `calibration_report.json` | 1.4 KB | Per-judge and aggregate metrics (r, kappa, MAE, agreement) |

## Schema

### `raw_judge_scores_n200.json`

Per-example records:

```json
[
  {
    "id": "GB-001",
    "problem": "...",
    "reference_solution": "...",
    "model_response": "...",
    "human_points": 5,
    "judge_scores": {
      "gpt-5.4":         5,
      "gemini-3.1-pro":  4,
      "qwen3-235b":      5
    }
  },
  ...
]
```

### `calibration_report.json`

```json
{
  "n_entries_evaluated": 200,
  "threshold_binary": 5.0,
  "per_judge": {
    "gpt-5.4":        {"pearson_r": 0.759, "kappa": 0.588, "mae": 1.57},
    "gemini-3.1-pro": {"pearson_r": 0.835, "kappa": 0.688, "mae": 0.98},
    "qwen3-235b":     {"pearson_r": 0.627, "kappa": 0.486, "mae": 1.96}
  },
  "aggregate_mean3": {"pearson_r": 0.820, "kappa": 0.677, "mae": 1.424}
}
```

## Source

IMO-GradingBench (Luong et al., 2025): 1000 olympiad-style examples each with `(problem, reference_solution, grading_guidelines, model_response, human_points)`. We sampled $N=200$ stratified by `human_points` distribution.

## Reproducing the calibration

1. Load `raw_judge_scores_n200.json`.
2. For each judge, compute Pearson $r$ between `judge_scores[judge]` and `human_points` across the 200 examples.
3. For Cohen $\kappa@5$: binarize both as `score >= 5`, compute Cohen's kappa.
4. For aggregate: `mean3 = mean of all 3 judge_scores per example`, then re-compute against human.
5. Should recover the values in `calibration_report.json`.
