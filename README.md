# Numina-HARD-guess: an anti-guessing olympiad-math benchmark (N = 881)

Code and data for the paper **"Answer-Only Accuracy Measures the Wrong Construct: An Anti-Guessing
Verification-Gap Protocol for Trustworthy Olympiad-Math Evaluation"**, by Santiago Rivadeneira
Quintero, Denys Pushkin and Emmanuel Abbe (EPFL), poster at the NeurIPS 2026 workshop TAE
(Trust-AI-Eval): Can We Trust AI Evaluation?
Paper and reviews: https://openreview.net/forum?id=gOQk5WDD6f

Answer-only accuracy can credit a model for a correct final answer that comes without a valid
derivation. The **verification gap** is answer-only accuracy minus graded solution correctness.
The filtering pipeline keeps the olympiad problems for which the final answer is hard to reach
without a rigorous solution, and this repository releases the resulting benchmark, the problems
the filter removed, and every intermediate output, so that each number in the paper can be
recomputed without API calls.

## Headline numbers (eight held-out comparable-tier models)

| Pool | N | Relative verification gap |
|---|---|---|
| **Numina-HARD-guess (released)** | **881** | **21.35%** |
| Stage A run (1,000 candidates) | 200 | 20.79% |
| Stage B run (3,000 candidates) | 681 | 21.00% |
| Easy-to-guess pool (removed by the filter) | 1,689 | 48.91% |
| Unfiltered evaluable pool (same source) | 3,729 | 37.92% |
| AIME 1983-2024 control (same models and protocol) | 210 | 41.74% |

Relative gap = (answer-only accuracy - solution accuracy) / answer-only accuracy, computed per
model and then averaged over models. A solution counts as correct iff the mean of three judge
scores (GPT-5.4, Gemini-3.1-Pro, Qwen3-235B) is > 6.5 on a 0-7 static rubric **and** the final
answer is correct. The judge ensemble agrees with human graders on 200 IMO-GradingBench examples
(Pearson r = 0.820, Cohen kappa@5 = 0.677, kappa@6.5 = 0.793).

## Reproduce

Standard library only, no API key, no GPU:

```bash
python3 reproduce_headline.py
```

Expected output:

```
stage_a: HARD N=200 rel gap=20.79%
stage_b: HARD N=681 rel gap=21.00%
combined hard: N=881 rel gap=21.35%
combined  all: N=3729 rel gap=37.92%
combined easy: N=1689 rel gap=48.91%
AIME control: N=210 rel gap=41.74%
```

To run the pipeline scripts themselves (guard effect, single-judge regrade, and the rest), see
[`code/README.md`](code/README.md).

## Pipeline (applied to NuminaMath-1.5 `olympiads` / `olympiads_ref`)

1. Guessability pre-filter (Gemini-3.1-Pro, five shortcut strategies); problems labelled
   `EASY_TO_GUESS` are removed.
2. Weak-canary multi-sample stage: GPT-3.5-turbo, Llama-3-8B-instruct and Qwen-2.5-7B-instruct,
   K = 8 direct-answer samples each; any correct sample removes the problem.
3. Geometry exclusion (Gemini-3.1-Pro classifier).
4. Triple-judge static-rubric grading of the eight target models' solutions.
5. Negative-gap residual guard (drops problems whose lenient per-problem gap is negative).
6. Near-duplicate decontamination against AIME, MATH-500 and OlympicArena (two problems removed).

Cheap filters run first and the expensive grading last. The two runs cost about USD 300 and
USD 806 (about USD 1.25 per released problem).

Target models: Claude-Haiku-4.5, Claude-Sonnet-4.5, Command-A, Llama-3.1-8B, Llama-3.3-70B,
Mistral-Small-3.2-24B, Nova-Pro-v1, Qwen2.5-72B (open-weight models from 8B to 111B parameters
and commercial models).

## Files

| Path | Content |
|---|---|
| `data/numina_hard_guess_881.json` | The released benchmark: problem, reference answer and solution, metadata |
| `data/easy_to_guess_pool.json` | Problems labelled easy to guess and removed (1,689) |
| `data/pipeline_labels.json` | One row per evaluable candidate (3,729): guessability label and strategy, canary hit, geometry flag, pre-guard and final membership |
| `data/stage_{a,b}/candidate_problems_*.json` | Sampled candidate pools (1,000 and 3,000, disjoint, SHA-256 deduplicated) |
| `data/stage_b/removed_near_duplicates.json` | The two problems removed by the near-duplicate audit |
| `data/stage_{a,b}/excluded_bad_data_pids.json` | Candidates excluded as upstream bad data (proof-only, not-found, empty, under-length) |
| `data/stage_{a,b}/guessability_labels_gemini-3.1-pro.json` | Guessability classifier outputs (the filter used) |
| `data/stage_{a,b}/guessability_labels_gpt-5.4_second_judge.json` | Second guessability judge, for the dual-judge robustness check |
| `data/stage_{a,b}/weak_canary_samples.json` | All weak-canary samples and hits |
| `data/stage_{a,b}/geometry_labels.json` | Geometry classifier outputs |
| `data/stage_{a,b}/static_rubrics.json` | 0-7 rubrics generated once per problem from problem + reference solution |
| `data/stage_{a,b}/model_solutions.json.gz` | Solutions of the eight target models (greedy decoding), gzip-compressed (`gunzip -k` to unpack) |
| `data/stage_{a,b}/answer_verification.json` | Final-answer verification against the reference (GPT-5.4) |
| `data/stage_{a,b}/grading_{judge}.json` | Per-judge rubric scores with the judges' raw outputs |
| `data/aime_control/` | AIME 1983-2024 control (210 problems, 5 per year) with the same eight models and protocol |
| `data/judge_calibration_gradingbench/` | Triple-judge calibration against 200 human-graded IMO-GradingBench examples |
| `data/contamination_audit/` | Near-duplicate audit of the pre-decontamination HARD pool: full per-problem match table (best match in AIME, MATH-500 and OlympicArena-Math) with the manual review of the three flags |
| `code/` | Pipeline code (Apache-2.0) and the scripts behind every number in the paper |
| `reproduce_headline.py` | Recomputes the table above from these files |

Problem ids (`pid`) are unique within a run; use (`stage`, `pid`) as the key.

## Scope and intended use

Non-geometry, English, text-only olympiad problems. Intended for measuring how much answer-only
accuracy over-credits a model relative to graded solutions, and for reporting answer-only and
graded accuracy side by side.

## License

Data: CC-BY-4.0 (`LICENSE-DATASET.txt`). Code: Apache-2.0 (`code/LICENSE`). The problems come
from NuminaMath-1.5 (Apache-2.0); please also acknowledge that corpus.

## Citation

```bibtex
@inproceedings{rivadeneira2026answeronly,
  title     = {Answer-Only Accuracy Measures the Wrong Construct: An Anti-Guessing Verification-Gap Protocol for Trustworthy Olympiad-Math Evaluation},
  author    = {Rivadeneira Quintero, Santiago and Pushkin, Denys and Abbe, Emmanuel},
  booktitle = {NeurIPS 2026 Workshop on Trust-AI-Eval (TAE): Can We Trust AI Evaluation?},
  year      = {2026},
  url       = {https://openreview.net/forum?id=gOQk5WDD6f}
}
```
