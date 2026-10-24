# Anti-guessing verification-gap pipeline (Numina-HARD-guess, N = 881)

Code for the multi-stage filtering pipeline that builds Numina-HARD-guess and for every number
reported in the paper. The released benchmark and all intermediate outputs (model solutions,
judge scores, rubrics, canary samples, stage labels) are in the top-level `data/` folder of this
repository.

## Reproduce the headline numbers (no API key, no GPU)

```bash
cd code
python3 scripts/00_fetch_released_data.py      # copies ../data into code/data/ (and unzips the model solutions)
python3 scripts/19_review_support.py          # standard library only
```

Expected key lines:

```
combined  pre N=1075 rel= 20.03%   post N= 881 rel= 21.35%   removed=194  delta=+1.32pp
combined unfiltered N=3729 rel=37.92%
combined EASY pool  N=1689 rel=48.91%
combined HARD pool  N=881 rel=21.35%
Gemini-only strict rel on canonical HARD (N=881): 21.75%  (3-judge ensemble = 21.35%)
```

The repository root also ships `reproduce_headline.py`, which recomputes the same table plus
the AIME control (41.74%) directly from the released files.

## Metric

- Per-problem verification gap: answer-only accuracy minus solution-correctness accuracy.
- Relative gap: the gap divided by answer-only accuracy, computed per model first and then
  averaged over the eight held-out models.
- Strict aggregator: a solution is correct iff the mean of the three judge scores
  (GPT-5.4, Gemini-3.1-Pro, Qwen3-235B) is > 6.5 on a 0-7 static rubric AND the answer is correct.
- Negative-gap guard: drop problems whose lenient per-problem gap is negative.

## Pipeline (`scripts/03_pipeline.py`)

1. Inference of the eight target models (greedy decoding).
2. Final-answer verification against the reference (GPT-5.4).
3. Static 0-7 rubric per problem, generated once from problem + reference solution (Gemini-3.1-Pro).
4. Triple-judge grading against the static rubric.
5. Guessability pre-filter (five shortcut strategies, `guessability_rubric_v2.py`).
6. Weak-canary multi-sample stage (GPT-3.5-turbo, Llama-3-8B, Qwen-2.5-7B; K = 8, any hit removes).
7. Geometry exclusion.

Cheap filters run before the expensive grading when building a new pool.

## Scripts

| Script | Purpose |
|---|---|
| `00_fetch_released_data.py` | Copy the released files from `../data` into `code/data/` under the names the scripts use |
| `01_compile_prior_pids.py`, `02_sample.py` | Disjoint sampling of candidate pools from NuminaMath-1.5 |
| `03_pipeline.py` | Main pipeline (API calls; see `.env.example`) |
| `05_compute_gap.py`, `10_compute_gap_phase4.py` | Gap computation and per-aggregator reports |
| `06_reextract_revalidate.py`, `07_identify_bad_data.py` | Answer re-extraction and upstream bad-data exclusion |
| `08_tier2_self_sampling.py`, `09_tier3_strong_grading.py` | Three-tier design used only in the cost ablation (scripts 08-10 document that ablation; they read intermediate files of the first run that are not part of this release) |
| `13_combined_883.py`, `14_supplementary_v4.py` | Combined pool and supplementary statistics (before decontamination) |
| `15_gpt_guessability_dual.py`, `16_dual_vs_single.py` | Second guessability judge and dual-judge comparison |
| `17_contamination_v4.py`, `18_decontaminate.py` | Near-duplicate audit and decontamination (N = 883 to 881) |
| `19_review_support.py` | Guard effect, within-source baseline, single-judge regrade |
| `pilot_run.py`, `static_rubric.py`, `guessability_rubric*.py` | Prompt modules used by the pipeline |

## Regenerate from scratch (requires API keys and budget)

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in the keys
python3 scripts/03_pipeline.py --input stage_a_1000.json --run stage_a --workers 300
```

The two runs cost about USD 300 (1,000 candidates) and USD 806 (3,000 candidates).

## License

Code: Apache-2.0 (`LICENSE`). Data: CC-BY-4.0 (`../LICENSE-DATASET.txt`).
