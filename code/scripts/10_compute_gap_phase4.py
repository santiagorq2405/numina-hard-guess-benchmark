"""
Three-tier ablation: recompute the gap on the problems that survive Tier 2 + Tier 3.

Reuses the existing outputs of the eight held-out models (no new inference) and reports the
gap of the simple pipeline next to the gap after the two extra tiers.

Usage:
  python 10_compute_gap_phase4.py --report results/stage_a_phase4_report.md
"""

import argparse
import json
from pathlib import Path
from collections import Counter

DATA = Path(__file__).parent.parent / "data"
RES = Path(__file__).parent.parent / "results"
ROOT = Path(__file__).parent.parent

JUDGES = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
SOL_THRESHOLD = 6.5
AIME_BASELINE = 17.27

TARGETS = ["claude-haiku-4.5", "claude-sonnet-4.5", "command-a", "llama-3.1-8b",
           "llama-3.3-70b", "mistral-small-3.2-24b", "nova-pro-v1", "qwen2.5-72b"]


def load_json(path):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_gap(pids_subset, ver, grading_per_judge, models):
    """Compute aggregate gap (lenient) over a subset of pids."""
    pmm = {}
    for pid in pids_subset:
        for m in models:
            key = f"{pid}__{m}"
            v = ver.get(key, {})
            ac = v.get("answer_correct")
            scores = []
            for j in JUDGES:
                e = grading_per_judge.get(j, {}).get(key)
                if e and e.get("score") is not None:
                    scores.append(e["score"])
            if len(scores) >= 2 and ac is not None:
                pmm[(pid, m)] = {
                    "ans": int(ac) if ac is True else 0,
                    "sol": 1 if (sum(scores)/len(scores)) > SOL_THRESHOLD else 0,
                }

    gaps = []
    for pid in pids_subset:
        ans = [pmm[(pid, m)]["ans"] for m in models if (pid, m) in pmm]
        sol = [pmm[(pid, m)]["sol"] for m in models if (pid, m) in pmm]
        if not ans or not sol:
            continue
        gaps.append((sum(ans)/len(ans) - sum(sol)/len(sol)) * 100)
    avg = sum(gaps)/len(gaps) if gaps else None
    return avg, gaps


def per_model_gap(pids_subset, ver, grading_per_judge, models):
    out = {}
    for m in models:
        ans, sol = [], []
        for pid in pids_subset:
            key = f"{pid}__{m}"
            v = ver.get(key, {})
            ac = v.get("answer_correct")
            scores = []
            for j in JUDGES:
                e = grading_per_judge.get(j, {}).get(key)
                if e and e.get("score") is not None:
                    scores.append(e["score"])
            if len(scores) >= 2 and ac is not None:
                ans.append(int(ac) if ac is True else 0)
                sol.append(1 if (sum(scores)/len(scores)) > SOL_THRESHOLD else 0)
        if ans and sol:
            ans_p = 100*sum(ans)/len(ans)
            sol_p = 100*sum(sol)/len(sol)
            out[m] = {"ans": ans_p, "sol": sol_p, "gap": ans_p - sol_p, "n": len(ans)}
        else:
            out[m] = None
    return out


def apply_guard(pids_subset, ver, grading_per_judge, models):
    """Drop pids with per-pid gap < 0."""
    keep = []
    for pid in pids_subset:
        ans = []
        sol = []
        for m in models:
            key = f"{pid}__{m}"
            v = ver.get(key, {})
            ac = v.get("answer_correct")
            scores = []
            for j in JUDGES:
                e = grading_per_judge.get(j, {}).get(key)
                if e and e.get("score") is not None:
                    scores.append(e["score"])
            if len(scores) >= 2 and ac is not None:
                ans.append(int(ac) if ac is True else 0)
                sol.append(1 if (sum(scores)/len(scores)) > SOL_THRESHOLD else 0)
        if not ans or not sol:
            continue
        gap = sum(ans)/len(ans) - sum(sol)/len(sol)
        if gap >= 0:
            keep.append(pid)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="results/stage_a_phase4_report.md")
    args = ap.parse_args()

    # Load all data
    inf = load_json(DATA / "stage_a_inference.json")
    ver = load_json(DATA / "stage_a_verification.json")
    grading_per_judge = {j: load_json(DATA / f"stage_a_grading_{j}.json") for j in JUDGES}

    # Pools:
    # 1. v4 HARD pool (after rubric+canary+geometry+guard) = 199 PIDs
    hard_200 = load_json(DATA / "stage_a_hard_200_pids.json")
    if isinstance(hard_200, dict):
        v4_pids = list(hard_200.keys())
    else:
        v4_pids = [p["pid"] for p in hard_200]
    print(f"v4 HARD pool: {len(v4_pids)}")

    # 2. Tier 2 survivors
    t2_surv = load_json(DATA / "stage_a_tier2_survivors.json")
    t2_pids = t2_surv.get("survivors", [])
    print(f"Tier 2 survivors: {len(t2_pids)}")

    # 3. Tier 3 survivors
    t3_surv = load_json(DATA / "stage_a_tier3_survivors.json")
    t3_pids = t3_surv.get("survivors", [])
    print(f"Tier 3 survivors: {len(t3_pids)}")

    # Apply guard to each pool
    v4_guard = apply_guard(v4_pids, ver, grading_per_judge, TARGETS)
    t2_guard = apply_guard(t2_pids, ver, grading_per_judge, TARGETS) if t2_pids else []
    t3_guard = apply_guard(t3_pids, ver, grading_per_judge, TARGETS) if t3_pids else []

    # Compute gaps
    gap_v4, _ = compute_gap(v4_pids, ver, grading_per_judge, TARGETS)
    gap_v4_guard, _ = compute_gap(v4_guard, ver, grading_per_judge, TARGETS)
    gap_t2, _ = compute_gap(t2_pids, ver, grading_per_judge, TARGETS) if t2_pids else (None, None)
    gap_t2_guard, _ = compute_gap(t2_guard, ver, grading_per_judge, TARGETS) if t2_guard else (None, None)
    gap_t3, _ = compute_gap(t3_pids, ver, grading_per_judge, TARGETS) if t3_pids else (None, None)
    gap_t3_guard, _ = compute_gap(t3_guard, ver, grading_per_judge, TARGETS) if t3_guard else (None, None)

    # Per-model
    pm_t3_guard = per_model_gap(t3_guard, ver, grading_per_judge, TARGETS) if t3_guard else {}

    # Build report
    md = f"""# Stage A + Phase 4 Pipeline Report

**Pipeline reorder narrative**: cheap filters first, expensive Tier 2/3 last.

## Headlines (lenient + guard, paper canonical)

| Pool | N | gap | vs AIME (+{AIME_BASELINE:.2f}) |
|---|---|---|---|
| v4 HARD (rubric+canary+geometry+guard) | {len(v4_guard)} | {gap_v4_guard:+.2f} pp | {gap_v4_guard - AIME_BASELINE:+.2f} pp |
| Tier 2 survivors (+ guard) | {len(t2_guard) if t2_guard else 'N/A'} | {gap_t2_guard:+.2f} pp | {gap_t2_guard - AIME_BASELINE:+.2f} pp |
| **Tier 3 survivors (+ guard) [final]** | **{len(t3_guard) if t3_guard else 'N/A'}** | **{gap_t3_guard:+.2f} pp** | **{gap_t3_guard - AIME_BASELINE:+.2f} pp** |

## Comparison v4 vs Phase 4

| Metric | v4 (without Phase 4) | Phase 4 (with Tier 2 + Tier 3) |
|---|---|---|
| HARD pool size | {len(v4_guard)} | {len(t3_guard) if t3_guard else 'N/A'} |
| gap_post (lenient + guard) | {gap_v4_guard:+.2f} pp | {gap_t3_guard:+.2f} pp |
| Diff vs AIME | {gap_v4_guard - AIME_BASELINE:+.2f} pp | {gap_t3_guard - AIME_BASELINE:+.2f} pp |
| Tier 2/3 contribution | - | shrinks pool by {len(v4_guard) - (len(t3_guard) if t3_guard else 0)} PIDs ({100*(len(v4_guard) - (len(t3_guard) if t3_guard else 0))/max(len(v4_guard),1):.0f}%) |

## Filter funnel

| Stage | N | Removed |
|---|---|---|
| HARD pool v4 (input to Phase 4) | {len(v4_pids)} | - |
| After Tier 2 (5 calibration K=7/15/30) | {len(t2_pids) if t2_pids else 'N/A'} | {len(v4_pids) - (len(t2_pids) if t2_pids else 0)} |
| After Tier 3 (claude-sonnet K=3 + triple-judge) | {len(t3_pids) if t3_pids else 'N/A'} | {(len(t2_pids) if t2_pids else 0) - (len(t3_pids) if t3_pids else 0)} |
| After negative-gap guard | {len(t3_guard) if t3_guard else 'N/A'} | {(len(t3_pids) if t3_pids else 0) - (len(t3_guard) if t3_guard else 0)} |

## Per-model gap (Phase 4 final HARD pool)

| Model | ans% | sol% | gap | n |
|---|---|---|---|---|
"""
    for m in TARGETS:
        v = pm_t3_guard.get(m)
        if v is None:
            md += f"| {m} | N/A | N/A | N/A | 0 |\n"
        else:
            md += f"| {m} | {v['ans']:.1f} | {v['sol']:.1f} | {v['gap']:+.2f} | {v['n']} |\n"

    md += f"""

## Comparison with paper original

| Metric | Paper Numina-HARD-guess | Our Phase 4 final |
|---|---|---|
| HARD pool N | 134 | {len(t3_guard) if t3_guard else 'N/A'} |
| gap_post (lenient + guard) | +2.61 pp | {gap_t3_guard:+.2f} pp |
| Diff vs AIME (+17.27) | -14.66 pp BELOW | {gap_t3_guard - AIME_BASELINE:+.2f} pp |
| Pipeline | Tier 1 + Tier 2 + Tier 3 over full pool | rubric+canary+geometry FIRST, then Tier 2 + Tier 3 over survivors |
| Cost | ~$1,300 | ~$430 (66% reduction) |

## Notes

- Pipeline reorder = cheap filters first (rubric_v2 + canary + geometry), expensive Tier 2/3 only on survivors
- Mathematically equivalent in HARD pool quality (Tier 2/3 only removes, never recovers)
- 5x cheaper Tier 2/3 application (over 199 survivors vs over 1000 original)
- HARD pool more restrictive than paper (subset of paper's HARD)
"""

    out_path = ROOT / args.report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport saved: {out_path}")
    print(f"\nHEADLINE Phase 4: gap_post (lenient+guard) = {gap_t3_guard:+.2f} pp on {len(t3_guard) if t3_guard else 0} HARD pool")


if __name__ == "__main__":
    main()
