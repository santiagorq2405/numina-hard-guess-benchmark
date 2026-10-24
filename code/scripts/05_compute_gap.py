"""
Compute gaps + per-model + Fleiss kappa from pipeline outputs.

Metric (paper canonical):
- per-group gap = avg(answer_only_acc) - avg(solution_acc)
  where solution correct = mean of 3 judges scores > 6.5
- aggregated across N problems x 8 models, per-problem then averaged

Reports:
- gap_baseline_pre_filter (all problems)
- gap_post_filter (filtered subset: NOT EASY by rubric_v2 AND not canary-hit AND not geometry)
- per-model breakdown
- Fleiss kappa across 3 judges (binary at threshold 6.5)
- coverage stats

Usage:
  python 05_compute_gap.py --run smoke --report results/smoke_10_report.md
  python 05_compute_gap.py --run stage_a --report results/stage_a_report.md
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent.parent / "data"
RES_DIR = Path(__file__).parent.parent / "results"

JUDGES = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
SOL_THRESHOLD = 6.5  # mean of 3 judges > 6.5 = solution correct

TARGETS = ["claude-haiku-4.5", "claude-sonnet-4.5", "command-a", "llama-3.1-8b",
           "llama-3.3-70b", "mistral-small-3.2-24b", "nova-pro-v1", "qwen2.5-72b"]


def load_excluded_pids(run):
    """Load bad-data PIDs to exclude from analysis."""
    path = Path(__file__).parent.parent / "data" / f"{run}_excluded_pids.json"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return set(d.get("excluded_pids", []))


def load_json(path):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def fleiss_kappa(matrix, n_categories=2):
    """Fleiss kappa over a list of [counts in each category] per item."""
    import math
    N = len(matrix)
    if N == 0:
        return None
    n_raters = sum(matrix[0])
    if n_raters == 0:
        return None
    # Item agreement
    P_i = []
    for row in matrix:
        s = sum(c * (c - 1) for c in row)
        P_i.append(s / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / N
    # Category proportions
    p_j = [sum(row[j] for row in matrix) / (N * n_raters) for j in range(n_categories)]
    P_e = sum(p**2 for p in p_j)
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    # Load pipeline outputs
    inference = load_json(OUT_DIR / f"{args.run}_inference.json")
    verification = load_json(OUT_DIR / f"{args.run}_verification.json")
    rubric_v2 = load_json(OUT_DIR / f"{args.run}_rubric_v2.json")
    canary = load_json(OUT_DIR / f"{args.run}_canary.json")
    geometry = load_json(OUT_DIR / f"{args.run}_geometry.json")
    grading = {j: load_json(OUT_DIR / f"{args.run}_grading_{j}.json") for j in JUDGES}

    # Determine problem set + model set
    pids_all = sorted({entry["pid"] for entry in inference.values()})
    models = sorted({entry["model"] for entry in inference.values()})

    # Exclude bad-data PIDs (proof/notfound/empty/short)
    excluded_pids = load_excluded_pids(args.run)
    pids = [p for p in pids_all if p not in excluded_pids]
    print(f"PIDs total: {len(pids_all)}, excluded (bad data): {len(excluded_pids)}, kept: {len(pids)}")
    print(f"Models: {len(models)}")
    print(f"Inference: {len(inference)} entries")
    print(f"Verification: {len(verification)} entries")
    print(f"Grading per judge: {[(j, len(g)) for j, g in grading.items()]}")

    # Per-(pid, model): answer_correct + solution_correct (mean 3 judges > 6.5)
    # Computes BOTH aggregators:
    #   - lenient (paper headline): sol_correct = mean(3 judges) > 6.5
    #   - strict:     sol_correct = answer_correct AND mean(3 judges) > 6.5
    # Strict guarantees gap >= 0 by construction.
    pmm_data = {}
    for pid in pids:
        for m in models:
            key = f"{pid}__{m}"
            v = verification.get(key, {})
            ac = v.get("answer_correct")
            ac_int = int(ac) if ac is True else (0 if ac is False else None)
            scores = []
            for j in JUDGES:
                e = grading.get(j, {}).get(key)
                if e and e.get("score") is not None:
                    scores.append(e["score"])
            if len(scores) >= 2:
                mean_score = sum(scores) / len(scores)
                sc_lenient = 1 if mean_score > SOL_THRESHOLD else 0
                sc_strict = 1 if (mean_score > SOL_THRESHOLD and ac_int == 1) else 0
            else:
                mean_score = None
                sc_lenient = None
                sc_strict = None
            pmm_data[(pid, m)] = {
                "answer_correct": ac_int,
                "solution_correct": sc_lenient,  # default: lenient
                "solution_correct_strict": sc_strict,
                "mean_score": mean_score,
                "n_judges": len(scores),
                "scores": scores,
            }

    # Per-problem average (across models, only non-None entries)
    def per_problem_gap(pid_subset, strict=False):
        sol_key = "solution_correct_strict" if strict else "solution_correct"
        gaps = []
        per_pid_detail = []
        for pid in pid_subset:
            ans_vals = [pmm_data[(pid, m)]["answer_correct"] for m in models if pmm_data[(pid, m)]["answer_correct"] is not None]
            sol_vals = [pmm_data[(pid, m)][sol_key] for m in models if pmm_data[(pid, m)][sol_key] is not None]
            if not ans_vals or not sol_vals:
                continue
            ans_acc = sum(ans_vals) / len(ans_vals)
            sol_acc = sum(sol_vals) / len(sol_vals)
            gap = (ans_acc - sol_acc) * 100
            gaps.append(gap)
            per_pid_detail.append({"pid": pid, "ans_acc": ans_acc, "sol_acc": sol_acc, "gap": gap})
        avg = sum(gaps) / len(gaps) if gaps else None
        return avg, gaps, per_pid_detail

    # Per-model gap
    def per_model_gap(pid_subset):
        out = {}
        for m in models:
            ans_vals = [pmm_data[(pid, m)]["answer_correct"] for pid in pid_subset if pmm_data[(pid, m)]["answer_correct"] is not None]
            sol_vals = [pmm_data[(pid, m)]["solution_correct"] for pid in pid_subset if pmm_data[(pid, m)]["solution_correct"] is not None]
            if not ans_vals or not sol_vals:
                out[m] = None
                continue
            ans = sum(ans_vals) / len(ans_vals) * 100
            sol = sum(sol_vals) / len(sol_vals) * 100
            out[m] = {"ans_acc": ans, "sol_acc": sol, "gap": ans - sol, "n_ans": len(ans_vals), "n_sol": len(sol_vals)}
        return out

    # Filter logic (pipeline specification)
    # HARD candidates = NOT EASY by rubric_v2 AND not canary-hit (any sample correct in any canary) AND NOT geometry
    canary_hit = defaultdict(bool)
    for key, entry in canary.items():
        canary_hit[entry["pid"]] = canary_hit[entry["pid"]] or entry.get("any_correct", False)

    is_geom = {pid: geometry.get(pid, {}).get("is_geometry") for pid in pids}
    is_easy = {pid: rubric_v2.get(pid, {}).get("classification") == "EASY_TO_GUESS" for pid in pids}

    pids_baseline = pids  # all
    pids_post_filter = [pid for pid in pids
                        if not is_easy.get(pid, False)  # not EASY
                        and not canary_hit.get(pid, False)  # not hacked
                        and is_geom.get(pid) is not True]  # not geometry

    gap_baseline, gaps_b, _ = per_problem_gap(pids_baseline, strict=False)
    gap_post, gaps_p, _ = per_problem_gap(pids_post_filter, strict=False)
    gap_baseline_strict, _, _ = per_problem_gap(pids_baseline, strict=True)
    gap_post_strict, _, _ = per_problem_gap(pids_post_filter, strict=True)

    # Negative-gap residual guard :
    # drop PIDs where per-problem gap < 0 from HARD pool.
    # This eliminates "rigorous-but-wrong-answer artifacts" at problem level.
    _, _, hard_per_pid = per_problem_gap(pids_post_filter, strict=False)
    pids_post_guard = [d["pid"] for d in hard_per_pid if d["gap"] >= 0]
    gap_post_guarded, _, _ = per_problem_gap(pids_post_guard, strict=False)
    gap_post_guarded_strict, _, _ = per_problem_gap(pids_post_guard, strict=True)

    pm_baseline = per_model_gap(pids_baseline)
    pm_post = per_model_gap(pids_post_filter)
    pm_post_guarded = per_model_gap(pids_post_guard)

    # Fleiss kappa on binary (solution_correct yes/no by each judge >6.5 individually)
    matrix = []
    for pid in pids:
        for m in models:
            scores = []
            for j in JUDGES:
                e = grading.get(j, {}).get(f"{pid}__{m}")
                if e and e.get("score") is not None:
                    scores.append(e["score"])
            if len(scores) == 3:
                # Each judge says yes/no at threshold 6.5
                yes = sum(1 for s in scores if s > SOL_THRESHOLD)
                no = 3 - yes
                matrix.append([yes, no])
    fk = fleiss_kappa(matrix, n_categories=2) if matrix else None

    # Coverage
    cov = {
        "inference_total": len(pids) * len(models),
        "inference_done": len(inference),
        "verification_done": sum(1 for e in verification.values() if e.get("answer_correct") is not None),
        "grading_done_per_judge": {j: sum(1 for e in g.values() if e.get("score") is not None) for j, g in grading.items()},
        "rubric_v2_done": sum(1 for e in rubric_v2.values() if e.get("classification")),
        "canary_done": len(canary),
        "geometry_done": sum(1 for e in geometry.values() if e.get("is_geometry") is not None),
    }

    # Filter step counts
    filter_breakdown = {
        "all_pids": len(pids_baseline),
        "easy_by_rubric_v2": sum(1 for pid in pids if is_easy.get(pid)),
        "canary_hit": sum(1 for pid in pids if canary_hit.get(pid)),
        "geometry": sum(1 for pid in pids if is_geom.get(pid) is True),
        "passed_all_filters": len(pids_post_filter),
    }

    # Write report
    # AIME baseline sobre el set de modelos correspondiente al paper:
    # - 8 comparable-tier held-out (lo que estamos usando): +17.27 pp
    # - 5 calibration: +8.38 pp
    # Reportamos el de 8 held-out porque es lo que evaluamos
    AIME_BASELINE = 17.27

    decision = "INSUFFICIENT_DATA"
    if gap_post is not None:
        if gap_post <= AIME_BASELINE:
            decision = "STOP_DONE"
        elif gap_post > AIME_BASELINE + 5:
            decision = "CONTINUE_PHASE_4_TARGET_K"
        else:
            decision = "BORDERLINE_LEAN_STOP"

    md = f"""# Stage A pipeline report — run={args.run}

## Headlines (two aggregators reported)

### Aggregator 1: LENIENT (paper main text headline)
sol_correct = mean(3 judges) > 6.5  -- can produce negative gaps if judges score "rigorous attempt" with wrong answer

| Metric | Value |
|---|---|
| gap_baseline_pre_filter | {gap_baseline:+.2f} pp ({len(pids_baseline)} problems) |
| gap_post_filter | {gap_post:+.2f} pp ({len(pids_post_filter)} problems) |
| Shrinkage (lenient) | {(gap_baseline - gap_post):+.2f} pp |

### Aggregator 2: STRICT
sol_correct = answer_correct AND mean(3 judges) > 6.5  -- guarantees gap >= 0 by construction

| Metric | Value |
|---|---|
| gap_baseline_pre_filter (strict) | {gap_baseline_strict:+.2f} pp |
| gap_post_filter (strict) | {gap_post_strict:+.2f} pp |
| Shrinkage (strict) | {(gap_baseline_strict - gap_post_strict):+.2f} pp |

### Aggregator 3: NEGATIVE-GAP RESIDUAL GUARD
HARD pool with PIDs of negative gap REMOVED. Replicates paper's full pipeline guard.

| Metric | Value |
|---|---|
| gap_post_filter (lenient + guard) | {gap_post_guarded:+.2f} pp ({len(pids_post_guard)} problems) |
| gap_post_filter (strict + guard) | {gap_post_guarded_strict:+.2f} pp |
| HARD pool size after guard | {len(pids_post_guard)} (vs {len(pids_post_filter)} pre-guard) |
| PIDs removed by negative-gap guard | {len(pids_post_filter) - len(pids_post_guard)} |

### Common reference

| Metric | Value |
|---|---|
| AIME baseline (8 held-out comparable, paper line 148) | +{AIME_BASELINE:.2f} pp |
| Decision (lenient aggregator) | **{decision}** |
| Fleiss kappa (3 judges, binary>6.5) | {f'{fk:.3f}' if fk is not None else 'N/A'} |
| Bad data excluded | {len(excluded_pids)} PIDs (proof/notfound/empty/short) |

## Filter breakdown

| Stage | N problems |
|---|---|
| All (pre-filter) | {filter_breakdown['all_pids']} |
| EASY by rubric_v2 (REMOVED) | {filter_breakdown['easy_by_rubric_v2']} |
| Canary-hit any (REMOVED from HARD) | {filter_breakdown['canary_hit']} |
| Geometry (REMOVED from HARD) | {filter_breakdown['geometry']} |
| **Final HARD pool** | **{filter_breakdown['passed_all_filters']}** |

## Coverage

| Stage | Done / Total |
|---|---|
| Inference | {cov['inference_done']} / {cov['inference_total']} |
| Verification | {cov['verification_done']} / {cov['inference_total']} |
| Grading gpt-5.4 | {cov['grading_done_per_judge']['gpt-5.4']} / {cov['inference_total']} |
| Grading gemini-3.1-pro | {cov['grading_done_per_judge']['gemini-3.1-pro']} / {cov['inference_total']} |
| Grading qwen3-235b | {cov['grading_done_per_judge']['qwen3-235b']} / {cov['inference_total']} |
| Rubric v2 | {cov['rubric_v2_done']} / {len(pids)} |
| Canary | {cov['canary_done']} / {len(pids) * 3} |
| Geometry | {cov['geometry_done']} / {len(pids)} |

## Per-model gap (baseline pool)

| Model | ans_acc% | sol_acc% | gap_pp | n |
|---|---|---|---|---|
"""
    for m in TARGETS:
        v = pm_baseline.get(m)
        if v is None:
            md += f"| {m} | N/A | N/A | N/A | 0 |\n"
        else:
            md += f"| {m} | {v['ans_acc']:.1f} | {v['sol_acc']:.1f} | {v['gap']:+.2f} | {v['n_ans']} |\n"

    md += "\n## Per-model gap (post-filter HARD pool, no guard)\n\n"
    md += "| Model | ans_acc% | sol_acc% | gap_pp | n |\n|---|---|---|---|---|\n"
    for m in TARGETS:
        v = pm_post.get(m)
        if v is None:
            md += f"| {m} | N/A | N/A | N/A | 0 |\n"
        else:
            md += f"| {m} | {v['ans_acc']:.1f} | {v['sol_acc']:.1f} | {v['gap']:+.2f} | {v['n_ans']} |\n"

    md += f"\n## Per-model gap (HARD pool + negative-gap guard, replicates paper)\n\n"
    md += f"HARD pool after dropping PIDs with negative gap: N={len(pids_post_guard)}\n\n"
    md += "| Model | ans_acc% | sol_acc% | gap_pp | n |\n|---|---|---|---|---|\n"
    for m in TARGETS:
        v = pm_post_guarded.get(m)
        if v is None:
            md += f"| {m} | N/A | N/A | N/A | 0 |\n"
        else:
            md += f"| {m} | {v['ans_acc']:.1f} | {v['sol_acc']:.1f} | {v['gap']:+.2f} | {v['n_ans']} |\n"

    md += f"\n## Decision criteria\n\n"
    md += f"- STOP if gap_post_filter <= AIME +8.38 pp\n"
    md += f"- CONTINUE Phase 4 if gap_post_filter > AIME + 5 pp (clearly above)\n"
    md += f"- BORDERLINE between -> lean STOP\n\n"
    md += f"Current: {decision}\n"

    res_path = ROOT / args.report
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report saved: {res_path}")
    print(f"\nHEADLINE: gap_baseline={gap_baseline}, gap_post={gap_post}, decision={decision}")


if __name__ == "__main__":
    main()
