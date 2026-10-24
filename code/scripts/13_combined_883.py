"""
Combined 883 HARD analysis: pool stage_a (200) + stage_b (683) and compute the
definitive relative gap on all 883, plus the per-stage replication numbers.

Replicates the exact HARD-pool selection + strict aggregator + negative-gap guard
from 05_compute_gap.py so the combined number is methodologically identical.

Relative gap = (grand-mean ans_acc - grand-mean sol_acc_strict) / grand-mean ans_acc.
Strict: sol_correct = answer_correct AND mean(3 judges) > 6.5.
Guard: drop pids whose per-problem LENIENT gap < 0.

Output: per-stage + combined table + bootstrap 95% CIs + per-model breakdown.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
SOL_THRESHOLD = 6.5
JUDGES = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
MODELS = [
    "claude-haiku-4.5", "claude-sonnet-4.5", "command-a", "llama-3.1-8b",
    "llama-3.3-70b", "mistral-small-3.2-24b", "nova-pro-v1", "qwen2.5-72b",
]


def load(stage: str, name: str) -> dict:
    return json.load(open(DATA / f"{stage}_{name}.json", encoding="utf-8"))


def load_excluded(stage: str) -> set:
    """Bad-data PIDs (proof/notfound/empty/short) to exclude, matching 05_compute_gap.py."""
    path = DATA / f"{stage}_excluded_pids.json"
    if not path.exists():
        return set()
    d = json.load(open(path, encoding="utf-8"))
    return set(d.get("excluded_pids", []))


def build_stage(stage: str):
    """Return (pmm, hard_pids_post_guard) for a stage using the canonical aggregator."""
    problems = load(stage, "3000" if stage == "stage_b" else "1000") if False else None
    # problem id list: derive from verification keys
    verif = load(stage, "verification")
    grad = {j: load(stage, f"grading_{j}") for j in JUDGES}
    rubric_v2 = load(stage, "rubric_v2")
    canary = load(stage, "canary")
    geometry = load(stage, "geometry")

    excluded = load_excluded(stage)
    pids = sorted({k.split("__")[0] for k in verif if k.split("__")[0] not in excluded})

    # per-(pid, model) data
    pmm = {}
    for pid in pids:
        for m in MODELS:
            key = f"{pid}__{m}"
            v = verif.get(key, {})
            ac = v.get("answer_correct")
            ac_int = 1 if ac is True else (0 if ac is False else None)
            scores = []
            for j in JUDGES:
                e = grad[j].get(key)
                if e and e.get("score") is not None:
                    scores.append(e["score"])
            if len(scores) >= 2 and ac_int is not None:
                mean_s = sum(scores) / len(scores)
                sc_lenient = 1 if mean_s > SOL_THRESHOLD else 0
                sc_strict = 1 if (mean_s > SOL_THRESHOLD and ac_int == 1) else 0
            else:
                sc_lenient = sc_strict = None
            pmm[(pid, m)] = {"ans": ac_int, "len": sc_lenient, "str": sc_strict}

    # filters (identical to 05_compute_gap.py)
    canary_hit = {}
    for key, entry in canary.items():
        p = entry["pid"]
        canary_hit[p] = canary_hit.get(p, False) or entry.get("any_correct", False)
    is_geom = {pid: geometry.get(pid, {}).get("is_geometry") for pid in pids}
    is_easy = {pid: rubric_v2.get(pid, {}).get("classification") == "EASY_TO_GUESS" for pid in pids}

    hard = [pid for pid in pids
            if not is_easy.get(pid, False)
            and not canary_hit.get(pid, False)
            and is_geom.get(pid) is not True]

    # negative-gap guard (lenient per-problem gap >= 0)
    def per_problem_gap_lenient(pid):
        ans_vals = [pmm[(pid, m)]["ans"] for m in MODELS if pmm[(pid, m)]["ans"] is not None]
        sol_vals = [pmm[(pid, m)]["len"] for m in MODELS if pmm[(pid, m)]["len"] is not None]
        if not ans_vals or not sol_vals:
            return None
        return (sum(ans_vals) / len(ans_vals) - sum(sol_vals) / len(sol_vals)) * 100

    hard_guard = [pid for pid in hard if (g := per_problem_gap_lenient(pid)) is not None and g >= 0]
    return pmm, hard_guard


def grand_rel_gap(pmm, pids):
    """Grand-mean relative gap (strict). Returns (ans%, sol%, abs_pp, rel%)."""
    ans_all, sol_all = [], []
    for pid in pids:
        for m in MODELS:
            d = pmm[(pid, m)]
            if d["ans"] is not None and d["str"] is not None:
                ans_all.append(d["ans"])
                sol_all.append(d["str"])
    if not ans_all:
        return None
    ans = sum(ans_all) / len(ans_all) * 100
    sol = sum(sol_all) / len(sol_all) * 100
    gap = ans - sol
    rel = gap / ans * 100 if ans > 0 else 0
    return ans, sol, gap, rel


def per_model_first_rel(pmm, pids):
    """Per-model-first relative gap: mean over models of (gap_m / ans_m).
    This is the canonical method per feedback_per_model_first_then_average."""
    rels = []
    gaps = []
    anss = []
    for m in MODELS:
        ans_vals = [pmm[(pid, m)]["ans"] for pid in pids if pmm[(pid, m)]["ans"] is not None]
        sol_vals = [pmm[(pid, m)]["str"] for pid in pids if pmm[(pid, m)]["str"] is not None]
        if not ans_vals or not sol_vals:
            continue
        ans = sum(ans_vals) / len(ans_vals) * 100
        sol = sum(sol_vals) / len(sol_vals) * 100
        gap = ans - sol
        anss.append(ans)
        gaps.append(gap)
        if ans > 0:
            rels.append(gap / ans * 100)
    # two variants: (a) mean of per-model rel gaps; (b) avg-gap / avg-ans
    rel_a = sum(rels) / len(rels) if rels else 0
    rel_b = (sum(gaps) / len(gaps)) / (sum(anss) / len(anss)) * 100 if anss else 0
    return rel_a, rel_b


def per_model_table(pmm, pids):
    rows = []
    for m in MODELS:
        ans_vals = [pmm[(pid, m)]["ans"] for pid in pids if pmm[(pid, m)]["ans"] is not None]
        sol_vals = [pmm[(pid, m)]["str"] for pid in pids if pmm[(pid, m)]["str"] is not None]
        if not ans_vals or not sol_vals:
            continue
        ans = sum(ans_vals) / len(ans_vals) * 100
        sol = sum(sol_vals) / len(sol_vals) * 100
        gap = ans - sol
        rel = gap / ans * 100 if ans > 0 else 0
        rows.append((m, ans, sol, gap, rel, len(ans_vals)))
    return rows


def bootstrap_ci(pmm, pids, B=1000, seed=42):
    """Bootstrap CI on the per-model-first relative gap (canonical method)."""
    rng = random.Random(seed)
    rels = []
    for _ in range(B):
        sample = [pids[rng.randrange(len(pids))] for _ in range(len(pids))]
        rel_a, _ = per_model_first_rel(pmm, sample)
        rels.append(rel_a)
    rels.sort()
    lo = rels[int(0.025 * len(rels))]
    hi = rels[int(0.975 * len(rels))]
    return lo, hi, sum(rels) / len(rels)


def main():
    print("=" * 72)
    print("COMBINED 883 HARD ANALYSIS (stage_a + stage_b, strict + guard)")
    print("=" * 72)

    pmm_a, hard_a = build_stage("stage_a")
    pmm_b, hard_b = build_stage("stage_b")

    print(f"\nStage A HARD (post-guard): {len(hard_a)}")
    print(f"Stage B HARD (post-guard): {len(hard_b)}")
    print(f"Combined HARD: {len(hard_a) + len(hard_b)}")

    # combined pmm with stage-prefixed keys to avoid pid collisions
    pmm_comb = {}
    pids_comb = []
    for (pid, m), v in pmm_a.items():
        pmm_comb[(f"A:{pid}", m)] = v
    for (pid, m), v in pmm_b.items():
        pmm_comb[(f"B:{pid}", m)] = v
    pids_comb = [f"A:{p}" for p in hard_a] + [f"B:{p}" for p in hard_b]

    for label, pmm, pids in [
        ("STAGE A (N=%d)" % len(hard_a), pmm_a, hard_a),
        ("STAGE B (N=%d)" % len(hard_b), pmm_b, hard_b),
        ("COMBINED 883 (N=%d)" % len(pids_comb), pmm_comb, pids_comb),
    ]:
        ans, sol, gap, rel = grand_rel_gap(pmm, pids)
        rel_pmf_a, rel_pmf_b = per_model_first_rel(pmm, pids)
        lo, hi, mean = bootstrap_ci(pmm, pids)
        print(f"\n--- {label} ---")
        print(f"  ans_acc = {ans:.2f}%  sol_acc = {sol:.2f}%  abs gap = +{gap:.2f} pp")
        print(f"  REL GAP grand-mean (ans-sol)/ans          = {rel:.2f}%")
        print(f"  REL GAP per-model-first mean(gap_m/ans_m)  = {rel_pmf_a:.2f}%  <-- canonical method")
        print(f"  REL GAP per-model avg-gap/avg-ans          = {rel_pmf_b:.2f}%")
        print(f"  Bootstrap 95% CI (grand-mean) = [{lo:.2f}%, {hi:.2f}%]  (mean {mean:.2f}%)")
        print(f"  vs AIME 41.74%: margin {rel_pmf_a - 41.74:+.2f} pp (per-model-first)")

    print("\n" + "=" * 72)
    print("PER-MODEL on COMBINED 883 (strict+guard)")
    print("=" * 72)
    print(f"{'Model':24s} {'ans%':>7s} {'sol%':>7s} {'gap pp':>8s} {'rel%':>7s} {'n':>5s}")
    for m, ans, sol, gap, rel, n in per_model_table(pmm_comb, pids_comb):
        print(f"{m:24s} {ans:7.2f} {sol:7.2f} {gap:+8.2f} {rel:7.2f} {n:5d}")

    # Save JSON (canonical metric = per-model-first relative gap)
    def block(pmm, pids):
        ans, sol, gap, _grand = grand_rel_gap(pmm, pids)
        rel_pmf, _ = per_model_first_rel(pmm, pids)
        lo, hi, _ = bootstrap_ci(pmm, pids)
        return {"n": len(pids), "ans": ans, "sol": sol, "abs_gap": gap,
                "rel_gap_canonical": rel_pmf, "rel_gap_grand_mean": _grand,
                "bootstrap_ci": [lo, hi]}
    out = {
        "method": "per-model-first relative gap (mean over models of gap_m/ans_m), strict aggregator (mean>6.5 AND ans_correct), negative-gap guard",
        "aime_baseline_rel": 41.74,
        "stage_a": block(pmm_a, hard_a),
        "stage_b": block(pmm_b, hard_b),
        "combined": block(pmm_comb, pids_comb),
    }
    out["per_model_combined"] = [
        {"model": m, "ans": ans, "sol": sol, "gap": gap, "rel": rel, "n": n}
        for m, ans, sol, gap, rel, n in per_model_table(pmm_comb, pids_comb)
    ]
    save_path = Path(__file__).parent.parent / "results" / "combined_883_report.json"
    json.dump(out, open(save_path, "w"), indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
