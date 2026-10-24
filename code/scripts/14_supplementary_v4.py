"""
Supplementary v4 numbers for the paper, computed from EXISTING data (no API calls).

For stage_a, stage_b, and combined, under the per-model-first relative-gap metric
(canonical), this script computes:
  1. Pre-filter baseline (all evaluable problems after bad-data exclusion).
  2. EASY-to-guess pool gap (problems removed by the guessability filter).
  3. Threshold sweep on the HARD pool (strict aggregator at mean > t for several t).
  4. Aggregator sensitivity on the HARD pool (lenient vs strict vs strict+guard).

It first reproduces the locked HARD pool sizes (200 / 683) and the locked strict+guard
relative gaps (20.79% / 20.95%) as a self-check before emitting the new numbers.

Output: console + results/supplementary_v4_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
JUDGES = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
MODELS = [
    "claude-haiku-4.5", "claude-sonnet-4.5", "command-a", "llama-3.1-8b",
    "llama-3.3-70b", "mistral-small-3.2-24b", "nova-pro-v1", "qwen2.5-72b",
]


def load(stage: str, name: str) -> dict:
    return json.load(open(DATA / f"{stage}_{name}.json", encoding="utf-8"))


def load_excluded(stage: str) -> set:
    path = DATA / f"{stage}_excluded_pids.json"
    if not path.exists():
        return set()
    return set(json.load(open(path, encoding="utf-8")).get("excluded_pids", []))


def build_stage(stage: str):
    """Return (pmm, pids_all, pids_easy, pids_hard_guard).

    pmm[(pid, model)] = {"ans": 0/1/None, "mean_s": float/None}.
    """
    verif = load(stage, "verification")
    grad = {j: load(stage, f"grading_{j}") for j in JUDGES}
    rubric_v2 = load(stage, "rubric_v2")
    canary = load(stage, "canary")
    geometry = load(stage, "geometry")
    excluded = load_excluded(stage)

    pids = sorted({k.split("__")[0] for k in verif if k.split("__")[0] not in excluded})

    pmm = {}
    for pid in pids:
        for m in MODELS:
            key = f"{pid}__{m}"
            v = verif.get(key, {})
            ac = v.get("answer_correct")
            ac_int = 1 if ac is True else (0 if ac is False else None)
            scores = [grad[j][key]["score"] for j in JUDGES
                      if grad[j].get(key) and grad[j][key].get("score") is not None]
            mean_s = sum(scores) / len(scores) if len(scores) >= 2 else None
            pmm[(pid, m)] = {"ans": ac_int, "mean_s": mean_s}

    canary_hit = {}
    for entry in canary.values():
        p = entry["pid"]
        canary_hit[p] = canary_hit.get(p, False) or entry.get("any_correct", False)
    is_geom = {pid: geometry.get(pid, {}).get("is_geometry") for pid in pids}
    is_easy = {pid: rubric_v2.get(pid, {}).get("classification") == "EASY_TO_GUESS" for pid in pids}

    pids_easy = [pid for pid in pids if is_easy.get(pid, False)]
    hard = [pid for pid in pids
            if not is_easy.get(pid, False)
            and not canary_hit.get(pid, False)
            and is_geom.get(pid) is not True]

    def ppgap_lenient(pid):
        a = [pmm[(pid, m)]["ans"] for m in MODELS if pmm[(pid, m)]["ans"] is not None]
        s = [1 if (pmm[(pid, m)]["mean_s"] is not None and pmm[(pid, m)]["mean_s"] > 6.5) else 0
             for m in MODELS if pmm[(pid, m)]["mean_s"] is not None]
        if not a or not s:
            return None
        return (sum(a) / len(a) - sum(s) / len(s)) * 100

    hard_guard = [pid for pid in hard if (g := ppgap_lenient(pid)) is not None and g >= 0]
    return pmm, pids, pids_easy, hard_guard


def sol_correct(d, thr: float, strict: bool):
    if d["mean_s"] is None:
        return None
    ok = d["mean_s"] > thr
    if strict:
        if d["ans"] is None:
            return None
        return 1 if (ok and d["ans"] == 1) else 0
    return 1 if ok else 0


def per_model_first_rel(pmm, pids, thr: float = 6.5, strict: bool = True):
    """Canonical: mean over models of (gap_m / ans_m). Returns (rel%, ans%, sol%, abs_pp)."""
    rels, gaps, anss, sols = [], [], [], []
    for m in MODELS:
        a = [pmm[(pid, m)]["ans"] for pid in pids if pmm[(pid, m)]["ans"] is not None]
        s = [sc for pid in pids if (sc := sol_correct(pmm[(pid, m)], thr, strict)) is not None]
        if not a or not s:
            continue
        ans = sum(a) / len(a) * 100
        sol = sum(s) / len(s) * 100
        gap = ans - sol
        anss.append(ans); sols.append(sol); gaps.append(gap)
        if ans > 0:
            rels.append(gap / ans * 100)
    rel = sum(rels) / len(rels) if rels else 0.0
    ans_m = sum(anss) / len(anss) if anss else 0.0
    sol_m = sum(sols) / len(sols) if sols else 0.0
    gap_m = sum(gaps) / len(gaps) if gaps else 0.0
    return rel, ans_m, sol_m, gap_m


def main():
    pmm_a, all_a, easy_a, hard_a = build_stage("stage_a")
    pmm_b, all_b, easy_b, hard_b = build_stage("stage_b")

    # combined (prefix to avoid pid collisions)
    pmm_c = {}
    for (pid, m), v in pmm_a.items():
        pmm_c[(f"A:{pid}", m)] = v
    for (pid, m), v in pmm_b.items():
        pmm_c[(f"B:{pid}", m)] = v
    all_c = [f"A:{p}" for p in all_a] + [f"B:{p}" for p in all_b]
    easy_c = [f"A:{p}" for p in easy_a] + [f"B:{p}" for p in easy_b]
    hard_c = [f"A:{p}" for p in hard_a] + [f"B:{p}" for p in hard_b]

    stages = [
        ("stage_a", pmm_a, all_a, easy_a, hard_a),
        ("stage_b", pmm_b, all_b, easy_b, hard_b),
        ("combined", pmm_c, all_c, easy_c, hard_c),
    ]

    print("=" * 78)
    print("SELF-CHECK (must match locked: HARD 200/683/883; strict+guard 20.79/20.95/21.31)")
    print("=" * 78)
    for name, pmm, _all, _easy, hard in stages:
        rel, ans, sol, gap = per_model_first_rel(pmm, hard, 6.5, True)
        print(f"{name:9s} HARD N={len(hard):4d}  strict@6.5 rel={rel:6.2f}%  (ans {ans:.2f} sol {sol:.2f} gap +{gap:.2f})")

    out = {"metric": "per-model-first relative gap; strict = mean(3 judges) > t AND answer correct",
           "aime_baseline_rel": 41.74, "stages": {}}

    for name, pmm, allp, easyp, hard in stages:
        print("\n" + "=" * 78)
        print(f"{name.upper()}  (all={len(allp)}  easy={len(easyp)}  hard={len(hard)})")
        print("=" * 78)

        pre_rel, pre_ans, pre_sol, pre_gap = per_model_first_rel(pmm, allp, 6.5, True)
        easy_rel, easy_ans, easy_sol, easy_gap = per_model_first_rel(pmm, easyp, 6.5, True)
        hard_rel, hard_ans, hard_sol, hard_gap = per_model_first_rel(pmm, hard, 6.5, True)

        print(f"  PRE-FILTER baseline (N={len(allp)}): rel={pre_rel:.2f}%  abs=+{pre_gap:.2f}  ans={pre_ans:.2f} sol={pre_sol:.2f}")
        print(f"  EASY-to-guess pool  (N={len(easyp)}): rel={easy_rel:.2f}%  abs=+{easy_gap:.2f}  ans={easy_ans:.2f} sol={easy_sol:.2f}")
        print(f"  HARD pool           (N={len(hard)}): rel={hard_rel:.2f}%  abs=+{hard_gap:.2f}  ans={hard_ans:.2f} sol={hard_sol:.2f}")
        print(f"  EASY - HARD separation: {easy_rel - hard_rel:+.2f} pp (relative), {easy_gap - hard_gap:+.2f} pp (absolute)")

        print("  Threshold sweep on HARD (strict, per-model-first rel gap):")
        sweep = {}
        for thr in [5.5, 6.0, 6.5, 7.0]:
            r, a, s, g = per_model_first_rel(pmm, hard, thr, True)
            sweep[f"{thr}"] = {"rel": r, "abs": g, "ans": a, "sol": s}
            print(f"    mean>{thr}: rel={r:6.2f}%  abs=+{g:5.2f}  (sol {s:.2f})")

        print("  Aggregator sensitivity on HARD pool:")
        len_rel, len_a, len_s, len_g = per_model_first_rel(pmm, hard, 6.5, False)
        str_rel, str_a, str_s, str_g = per_model_first_rel(pmm, hard, 6.5, True)
        print(f"    lenient@6.5   : rel={len_rel:6.2f}%  abs=+{len_g:5.2f}")
        print(f"    strict@6.5    : rel={str_rel:6.2f}%  abs=+{str_g:5.2f}  (= strict+guard, pool already guarded)")

        out["stages"][name] = {
            "n_all": len(allp), "n_easy": len(easyp), "n_hard": len(hard),
            "pre_filter": {"rel": pre_rel, "abs": pre_gap, "ans": pre_ans, "sol": pre_sol},
            "easy_pool": {"rel": easy_rel, "abs": easy_gap, "ans": easy_ans, "sol": easy_sol},
            "hard_pool": {"rel": hard_rel, "abs": hard_gap, "ans": hard_ans, "sol": hard_sol},
            "separation_rel": easy_rel - hard_rel, "separation_abs": easy_gap - hard_gap,
            "threshold_sweep": sweep,
            "aggregator": {"lenient_6.5": {"rel": len_rel, "abs": len_g},
                           "strict_6.5": {"rel": str_rel, "abs": str_g}},
        }

    save = Path(__file__).parent.parent / "results" / "supplementary_v4_report.json"
    json.dump(out, open(save, "w"), indent=2)
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
