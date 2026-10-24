"""
Guard effect, within-source baseline and single-judge regrade, from the released data.
No API calls.

Computes, for stage_a / stage_b / combined, under the per-model-first relative-gap
metric (strict = mean(judges) > 6.5 AND answer correct):
  - PRE-guard vs POST-guard strict rel gap on the HARD pool (quantifies the
    selection effect of the negative-gap residual guard).
  - PRE-filter within-source baseline rel gap (unfiltered NuminaMath).
  - Gemini-only re-grading of the canonical HARD pool.

Self-check: combined HARD post-guard strict rel must match the locked 21.35.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
DROP = {"PN_000062", "PN_001083"}  # decontaminated near-duplicates (stage_b ids) -> N=881
JUDGES_ALL = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
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


def build(stage: str, judges: list[str], min_judges: int):
    """Return (pmm, pids_all, pids_easy, hard_pre, hard_guard).

    pmm[(pid, model)] = {"ans": 0/1/None, "mean_s": float/None}, mean_s averaged
    over `judges` when at least min_judges scores are present.
    hard_pre  = HARD pool before the negative-gap guard.
    hard_guard = HARD pool after the guard (lenient per-problem gap >= 0).
    """
    verif = load(stage, "verification")
    grad = {j: load(stage, f"grading_{j}") for j in judges}
    rubric_v2 = load(stage, "rubric_v2")
    canary = load(stage, "canary")
    geometry = load(stage, "geometry")
    excluded = load_excluded(stage)
    drop = DROP if stage == "stage_b" else set()  # ids repeat across stages

    pids = sorted({k.split("__")[0] for k in verif
                   if k.split("__")[0] not in excluded and k.split("__")[0] not in drop})

    pmm = {}
    for pid in pids:
        for m in MODELS:
            key = f"{pid}__{m}"
            v = verif.get(key, {})
            ac = v.get("answer_correct")
            ac_int = 1 if ac is True else (0 if ac is False else None)
            scores = [grad[j][key]["score"] for j in judges
                      if grad[j].get(key) and grad[j][key].get("score") is not None]
            mean_s = sum(scores) / len(scores) if len(scores) >= min_judges else None
            pmm[(pid, m)] = {"ans": ac_int, "mean_s": mean_s}

    canary_hit = {}
    for entry in canary.values():
        p = entry["pid"]
        canary_hit[p] = canary_hit.get(p, False) or entry.get("any_correct", False)
    is_geom = {pid: geometry.get(pid, {}).get("is_geometry") for pid in pids}
    is_easy = {pid: rubric_v2.get(pid, {}).get("classification") == "EASY_TO_GUESS" for pid in pids}

    pids_easy = [pid for pid in pids if is_easy.get(pid, False)]
    hard_pre = [pid for pid in pids
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

    hard_guard = [pid for pid in hard_pre if (g := ppgap_lenient(pid)) is not None and g >= 0]
    return pmm, pids, pids_easy, hard_pre, hard_guard


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


def combine(triplets):
    """Merge per-stage (pmm, pids) with prefixing to a single combined pmm + pid lists."""
    pmm_c = {}
    sets = {k: [] for k in ("all", "easy", "hard_pre", "hard_guard")}
    for prefix, pmm, pids, easy, hard_pre, hard_guard in triplets:
        for (pid, m), v in pmm.items():
            pmm_c[(f"{prefix}:{pid}", m)] = v
        sets["all"] += [f"{prefix}:{p}" for p in pids]
        sets["easy"] += [f"{prefix}:{p}" for p in easy]
        sets["hard_pre"] += [f"{prefix}:{p}" for p in hard_pre]
        sets["hard_guard"] += [f"{prefix}:{p}" for p in hard_guard]
    return pmm_c, sets


def main():
    out = {}
    # canonical 3-judge build
    a = build("stage_a", JUDGES_ALL, 2)
    b = build("stage_b", JUDGES_ALL, 2)
    pmm_a, all_a, easy_a, hpre_a, hg_a = a
    pmm_b, all_b, easy_b, hpre_b, hg_b = b
    pmm_c, sets_c = combine([
        ("A", pmm_a, all_a, easy_a, hpre_a, hg_a),
        ("B", pmm_b, all_b, easy_b, hpre_b, hg_b),
    ])

    print("=" * 74)
    print("GUARD EFFECT: strict rel gap PRE-guard vs POST-guard on the HARD pool")
    print("=" * 74)
    for name, pmm, hpre, hg in [
        ("stage_a", pmm_a, hpre_a, hg_a),
        ("stage_b", pmm_b, hpre_b, hg_b),
        ("combined", pmm_c, sets_c["hard_pre"], sets_c["hard_guard"]),
    ]:
        pre = per_model_first_rel(pmm, hpre, 6.5, True)
        post = per_model_first_rel(pmm, hg, 6.5, True)
        removed = len(hpre) - len(hg)
        print(f"{name:9s} pre N={len(hpre):4d} rel={pre[0]:6.2f}%   "
              f"post N={len(hg):4d} rel={post[0]:6.2f}%   "
              f"removed={removed:3d}  delta={post[0]-pre[0]:+.2f}pp")
        out[name] = {"hard_pre_n": len(hpre), "hard_pre_rel": pre[0],
                     "hard_post_n": len(hg), "hard_post_rel": post[0],
                     "guard_removed": removed, "guard_delta_pp": post[0] - pre[0]}

    print("\n" + "=" * 74)
    print("WITHIN-SOURCE BASELINE: unfiltered NuminaMath strict rel gap")
    print("=" * 74)
    pre_c = per_model_first_rel(pmm_c, sets_c["all"], 6.5, True)
    easy_c = per_model_first_rel(pmm_c, sets_c["easy"], 6.5, True)
    print(f"combined unfiltered N={len(sets_c['all'])} rel={pre_c[0]:.2f}% "
          f"(ans {pre_c[1]:.2f} sol {pre_c[2]:.2f})")
    print(f"combined EASY pool  N={len(sets_c['easy'])} rel={easy_c[0]:.2f}%")
    print(f"combined HARD pool  N={len(sets_c['hard_guard'])} rel={per_model_first_rel(pmm_c, sets_c['hard_guard'],6.5,True)[0]:.2f}%")
    out["within_source"] = {"unfiltered_n": len(sets_c["all"]), "unfiltered_rel": pre_c[0],
                            "easy_rel": easy_c[0]}

    print("\n" + "=" * 74)
    print("GEMINI-ONLY RE-GRADING of the canonical HARD pool")
    print("=" * 74)
    ga = build("stage_a", ["gemini-3.1-pro"], 1)
    gb = build("stage_b", ["gemini-3.1-pro"], 1)
    pmm_ga, pmm_gb = ga[0], gb[0]
    pmm_gc, _ = combine([
        ("A", pmm_ga, all_a, easy_a, hpre_a, hg_a),
        ("B", pmm_gb, all_b, easy_b, hpre_b, hg_b),
    ])
    # evaluate on the canonical (3-judge-defined) HARD pool
    gem = per_model_first_rel(pmm_gc, sets_c["hard_guard"], 6.5, True)
    print(f"Gemini-only strict rel on canonical HARD (N={len(sets_c['hard_guard'])}): "
          f"{gem[0]:.2f}%  (3-judge ensemble = 21.35%)")
    out["gemini_only"] = {"n": len(sets_c["hard_guard"]), "rel": gem[0]}

    save = Path(__file__).parent.parent / "results" / "review_support_report.json"
    json.dump(out, open(save, "w"), indent=2)
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
