"""
Compare guessability judging rules on the released data (no API calls):
  - SINGLE  (Gemini-3.1-Pro only)            <- the rule used for Numina-HARD-guess (200 + 681 = 881)
  - GPTONLY (GPT-5.4 only)
  - DUAL    (AND rule: HARD only if BOTH say HARD; remove if EITHER says EASY)

For each rule and each stage (+combined), reports HARD pool size and the per-model-first
relative gap (strict + negative-gap guard), plus bootstrap CI for DUAL.

GPT-5.4 guessability comes from stage_*_rubric_v2_gpt.json (produced by 15_*.py).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
DROP = {"PN_000062", "PN_001083"}  # decontaminated near-duplicates (stage_b ids) -> N=881
JUDGES = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
MODELS = [
    "claude-haiku-4.5", "claude-sonnet-4.5", "command-a", "llama-3.1-8b",
    "llama-3.3-70b", "mistral-small-3.2-24b", "nova-pro-v1", "qwen2.5-72b",
]


def load(stage, name):
    return json.load(open(DATA / f"{stage}_{name}.json", encoding="utf-8"))


def load_excluded(stage):
    p = DATA / f"{stage}_excluded_pids.json"
    return set(json.load(open(p, encoding="utf-8")).get("excluded_pids", [])) if p.exists() else set()


def build(stage):
    verif = load(stage, "verification")
    grad = {j: load(stage, f"grading_{j}") for j in JUDGES}
    rub_gem = load(stage, "rubric_v2")
    rub_gpt = load(stage, "rubric_v2_gpt")
    canary = load(stage, "canary")
    geometry = load(stage, "geometry")
    excluded = load_excluded(stage)
    drop = DROP if stage == "stage_b" else set()
    pids = sorted({k.split("__")[0] for k in verif
                   if k.split("__")[0] not in excluded and k.split("__")[0] not in drop})

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
    for e in canary.values():
        canary_hit[e["pid"]] = canary_hit.get(e["pid"], False) or e.get("any_correct", False)
    is_geom = {pid: geometry.get(pid, {}).get("is_geometry") for pid in pids}
    easy_gem = {pid: rub_gem.get(pid, {}).get("classification") == "EASY_TO_GUESS" for pid in pids}
    easy_gpt = {pid: rub_gpt.get(pid, {}).get("classification") == "EASY_TO_GUESS" for pid in pids}

    return pmm, pids, canary_hit, is_geom, easy_gem, easy_gpt


def hard_pool(pids, pmm, canary_hit, is_geom, is_easy_fn):
    cand = [pid for pid in pids
            if not is_easy_fn(pid) and not canary_hit.get(pid, False)
            and is_geom.get(pid) is not True]

    def ppgap_lenient(pid):
        a = [pmm[(pid, m)]["ans"] for m in MODELS if pmm[(pid, m)]["ans"] is not None]
        s = [1 if (pmm[(pid, m)]["mean_s"] is not None and pmm[(pid, m)]["mean_s"] > 6.5) else 0
             for m in MODELS if pmm[(pid, m)]["mean_s"] is not None]
        if not a or not s:
            return None
        return (sum(a) / len(a) - sum(s) / len(s)) * 100

    return [pid for pid in cand if (g := ppgap_lenient(pid)) is not None and g >= 0]


def pmf_rel(pmm, pids):
    """per-model-first strict relative gap; returns (rel%, ans%, sol%, abs_pp)."""
    rels, gaps, anss, sols = [], [], [], []
    for m in MODELS:
        a = [pmm[(pid, m)]["ans"] for pid in pids if pmm[(pid, m)]["ans"] is not None]
        s = [(1 if (pmm[(pid, m)]["mean_s"] is not None and pmm[(pid, m)]["mean_s"] > 6.5
                    and pmm[(pid, m)]["ans"] == 1) else 0)
             for pid in pids if pmm[(pid, m)]["mean_s"] is not None and pmm[(pid, m)]["ans"] is not None]
        if not a or not s:
            continue
        ans, sol = sum(a) / len(a) * 100, sum(s) / len(s) * 100
        anss.append(ans); sols.append(sol); gaps.append(ans - sol)
        if ans > 0:
            rels.append((ans - sol) / ans * 100)
    rel = sum(rels) / len(rels) if rels else 0.0
    return rel, (sum(anss)/len(anss) if anss else 0), (sum(sols)/len(sols) if sols else 0), (sum(gaps)/len(gaps) if gaps else 0)


def bootstrap(pmm, pids, B=1000, seed=42):
    rng = random.Random(seed)
    rels = []
    for _ in range(B):
        samp = [pids[rng.randrange(len(pids))] for _ in range(len(pids))]
        rels.append(pmf_rel(pmm, samp)[0])
    rels.sort()
    return rels[int(0.025*len(rels))], rels[int(0.975*len(rels))]


def main():
    stages = {}
    for st in ["stage_a", "stage_b"]:
        stages[st] = build(st)

    rules = {
        "SINGLE (Gemini only)": lambda eg, ep: (lambda pid: eg.get(pid, False)),
        "GPTONLY (GPT-5.4 only)": lambda eg, ep: (lambda pid: ep.get(pid, False)),
        "DUAL (AND: either EASY removes)": lambda eg, ep: (lambda pid: eg.get(pid, False) or ep.get(pid, False)),
    }

    # diagnostic: judge EASY counts + agreement
    print("=" * 80)
    print("GUESSABILITY JUDGE DIAGNOSTICS (on evaluable pids)")
    print("=" * 80)
    for st in ["stage_a", "stage_b"]:
        pmm, pids, ch, ig, eg, ep = stages[st]
        n = len(pids)
        ng = sum(eg.values()); npt = sum(ep.values())
        both = sum(1 for pid in pids if eg.get(pid) and ep.get(pid))
        either = sum(1 for pid in pids if eg.get(pid) or ep.get(pid))
        agree = sum(1 for pid in pids if eg.get(pid, False) == ep.get(pid, False))
        print(f"{st}: evaluable={n}  Gemini EASY={ng} ({ng/n*100:.0f}%)  GPT EASY={npt} ({npt/n*100:.0f}%)  "
              f"both EASY={both}  either EASY={either}  judges agree={agree} ({agree/n*100:.0f}%)")

    results = {}
    for rule_name, rule_factory in rules.items():
        print("\n" + "=" * 80)
        print(f"RULE: {rule_name}")
        print("=" * 80)
        comb_pmm, comb_pids = {}, []
        per_stage = {}
        for st in ["stage_a", "stage_b"]:
            pmm, pids, ch, ig, eg, ep = stages[st]
            is_easy = rule_factory(eg, ep)
            hp = hard_pool(pids, pmm, ch, ig, is_easy)
            rel, ans, sol, gap = pmf_rel(pmm, hp)
            per_stage[st] = (len(hp), rel, ans, sol, gap)
            print(f"  {st}: N_hard={len(hp):4d}  rel={rel:6.2f}%  abs=+{gap:.2f}  (ans {ans:.2f} sol {sol:.2f})")
            for (pid, m), v in pmm.items():
                comb_pmm[(f"{st}:{pid}", m)] = v
            comb_pids += [f"{st}:{pid}" for pid in hp]
        rel, ans, sol, gap = pmf_rel(comb_pmm, comb_pids)
        lo, hi = bootstrap(comb_pmm, comb_pids)
        print(f"  COMBINED: N_hard={len(comb_pids):4d}  rel={rel:6.2f}%  abs=+{gap:.2f}  CI=[{lo:.2f}, {hi:.2f}]  vs AIME 41.74%")
        results[rule_name] = {"per_stage": per_stage,
                              "combined": {"n": len(comb_pids), "rel": rel, "abs": gap, "ci": [lo, hi]}}

    # marginal removals: problems Gemini KEPT but GPT removed (and their gap)
    print("\n" + "=" * 80)
    print("MARGINAL: problems Gemini->HARD but GPT->EASY (what dual additionally removes)")
    print("=" * 80)
    for st in ["stage_a", "stage_b"]:
        pmm, pids, ch, ig, eg, ep = stages[st]
        # in the single HARD pool (Gemini), how many would dual remove via GPT-EASY?
        single_hp = hard_pool(pids, pmm, ch, ig, lambda pid: eg.get(pid, False))
        removed_by_gpt = [pid for pid in single_hp if ep.get(pid, False)]
        kept = [pid for pid in single_hp if not ep.get(pid, False)]
        if removed_by_gpt:
            rel_rm, ans_rm, sol_rm, gap_rm = pmf_rel(pmm, removed_by_gpt)
        else:
            rel_rm = gap_rm = ans_rm = 0
        rel_kept, _, _, gap_kept = pmf_rel(pmm, kept)
        print(f"  {st}: single HARD={len(single_hp)} -> GPT removes {len(removed_by_gpt)} (rel={rel_rm:.2f}% abs=+{gap_rm:.2f}), "
              f"keeps {len(kept)} (rel={rel_kept:.2f}% abs=+{gap_kept:.2f})")
        print(f"        (if GPT removes problems with HIGHER gap than kept, dual is genuinely better)")

    out = Path(__file__).parent.parent / "results" / "dual_vs_single_report.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
