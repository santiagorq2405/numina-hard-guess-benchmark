"""Recompute the headline numbers of Numina-HARD-guess from the released files.

Standard library only; no API calls. Run from the repository root:

    python3 reproduce_headline.py

Expected: HARD (N=881) 21.35%, stage A 20.79%, stage B 21.00%,
unfiltered pool 37.92%, easy-to-guess pool 48.91%, AIME control 41.74%.

Metric: per-model-first relative verification gap,
(answer-only accuracy - solution accuracy) / answer-only accuracy, averaged over
the eight held-out models. A solution counts as correct iff the mean of the three
judge scores is > 6.5/7 AND the final answer is correct (strict aggregator).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent / "data"
JUDGES = ["gpt-5.4", "gemini-3.1-pro", "qwen3-235b"]
MODELS = [
    "claude-haiku-4.5", "claude-sonnet-4.5", "command-a", "llama-3.1-8b",
    "llama-3.3-70b", "mistral-small-3.2-24b", "nova-pro-v1", "qwen2.5-72b",
]
THRESHOLD = 6.5


def load(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pair_table(folder: Path, pids: list[str]) -> dict:
    """(pid, model) -> (answer_correct or None, mean judge score or None)."""
    verif = load(folder / "answer_verification.json")
    grades = {j: load(folder / f"grading_{j}.json") for j in JUDGES}
    table = {}
    for pid in pids:
        for m in MODELS:
            key = f"{pid}__{m}"
            ac = verif.get(key, {}).get("answer_correct")
            scores = [grades[j][key]["score"] for j in JUDGES
                      if grades[j].get(key) and grades[j][key].get("score") is not None]
            mean = sum(scores) / len(scores) if len(scores) >= 2 else None
            table[(pid, m)] = (None if ac is None else bool(ac), mean)
    return table


def relative_gap(table: dict, pids: list[str]) -> float:
    rels = []
    for m in MODELS:
        ans, sol = [], []
        for pid in pids:
            ac, mean = table[(pid, m)]
            if ac is None or mean is None:
                continue
            ans.append(1 if ac else 0)
            sol.append(1 if (ac and mean > THRESHOLD) else 0)
        a = 100 * sum(ans) / len(ans)
        s = 100 * sum(sol) / len(sol)
        rels.append((a - s) / a * 100 if a > 0 else 0.0)
    return sum(rels) / len(rels)


def main() -> None:
    labels = load(ROOT / "pipeline_labels.json")
    tables, pools = {}, {"hard": [], "easy": [], "all": []}
    for stage in ("stage_a", "stage_b"):
        rows = [r for r in labels if r["stage"] == stage]
        pids = [r["pid"] for r in rows]
        tables[stage] = pair_table(ROOT / stage, pids)
        hard = [r["pid"] for r in rows if r["in_numina_hard_guess"]]
        easy = [r["pid"] for r in rows if r["guessability_label"] == "EASY_TO_GUESS"]
        print(f"{stage}: HARD N={len(hard)} rel gap={relative_gap(tables[stage], hard):.2f}%")
        pools["hard"] += [(stage, p) for p in hard]
        pools["easy"] += [(stage, p) for p in easy]
        pools["all"] += [(stage, p) for p in pids]

    combined = {(f"{st}:{p}", m): v for st, t in tables.items() for (p, m), v in t.items()}
    for name in ("hard", "all", "easy"):
        pids = [f"{st}:{p}" for st, p in pools[name]]
        print(f"combined {name:>4}: N={len(pids)} rel gap={relative_gap(combined, pids):.2f}%")

    aime_dir = ROOT / "aime_control"
    aime_pids = sorted({k.split("__")[0] for k in load(aime_dir / "answer_verification.json")})
    aime = relative_gap(pair_table(aime_dir, aime_pids), aime_pids)
    print(f"AIME control: N={len(aime_pids)} rel gap={aime:.2f}%")


if __name__ == "__main__":
    main()
