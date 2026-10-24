"""
Compile SHA256 hashes of all problems already used in prior runs.
Output: data/prior_pids.json (set of hashes for disjointness check).

Sources walked:
- data/pilot_100_problems.json
- data/scale_500_problems.json
- data/smoke_500/smoke_500_problems.json (if exists)
- data/fase_a/* problem JSONs
- data/fase_b/fase_b_prefiltered_2000.json
- data/prior/FINAL_HARD_268.json (earlier version)
- AIME data if any (separate source from NuminaMath, but include for safety)

Normalization: strip + lowercase + collapse whitespace before hashing.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # working dir root
OUT = Path(__file__).parent.parent / "data" / "prior_pids.json"

CANDIDATE_FILES = [
    "data/pilot/pilot_100_problems.json",
    "data/pilot_100_problems.json",
    "data/scale_500/scale_500_problems.json",
    "data/scale_500_problems.json",
    "data/smoke_500/smoke_500_problems.json",
    "data/fase_a/fase_a_problems.json",
    "data/fase_b/fase_b_prefiltered_2000.json",
    "data/fase_b/fase_b_olympiads_pool.json",
    "data/aime/aime_problems.json",
    "data/held_out/heldout_problems.json",
    "data/prior/FINAL_HARD_268.json",
    "data/prior/FINAL_HARD_134.json",
    "data/FINAL_HARD/FINAL_HARD_pilot.json",
    "data/FINAL_HARD/FINAL_HARD_pilot_v2.json",
    "data/FINAL_HARD_final.json",
    "data/manual_check/hacked_problems.json",
    # stage_a 1000 (for stage_b_3000 disjointness)
    "data/stage_a_1000.json",
]


def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def hash_problem(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def extract_problems_from_obj(obj):
    """Yield problem texts from arbitrary JSON structure."""
    if isinstance(obj, dict):
        # PID-keyed dict: {"pilot_id": {...}}
        if all(isinstance(v, dict) for v in obj.values()):
            for v in obj.values():
                yield from extract_problems_from_obj(v)
        # Single problem dict
        for key in ["problem", "problem_text", "question"]:
            if key in obj and isinstance(obj[key], str):
                yield obj[key]
                return
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_problems_from_obj(item)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pid_set = set()
    sources_walked = {}
    for rel in CANDIDATE_FILES:
        path = ROOT / rel
        if not path.exists():
            sources_walked[rel] = "MISSING"
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            sources_walked[rel] = f"ERROR: {e}"
            continue
        before = len(pid_set)
        for problem_text in extract_problems_from_obj(data):
            if problem_text and len(problem_text) > 20:
                pid_set.add(hash_problem(problem_text))
        sources_walked[rel] = f"OK ({len(pid_set) - before} new)"

    out = {
        "total_unique_hashes": len(pid_set),
        "sources_walked": sources_walked,
        "hashes": sorted(pid_set),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Total unique prior hashes: {len(pid_set)}")
    for src, status in sources_walked.items():
        print(f"  {src}: {status}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
