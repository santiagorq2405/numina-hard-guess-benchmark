"""
Sample N problems from NuminaMath-1.5 olympiads_ref answer-based, EXCLUDING prior_pids.

Design specification:
- Source: AI-MO/NuminaMath-1.5 (public)
- Filter: source == 'olympiads_ref'  (literal "Olympic ref. Answer based problems")
- question_type == 'math-word-problem'  (answer-based vs proof-based)
- problem_is_valid and solution_is_valid

Usage:
  python 02_sample.py --n 10 --out smoke_10.json --seed 2026
  python 02_sample.py --n 1000 --out stage_a_1000.json --seed 2026

Disjoint from data/prior_pids.json (set of SHA256 hashes of normalized problem text).
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRIOR_PATH = Path(__file__).parent.parent / "data" / "prior_pids.json"
OUT_DIR = Path(__file__).parent.parent / "data"


def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def hash_problem(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True, help="filename inside data/")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scan-limit", type=int, default=200000, help="cap streaming scan")
    args = ap.parse_args()

    if not PRIOR_PATH.exists():
        print(f"ERROR: {PRIOR_PATH} not found. Run 01_compile_prior_pids.py first.")
        sys.exit(1)

    with open(PRIOR_PATH, encoding="utf-8") as f:
        prior_data = json.load(f)
    prior_set = set(prior_data["hashes"])
    print(f"Prior PIDs to exclude: {len(prior_set)}")

    from datasets import load_dataset
    print("Streaming AI-MO/NuminaMath-1.5...")
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train", streaming=True)

    # Reservoir sample with seed for reproducibility, after filter + disjoint check
    import random
    rng = random.Random(args.seed)
    candidates = []
    seen = 0
    matched = 0
    skipped_prior = 0
    skipped_invalid = 0

    # NuminaMath-1.5 uses 'olympiads' source for the answer-based subset (no 'olympiads_ref' in v1.5).
    # Answer-based olympiad set = source='olympiads' + question_type='math-word-problem'.
    VALID_SOURCES = {"olympiads", "olympiads_ref"}
    for entry in ds:
        seen += 1
        if entry.get("source") not in VALID_SOURCES:
            continue
        if entry.get("question_type") != "math-word-problem":
            continue
        if not entry.get("problem_is_valid", True):
            skipped_invalid += 1
            continue
        if not entry.get("solution_is_valid", True):
            skipped_invalid += 1
            continue
        problem = entry.get("problem", "")
        h = hash_problem(problem)
        if h in prior_set:
            skipped_prior += 1
            continue
        matched += 1
        candidates.append({
            "post_neurips_id": f"PN_{matched:06d}",
            "hash": h,
            "problem": problem,
            "answer": entry.get("answer", ""),
            "solution": entry.get("solution", ""),
            "problem_type": entry.get("problem_type", ""),
            "question_type": entry.get("question_type", ""),
            "source": entry.get("source", ""),
        })
        if seen % 10000 == 0:
            print(f"  scanned {seen}, matched {matched}, prior {skipped_prior}, invalid {skipped_invalid}")
        if seen >= args.scan_limit:
            break

    print(f"Scan complete. seen={seen}, matched={matched}, skipped_prior={skipped_prior}, skipped_invalid={skipped_invalid}")
    if matched < args.n:
        print(f"WARNING: only {matched} candidates available, requested {args.n}")
        sample = candidates
    else:
        sample = rng.sample(candidates, args.n)

    # Renumber post_neurips_id sequentially in sample order
    for i, p in enumerate(sample, 1):
        p["post_neurips_id"] = f"PN_{i:06d}"

    out_path = OUT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sample)} problems to {out_path}")

    # Disjointness self-check
    sample_hashes = {p["hash"] for p in sample}
    overlap = sample_hashes & prior_set
    print(f"Disjointness check: overlap with prior = {len(overlap)} (must be 0)")
    if overlap:
        print(f"FAIL: sample overlaps with prior. First 3: {list(overlap)[:3]}")
        sys.exit(2)
    print("Disjointness OK")


if __name__ == "__main__":
    main()
