"""
Re-extract \\boxed{} from existing inference solutions (bug fix 2026-05-08).
Then re-run S2 verification on entries with new extracted_answer.

NO re-runs inference. Solutions are intact. Only extracted_answer was buggy (last-char-only).

Usage:
  python 06_reextract_revalidate.py --run stage_a
  python 06_reextract_revalidate.py --run smoke
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pilot_run import extract_boxed_from_solution

DATA = Path(__file__).parent.parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    inf_path = DATA / f"{args.run}_inference.json"
    ver_path = DATA / f"{args.run}_verification.json"

    with open(inf_path) as f:
        inf = json.load(f)

    print(f"Re-extracting boxed answers from {len(inf)} solutions...")
    n_changed = 0
    n_now_extracted = 0
    n_was_extracted = 0
    for k, e in inf.items():
        sol = e.get("solution") or ""
        old = e.get("extracted_answer")
        new = extract_boxed_from_solution(sol)  # returns string or None
        if old:
            n_was_extracted += 1
        if new:
            n_now_extracted += 1
        if old != new:
            n_changed += 1
            e["extracted_answer"] = new

    with open(inf_path, "w") as f:
        json.dump(inf, f, indent=2, ensure_ascii=False)
    print(f"  Changed: {n_changed}/{len(inf)} entries")
    print(f"  Was extracted (old): {n_was_extracted}, Now extracted: {n_now_extracted}")

    # Reset verification entries that need re-verification (those whose extracted_answer changed)
    if not ver_path.exists():
        print("No verification file to reset")
        return
    with open(ver_path) as f:
        ver = json.load(f)
    n_reset = 0
    for k, e in inf.items():
        if k in ver:
            old_model_answer = ver[k].get("model_answer", "")
            new_extracted = e.get("extracted_answer", "")
            if not new_extracted:
                continue
            # If the verifier was given old (truncated 1-char) value but model has real answer now -> reset
            if old_model_answer != new_extracted[:300]:
                # Reset this entry
                ver[k] = {
                    "pid": e["pid"], "model": e["model"],
                    "answer_correct": None,
                    "model_answer": new_extracted[:300],
                    "reference_answer": ver[k].get("reference_answer", ""),
                    "raw": "",
                }
                n_reset += 1
    with open(ver_path, "w") as f:
        json.dump(ver, f, indent=2, ensure_ascii=False)
    print(f"  Reset {n_reset} verification entries for re-run")
    print(f"\nNext: re-run S2 verification (resumable, picks up reset entries)")
    print(f"  python 03_pipeline.py --input {args.run}_1000.json --run {args.run} --workers 300 --stages 2")


if __name__ == "__main__":
    main()
