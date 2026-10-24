"""
Phase 1: Identify bad-data PIDs in stage_a_1000.json.

Bad data categories (from 9-agent audit):
- answer in {"proof", "qed", "see solution", "notfound", ""}
- problem text < 100 chars (truncated upstream)
- empty solution field

Output: data/stage_a_excluded_pids.json
"""

import json
from pathlib import Path
from collections import Counter

DATA = Path(__file__).parent.parent / "data"

BAD_ANSWERS = {"proof", "qed", "qed.", "see solution", "see above",
               "notfound", "not found", "see proof", ""}


def main():
    with open(DATA / "stage_a_1000.json") as f:
        problems = json.load(f)

    excluded = []
    reasons = Counter()
    for p in problems:
        pid = p["post_neurips_id"]
        ans = (p.get("answer") or "").strip().lower()
        prob_text = p.get("problem") or ""
        sol = p.get("solution") or ""
        reason = None
        if ans in BAD_ANSWERS:
            reason = f"bad_answer:{ans!r}"
        elif len(prob_text.strip()) < 100:
            reason = "short_problem"
        elif not sol.strip():
            reason = "empty_solution"
        if reason:
            excluded.append({"pid": pid, "reason": reason,
                             "answer": p.get("answer", "")[:50],
                             "problem_preview": prob_text[:120]})
            reasons[reason.split(":")[0]] += 1

    out = {
        "n_total": len(problems),
        "n_excluded": len(excluded),
        "n_kept": len(problems) - len(excluded),
        "reasons": dict(reasons),
        "excluded_pids": [e["pid"] for e in excluded],
        "details": excluded,
    }
    out_path = DATA / "stage_a_excluded_pids.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Total: {out['n_total']}, Excluded: {out['n_excluded']}, Kept: {out['n_kept']}")
    print(f"Reasons: {dict(reasons)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
