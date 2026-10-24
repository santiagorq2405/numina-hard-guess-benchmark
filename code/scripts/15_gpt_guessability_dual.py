"""
Second guessability judge: GPT-5.4 on stage_a (1000) + stage_b (3000).

Reuses the EXACT same rubric_v2 prompt, parser, and call_api as the canonical
pipeline (03_pipeline.py), but with GPT-5.4 as the model instead of Gemini-3.1-Pro.
This lets us reconstruct the dual-judge AND rule (HARD only if BOTH judges say HARD)
that the original paper used, and compare against the single-judge (Gemini) v4.

Outputs: data/stage_a_rubric_v2_gpt.json, data/stage_b_rubric_v2_gpt.json
Resumable. Tracks token usage for a real cost estimate.

Usage:
  python 15_gpt_guessability_dual.py --limit 5        # smoke (5 per stage)
  python 15_gpt_guessability_dual.py                  # full 4000
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).parent
DATA = SCRIPTS.parent / "data"

# Load the canonical pipeline module (name starts with a digit -> importlib)
_spec = importlib.util.spec_from_file_location("pipeline_mod", SCRIPTS / "03_pipeline.py")
pipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipe)

GPT = {"client": "openai", "model_id": "gpt-5.4", "max_tokens": 64000}

_tok_lock = threading.Lock()
_tok = {"in": 0, "out": 0}


def run_stage(stage_file: str, out_file: str, limit: int | None, workers: int):
    problems = pipe.load_json(DATA / stage_file, [])
    if not isinstance(problems, list):
        raise SystemExit(f"{stage_file} not a list")
    if limit:
        problems = problems[:limit]
    out_path = DATA / out_file
    state = pipe.load_json(out_path, {})

    def task(p):
        pid = p["post_neurips_id"]
        if pid in state and state[pid].get("classification"):
            return None
        prompt = pipe.GUESSABILITY_RUBRIC_V2_PROMPT.format(
            problem=p["problem"], answer=p["answer"], solution=(p.get("solution") or "")[:3000],
        )
        r = pipe.call_api(GPT["client"], GPT["model_id"], prompt, max_tokens=GPT["max_tokens"])
        with _tok_lock:
            _tok["in"] += r.get("input_tokens", 0)
            _tok["out"] += r.get("output_tokens", 0)
        parsed = pipe.parse_rubric_v2_response(r.get("content", ""))
        return pid, {"pid": pid, "classification": parsed.get("classification"),
                     "primary_strategy": parsed.get("primary_strategy"),
                     "confidence": parsed.get("confidence"),
                     "raw": (r.get("content") or "")[:1000],
                     "error": r.get("error")}

    todo = [p for p in problems if p["post_neurips_id"] not in state
            or not state[p["post_neurips_id"]].get("classification")]
    print(f"[{stage_file}] {len(problems)} problems, {len(todo)} pending", flush=True)
    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, p) for p in todo]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                pid, entry = r
                state[pid] = entry
            n_done += 1
            if n_done % max(20, len(todo) // 20 or 1) == 0 or n_done == len(todo):
                pipe.save_atomic(state, out_path)
                print(f"  [{stage_file}] {n_done}/{len(todo)}", flush=True)
    pipe.save_atomic(state, out_path)

    # quick classification tally
    easy = sum(1 for v in state.values() if v.get("classification") == "EASY_TO_GUESS")
    hard = sum(1 for v in state.values() if v.get("classification") == "HARD_TO_GUESS")
    err = sum(1 for v in state.values() if v.get("error"))
    print(f"  [{stage_file}] DONE: {len(state)} labeled  EASY={easy} HARD={hard} errors={err}", flush=True)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="problems per stage (smoke)")
    ap.add_argument("--workers", type=int, default=150)
    ap.add_argument("--stage", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    t0 = time.time()
    if args.stage in ("a", "both"):
        run_stage("stage_a_1000.json", "stage_a_rubric_v2_gpt.json", args.limit, args.workers)
    if args.stage in ("b", "both"):
        run_stage("stage_b_3000.json", "stage_b_rubric_v2_gpt.json", args.limit, args.workers)

    dt = time.time() - t0
    n_calls = sum(1 for _ in [None])  # placeholder; token-based estimate below
    # GPT-5.4 pricing assumption (OpenAI): adjust if your account differs.
    PRICE_IN, PRICE_OUT = 5.0, 30.0  # USD per 1M tokens (conservative upper-bound)
    cost = _tok["in"] / 1e6 * PRICE_IN + _tok["out"] / 1e6 * PRICE_OUT
    print("=" * 70)
    print(f"Wall-clock: {dt/60:.1f} min")
    print(f"Tokens: in={_tok['in']:,}  out={_tok['out']:,}")
    print(f"Estimated cost (this run): ${cost:.2f}  (at ${PRICE_IN}/${PRICE_OUT} per 1M in/out)")
    if args.limit:
        per_problem = cost / max(1, (args.limit * (2 if args.stage == 'both' else 1)))
        print(f"Per-problem: ${per_problem:.4f}  ->  projected full 4000: ${per_problem*4000:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
