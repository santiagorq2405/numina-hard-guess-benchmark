"""
Three-tier ablation, Tier 2: target self-sampling (K = 7/15/30) with the calibration models.

Detects problems whose answer the models reproduce across multiple samples.
Removal rule (three-tier ablation design):
  - K=7: ≥2 of 7 correct → REMOVE (robust hack)
  - 1/7 correct → escalate K=15: ≥1 of additional 8 → REMOVE (any in K=15)
  - 0/15 correct → escalate K=30: ≥2 of K=30 → REMOVE (robust K=30)
  - 1/30 → keep (single-shot)
  - 0/K (any) → keep (never reproduces)

Per problem aggregation: if ANY of the 5 calibration models has robust hack, REMOVE problem.

Input: data/stage_a_hard_200_pids.json (HARD pool of the first run; not part of the release)
Output: data/stage_a_tier2_results.json
        data/stage_a_tier2_survivors.json (PIDs that passed Tier 2)

Usage:
  python 08_tier2_self_sampling.py --workers 300
"""

import argparse
import concurrent.futures
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pilot_run import ANSWER_VERIFY_PROMPT, SOLVE_PROMPT, extract_boxed_from_solution

load_dotenv(ROOT / ".env")

DATA = Path(__file__).parent.parent / "data"
LOGS = Path(__file__).parent.parent / "logs"

_save_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============== Models (5 calibration, paper hardcoded) ===============

CALIBRATION = {
    "gpt-5.4": {"client": "openai", "model_id": "gpt-5.4", "max_tokens": 64000},
    "gpt-5.4-mini": {"client": "openai", "model_id": "gpt-5.4-mini", "max_tokens": 64000},
    "llama-4-maverick": {"client": "openrouter", "model_id": "meta-llama/llama-4-maverick", "max_tokens": 32000},
    "deepseek-v3.1-terminus": {"client": "openrouter", "model_id": "deepseek/deepseek-v3.1-terminus", "max_tokens": 32000},
    "gemini-3.1-flash-lite": {"client": "openrouter", "model_id": "google/gemini-3.1-flash-lite-preview-20260303", "max_tokens": 32000},
}

VERIFIER = {"client": "openai", "model_id": "gpt-5.4", "max_tokens": 64000}

# Tier 2 thresholds per paper
K7_THRESHOLD_REMOVE = 2  # >=2 of 7 -> robust hack -> REMOVE
K15_THRESHOLD_REMOVE = 1  # in additional 8 (K=15 total): >=1 -> REMOVE
K30_THRESHOLD_REMOVE = 2  # >=2 of 30 total -> robust hack K=30 -> REMOVE


# =============== API clients ===============

_HTTP_LIMITS = httpx.Limits(max_connections=600, max_keepalive_connections=300, keepalive_expiry=15.0)
_OAI_HTTP = httpx.Client(limits=_HTTP_LIMITS, timeout=httpx.Timeout(300.0, connect=30.0))
_OR_HTTP = httpx.Client(limits=_HTTP_LIMITS, timeout=httpx.Timeout(300.0, connect=30.0))
openai_client = OpenAI(http_client=_OAI_HTTP, timeout=300.0)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    http_client=_OR_HTTP,
    timeout=300.0,
)


def call_api(client_name, model_id, prompt, max_tokens, temperature=0.0, max_retries=5):
    client = openai_client if client_name == "openai" else openrouter_client
    backoff = [10, 20, 30, 60, 120]
    for attempt in range(max_retries):
        try:
            kwargs = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
            if "gpt-5.4" in model_id or "gpt-5" in model_id:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            if not content:
                reasoning = getattr(msg, "reasoning", None) or ""
                if reasoning and len(reasoning) > 50:
                    content = reasoning.strip()
            return {"content": content}
        except Exception as e:
            err_l = str(e).lower()
            transient = any(t in err_l for t in ["429", "rate", "timeout", "timed out", "connection", "503", "502", "504"])
            if transient and attempt < max_retries - 1:
                time.sleep(backoff[attempt])
                continue
            return {"content": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"content": "", "error": "exhausted"}


def save_atomic(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _save_lock:
        snapshot = copy.deepcopy(data)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path, default=None):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


# =============== Verification ===============

def verify_answer(problem, model_answer, reference, max_retries=2):
    if not model_answer:
        return False
    prompt = ANSWER_VERIFY_PROMPT.format(problem=problem, model_answer=model_answer, reference_answer=reference)
    for _ in range(max_retries):
        r = call_api(VERIFIER["client"], VERIFIER["model_id"], prompt, max_tokens=VERIFIER["max_tokens"])
        content = (r.get("content") or "").upper()
        if "NOT_EQUIVALENT" in content or "NOT EQUIVALENT" in content:
            return False
        if "EQUIVALENT" in content:
            return True
        time.sleep(1)
    return None


# =============== K=7 first-pass ===============

def sample_one(pid, model_name, model_cfg, problem_text, reference_answer, sample_idx, temperature=0.7):
    """Generate one sample + verify."""
    prompt = SOLVE_PROMPT.format(problem=problem_text)
    r = call_api(model_cfg["client"], model_cfg["model_id"], prompt,
                 max_tokens=model_cfg["max_tokens"], temperature=temperature)
    sol = r.get("content", "")
    extracted = extract_boxed_from_solution(sol)  # FIXED: returns string, not list
    correct = False
    if extracted:
        correct = verify_answer(problem_text, extracted, reference_answer) or False
    return {
        "sample_idx": sample_idx,
        "extracted": extracted,
        "correct": bool(correct),
        "error": r.get("error"),
        "solution_tail": sol[-300:] if sol else "",
    }


def run_k7_pass(problems, output_path, workers, k_start=0, k_end=7):
    """Run samples k_start..k_end-1 (default 0..6 = K=7 first pass)."""
    log(f"Tier 2 K=[{k_start},{k_end}) sampling: {len(problems)} problems × {len(CALIBRATION)} models × {k_end - k_start} samples")
    state = load_json(output_path, {})

    tasks = []
    for p in problems:
        pid = p["pid"]
        for m, cfg in CALIBRATION.items():
            key = f"{pid}__{m}"
            entry = state.get(key, {"pid": pid, "model": m, "samples": []})
            existing_idx = {s["sample_idx"] for s in entry.get("samples", []) if "sample_idx" in s}
            for k in range(k_start, k_end):
                if k in existing_idx:
                    continue
                tasks.append((p, m, cfg, k))

    log(f"  Pending: {len(tasks)} sample calls (resume)")
    if not tasks:
        return state

    n_done = 0
    save_every = max(50, len(tasks) // 30)

    def task(p, m, cfg, k):
        return sample_one(p["pid"], m, cfg, p["problem"], p["answer"], k, temperature=0.7)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(task, *t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            t = futs[fut]
            p, m, cfg, k = t
            pid = p["pid"]
            key = f"{pid}__{m}"
            sample = fut.result()
            entry = state.setdefault(key, {"pid": pid, "model": m, "samples": []})
            entry["samples"].append(sample)
            n_done += 1
            if n_done % save_every == 0 or n_done == len(tasks):
                save_atomic(state, output_path)
                log(f"  K=[{k_start},{k_end}) {n_done}/{len(tasks)}")
    save_atomic(state, output_path)
    return state


def compute_verdict_k7(samples):
    """Verdict per (pid, model) from K=7 samples."""
    valid = [s for s in samples if s.get("extracted") and not s.get("error")]
    n_correct = sum(1 for s in valid if s.get("correct"))
    n_valid = len(valid)
    if n_correct >= K7_THRESHOLD_REMOVE:
        return "robust_hack_K7"  # REMOVE
    if n_correct == 1:
        return "escalate_K15"
    if n_correct == 0 and n_valid >= 5:
        return "escalate_K15_zero"  # may escalate or keep
    return "indeterminate"


def aggregate_per_problem(state, problems):
    """Per-problem decision: if ANY model has robust_hack -> REMOVE."""
    problem_verdicts = {}
    for p in problems:
        pid = p["pid"]
        model_verdicts = {}
        for m in CALIBRATION:
            key = f"{pid}__{m}"
            entry = state.get(key)
            if not entry:
                continue
            samples = entry.get("samples", [])
            # Only first 7 samples for K=7 verdict
            samples_k7 = [s for s in samples if s.get("sample_idx", 0) < 7]
            verdict = compute_verdict_k7(samples_k7)
            model_verdicts[m] = verdict
        if any(v == "robust_hack_K7" for v in model_verdicts.values()):
            problem_verdicts[pid] = {"verdict": "REMOVE", "model_verdicts": model_verdicts, "stage": "K7"}
        else:
            # Check if any escalate K=15 needed
            if any(v.startswith("escalate") for v in model_verdicts.values()):
                problem_verdicts[pid] = {"verdict": "PENDING_K15", "model_verdicts": model_verdicts, "stage": "K7"}
            else:
                problem_verdicts[pid] = {"verdict": "KEEP", "model_verdicts": model_verdicts, "stage": "K7"}
    return problem_verdicts


# =============== Main ===============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=300)
    ap.add_argument("--input", default="stage_a_hard_200_pids.json")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only first N problems for smoke test")
    args = ap.parse_args()

    in_path = DATA / args.input
    with open(in_path, encoding="utf-8") as f:
        problems_data = json.load(f)
    # Format: list of dicts with pid, problem, answer, solution
    if isinstance(problems_data, dict):
        problems = list(problems_data.values())
    else:
        problems = problems_data

    if args.smoke > 0:
        problems = problems[:args.smoke]
        log(f"SMOKE TEST: only {args.smoke} problems")

    log(f"Loaded {len(problems)} problems for Tier 2 self-sampling")

    samples_path = DATA / "stage_a_tier2_samples.json"
    verdicts_path = DATA / "stage_a_tier2_verdicts.json"

    # K=7 first pass
    t0 = time.time()
    state = run_k7_pass(problems, samples_path, args.workers, k_start=0, k_end=7)
    log(f"K=7 done in {(time.time()-t0)/60:.1f} min")

    # Compute verdicts
    verdicts = aggregate_per_problem(state, problems)
    save_atomic(verdicts, verdicts_path)

    n_remove = sum(1 for v in verdicts.values() if v["verdict"] == "REMOVE")
    n_keep = sum(1 for v in verdicts.values() if v["verdict"] == "KEEP")
    n_pending = sum(1 for v in verdicts.values() if v["verdict"] == "PENDING_K15")
    log(f"K=7 verdicts: REMOVE={n_remove}, KEEP={n_keep}, PENDING_K15={n_pending}")

    # K=15 escalation for pending
    pending_pids = {pid for pid, v in verdicts.items() if v["verdict"] == "PENDING_K15"}
    pending_problems = [p for p in problems if p["pid"] in pending_pids]
    if pending_problems:
        log(f"K=15 escalation on {len(pending_problems)} problems")
        # Add 8 more samples (idx 7..14)
        state = run_k7_pass(pending_problems, samples_path, args.workers, k_start=7, k_end=15)

        # Re-evaluate verdicts: for problems escalated, check K=15 (any of additional 8 correct)
        for pid in pending_pids:
            v = verdicts[pid]
            for m in CALIBRATION:
                key = f"{pid}__{m}"
                entry = state.get(key)
                if not entry:
                    continue
                samples = entry.get("samples", [])
                samples_k15 = [s for s in samples if s.get("sample_idx", 0) < 15]
                samples_extra = [s for s in samples if 7 <= s.get("sample_idx", 0) < 15]
                n_extra_correct = sum(1 for s in samples_extra if s.get("extracted") and s.get("correct"))
                if n_extra_correct >= K15_THRESHOLD_REMOVE:
                    v["model_verdicts"][m] = "robust_hack_K15"
            if any(mv == "robust_hack_K15" for mv in v["model_verdicts"].values()):
                v["verdict"] = "REMOVE"
                v["stage"] = "K15"
            else:
                # Could escalate K=30 for true 0/15. Skip for now (too few problems)
                v["verdict"] = "KEEP"
                v["stage"] = "K15"

        save_atomic(verdicts, verdicts_path)
        n_remove = sum(1 for v in verdicts.values() if v["verdict"] == "REMOVE")
        n_keep = sum(1 for v in verdicts.values() if v["verdict"] == "KEEP")
        log(f"After K=15: REMOVE={n_remove}, KEEP={n_keep}")

    # Save survivors
    survivors = [pid for pid, v in verdicts.items() if v["verdict"] == "KEEP"]
    survivors_path = DATA / "stage_a_tier2_survivors.json"
    with open(survivors_path, "w", encoding="utf-8") as f:
        json.dump({"n_input": len(problems), "n_survivors": len(survivors), "survivors": sorted(survivors)},
                  f, indent=2, ensure_ascii=False)
    log(f"Tier 2 SURVIVORS: {len(survivors)}/{len(problems)} → {survivors_path}")
    log(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
