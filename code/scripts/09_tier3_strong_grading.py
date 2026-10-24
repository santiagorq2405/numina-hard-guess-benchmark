"""
Three-tier ablation, Tier 3: strong-tier solution grading (K = 3 candidates + triple-judge).

Detects "cheat instances": answer correct but solution low-rigor.

Steps (three-tier ablation design):
  1. claude-sonnet-4.6 generates K=3 candidate solutions
  2. Verify each candidate's answer
  3. For each correct candidate, grade with triple-judge (mean of 3)
  4. Per-problem aggregation: mean of per-candidate consensus scores across CORRECT candidates
  5. If mean > 6.5 → "rigorous_solver" (KEEP); else → "cheat_instance" (REMOVE)

Input: data/stage_a_tier2_survivors.json (PIDs that passed Tier 2)
Output: data/stage_a_tier3_results.json
        data/stage_a_tier3_survivors.json (PIDs that passed Tier 2 + Tier 3)

Usage:
  python 09_tier3_strong_grading.py --workers 300
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
import re

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pilot_run import ANSWER_VERIFY_PROMPT, SOLVE_PROMPT, extract_boxed_from_solution
from static_rubric import STATIC_RUBRIC_PROMPT, GRADE_WITH_STATIC_PROMPT, parse_score

load_dotenv(ROOT / ".env")

DATA = Path(__file__).parent.parent / "data"

_save_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============== Models ===============

STRONG_MODEL = {"client": "openrouter", "model_id": "anthropic/claude-sonnet-4.6", "max_tokens": 64000}
K_STRONG = 3
SAMPLE_TEMP = 0.7
SOL_THRESHOLD = 6.5

JUDGES = {
    "gpt-5.4": {"client": "openai", "model_id": "gpt-5.4", "max_tokens": 64000},
    "gemini-3.1-pro": {"client": "openrouter", "model_id": "google/gemini-3.1-pro-preview", "max_tokens": 64000},
    "qwen3-235b": {"client": "openrouter", "model_id": "qwen/qwen3-235b-a22b", "max_tokens": 64000},
}

VERIFIER = {"client": "openai", "model_id": "gpt-5.4", "max_tokens": 64000}

RUBRIC_GEMINI = {"client": "openrouter", "model_id": "google/gemini-3.1-pro-preview", "max_tokens": 64000}


# =============== API ===============

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


# =============== Sample + grade ===============

def generate_strong_sample(p, sample_idx):
    """Generate one Claude-Sonnet-4.6 sample for problem p."""
    prompt = SOLVE_PROMPT.format(problem=p["problem"])
    r = call_api(STRONG_MODEL["client"], STRONG_MODEL["model_id"], prompt,
                 max_tokens=STRONG_MODEL["max_tokens"], temperature=SAMPLE_TEMP)
    sol = r.get("content", "")
    extracted = extract_boxed_from_solution(sol)
    correct = False
    if extracted:
        correct = verify_answer(p["problem"], extracted, p["answer"]) or False
    return {
        "sample_idx": sample_idx,
        "solution": sol,
        "extracted_answer": extracted,
        "answer_correct": bool(correct),
        "error": r.get("error"),
    }


def grade_sample_triple(p, sample, rubric):
    """Triple-judge grade of a strong-tier sample's solution."""
    if not sample.get("answer_correct"):
        return {}
    if not sample.get("solution"):
        return {}
    scores = {}
    for j_name, cfg in JUDGES.items():
        prompt = GRADE_WITH_STATIC_PROMPT.format(
            problem=p["problem"], answer=p["answer"], rubric=rubric,
            student_solution=sample["solution"][:8000],
        )
        r = call_api(cfg["client"], cfg["model_id"], prompt, max_tokens=cfg["max_tokens"])
        content = r.get("content", "")
        score = parse_score(content)
        scores[j_name] = score
    return scores


def generate_rubric(p):
    """Static rubric per paper, used for grading strong-tier samples."""
    prompt = STATIC_RUBRIC_PROMPT.format(
        problem=p["problem"], answer=p["answer"], solution=(p.get("solution") or "")[:4000],
    )
    r = call_api(RUBRIC_GEMINI["client"], RUBRIC_GEMINI["model_id"], prompt, max_tokens=RUBRIC_GEMINI["max_tokens"])
    return r.get("content", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=300)
    ap.add_argument("--input-survivors", default="stage_a_tier2_survivors.json")
    ap.add_argument("--input-problems", default="stage_a_hard_200_pids.json")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    # Load survivors
    surv = load_json(DATA / args.input_survivors)
    surv_pids = set(surv.get("survivors", []))
    log(f"Tier 2 survivors: {len(surv_pids)}")

    # Load problem details
    prob_data = load_json(DATA / args.input_problems)
    if isinstance(prob_data, dict):
        all_problems = list(prob_data.values())
    else:
        all_problems = prob_data

    problems = [p for p in all_problems if p["pid"] in surv_pids]
    log(f"Problems to grade with Tier 3: {len(problems)}")

    if args.smoke > 0:
        problems = problems[:args.smoke]
        log(f"SMOKE: only {args.smoke} problems")

    samples_path = DATA / "stage_a_tier3_samples.json"
    rubrics_path = DATA / "stage_a_tier3_rubrics.json"
    grading_path = DATA / "stage_a_tier3_grading.json"
    final_path = DATA / "stage_a_tier3_verdicts.json"

    # Step 1: K=3 strong-tier samples
    t0 = time.time()
    samples_state = load_json(samples_path, {})
    log(f"Step 1: K={K_STRONG} samples per problem (strong-tier)")
    tasks = []
    for p in problems:
        pid = p["pid"]
        existing = samples_state.get(pid, {"pid": pid, "samples": []})
        existing_idx = {s["sample_idx"] for s in existing.get("samples", []) if "sample_idx" in s}
        for k in range(K_STRONG):
            if k not in existing_idx:
                tasks.append((p, k))

    log(f"  Pending: {len(tasks)} sample calls")
    n_done = 0
    save_every = max(20, len(tasks) // 20)
    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(lambda p_=t[0], k_=t[1]: generate_strong_sample(p_, k_)): t for t in tasks}
            for fut in concurrent.futures.as_completed(futs):
                t = futs[fut]
                p, k = t
                pid = p["pid"]
                sample = fut.result()
                entry = samples_state.setdefault(pid, {"pid": pid, "samples": []})
                entry["samples"].append(sample)
                n_done += 1
                if n_done % save_every == 0 or n_done == len(tasks):
                    save_atomic(samples_state, samples_path)
                    log(f"  K=3 samples {n_done}/{len(tasks)}")
        save_atomic(samples_state, samples_path)
    log(f"Step 1 done in {(time.time()-t0)/60:.1f} min")

    # Step 2: rubrics
    t1 = time.time()
    rubrics_state = load_json(rubrics_path, {})
    rubric_pending = [p for p in problems if p["pid"] not in rubrics_state or not rubrics_state[p["pid"]].get("rubric")]
    log(f"Step 2: rubrics {len(rubric_pending)} pending")
    if rubric_pending:
        n_done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(generate_rubric, p): p for p in rubric_pending}
            for fut in concurrent.futures.as_completed(futs):
                p = futs[fut]
                rubric = fut.result()
                rubrics_state[p["pid"]] = {"pid": p["pid"], "rubric": rubric}
                n_done += 1
                if n_done % 20 == 0 or n_done == len(rubric_pending):
                    save_atomic(rubrics_state, rubrics_path)
        save_atomic(rubrics_state, rubrics_path)
    log(f"Step 2 done in {(time.time()-t1)/60:.1f} min")

    # Step 3: triple-judge grading on correct samples
    t2 = time.time()
    grading_state = load_json(grading_path, {})
    log(f"Step 3: triple-judge grade correct samples")
    grade_tasks = []
    for p in problems:
        pid = p["pid"]
        rubric = rubrics_state.get(pid, {}).get("rubric", "")
        if not rubric:
            continue
        samples = samples_state.get(pid, {}).get("samples", [])
        for s in samples:
            if not s.get("answer_correct"):
                continue
            key = f"{pid}__sample_{s['sample_idx']}"
            if key in grading_state and grading_state[key].get("scores"):
                continue
            grade_tasks.append((p, s, rubric, key))

    log(f"  Pending grading: {len(grade_tasks)} (only correct samples)")
    n_done = 0
    save_every = max(10, len(grade_tasks) // 20)
    if grade_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            def gtask(p_, s_, rub_, key_):
                scores = grade_sample_triple(p_, s_, rub_)
                return key_, p_["pid"], s_["sample_idx"], scores
            futs = [ex.submit(gtask, *t) for t in grade_tasks]
            for fut in concurrent.futures.as_completed(futs):
                key, pid, sidx, scores = fut.result()
                grading_state[key] = {"pid": pid, "sample_idx": sidx, "scores": scores}
                n_done += 1
                if n_done % save_every == 0 or n_done == len(grade_tasks):
                    save_atomic(grading_state, grading_path)
        save_atomic(grading_state, grading_path)
    log(f"Step 3 done in {(time.time()-t2)/60:.1f} min")

    # Step 4: per-problem verdict
    log(f"Step 4: aggregate verdicts")
    verdicts = {}
    for p in problems:
        pid = p["pid"]
        samples = samples_state.get(pid, {}).get("samples", [])
        correct_samples = [s for s in samples if s.get("answer_correct")]
        if not correct_samples:
            # No correct sample → can't decide via Tier 3 → KEEP (no evidence of cheat)
            verdicts[pid] = {"verdict": "KEEP_NO_CORRECT", "n_correct": 0, "mean_rigor": None}
            continue
        per_sample_means = []
        for s in correct_samples:
            key = f"{pid}__sample_{s['sample_idx']}"
            ge = grading_state.get(key)
            if not ge:
                continue
            scores = [v for v in ge["scores"].values() if v is not None]
            if len(scores) >= 2:
                per_sample_means.append(sum(scores) / len(scores))
        if not per_sample_means:
            verdicts[pid] = {"verdict": "KEEP_NO_GRADES", "n_correct": len(correct_samples), "mean_rigor": None}
            continue
        mean_rigor = sum(per_sample_means) / len(per_sample_means)
        if mean_rigor > SOL_THRESHOLD:
            verdicts[pid] = {"verdict": "KEEP_RIGOROUS", "n_correct": len(correct_samples), "mean_rigor": mean_rigor}
        else:
            verdicts[pid] = {"verdict": "REMOVE_CHEAT", "n_correct": len(correct_samples), "mean_rigor": mean_rigor}

    save_atomic(verdicts, final_path)
    n_remove = sum(1 for v in verdicts.values() if v["verdict"] == "REMOVE_CHEAT")
    n_keep = sum(1 for v in verdicts.values() if v["verdict"].startswith("KEEP"))
    log(f"Tier 3 verdicts: REMOVE_CHEAT={n_remove}, KEEP={n_keep}")

    # Survivors of Tier 2 + Tier 3
    survivors_t3 = [pid for pid, v in verdicts.items() if v["verdict"].startswith("KEEP")]
    surv_path_final = DATA / "stage_a_tier3_survivors.json"
    with open(surv_path_final, "w", encoding="utf-8") as f:
        json.dump({"n_input": len(problems), "n_survivors": len(survivors_t3), "survivors": sorted(survivors_t3)},
                  f, indent=2, ensure_ascii=False)
    log(f"Tier 3 SURVIVORS: {len(survivors_t3)}/{len(problems)} → {surv_path_final}")
    log(f"Total Tier 3 time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
