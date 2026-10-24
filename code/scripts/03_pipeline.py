"""
Stage A pipeline (unified script: smoke 10 / stage_a 1000).

Pipeline specification:
  DROP: olympiad-difficulty pre-filter
  KEEP: 5-stage rubric guessability v2 (Gemini-3.1-Pro)
  KEEP: weak canary K=8 (gpt-3.5 + llama-3-8b + qwen-2.5-7b)
  KEEP: geometry remove (Gemini-3.1-Pro classifier)
  DEFER: target K=7/15/30 + strong-tier triple-judge

Stages:
  S1 inference        8 comparable-tier models on N problems
  S2 verification     GPT-5.4 verifies answer correctness (per model output)
  S3 static rubric    Gemini-3.1-Pro generates rubric per problem
  S4 grading          triple-judge GPT-5.4 + Gemini-3.1-Pro + Qwen3-235B grade each (problem, model)
  S5 rubric_v2        Gemini-3.1-Pro EASY/HARD label per problem
  S6 canary           K=8 direct-answer per problem per weak model
  S7 geometry         Gemini-3.1-Pro classifies geometry yes/no

Outputs to data/<run>_<stage>.json (resumable per-key).

Usage:
  python 03_pipeline.py --input smoke_10.json --run smoke --workers 32
  python 03_pipeline.py --input stage_a_1000.json --run stage_a --workers 300

Bug catalog applied:
  #1 lock + deepcopy before json.dump
  #4 OpenAI(timeout=300.0) explicit
  #6 max_tokens always at provider cap (64k for reasoning models)
  #11 APIConnectionError in retry whitelist
  #12 atomic writes (.tmp + os.replace)
"""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Reuse existing module: prompts + parsers from paper version (consistent with submission)
from pilot_run import ANSWER_VERIFY_PROMPT, SOLVE_PROMPT, extract_boxed_from_solution
from static_rubric import STATIC_RUBRIC_PROMPT, GRADE_WITH_STATIC_PROMPT, parse_score
from guessability_rubric_v2 import GUESSABILITY_RUBRIC_V2_PROMPT, parse_rubric_v2_response

load_dotenv(ROOT / ".env")

OUT_DIR = Path(__file__).parent.parent / "data"

_save_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============== Model configs ===============

# 8 comparable-tier targets (paper Tabla 2)
TARGETS = {
    "claude-haiku-4.5": {"client": "openrouter", "model_id": "anthropic/claude-haiku-4.5", "max_tokens": 64000},
    "claude-sonnet-4.5": {"client": "openrouter", "model_id": "anthropic/claude-sonnet-4.5", "max_tokens": 64000},
    "command-a": {"client": "openrouter", "model_id": "cohere/command-a", "max_tokens": 32000},
    "llama-3.1-8b": {"client": "openrouter", "model_id": "meta-llama/llama-3.1-8b-instruct", "max_tokens": 64000},
    "llama-3.3-70b": {"client": "openrouter", "model_id": "meta-llama/llama-3.3-70b-instruct", "max_tokens": 32000},
    "mistral-small-3.2-24b": {"client": "openrouter", "model_id": "mistralai/mistral-small-3.2-24b-instruct", "max_tokens": 32000},
    "nova-pro-v1": {"client": "openrouter", "model_id": "amazon/nova-pro-v1", "max_tokens": 32000},
    "qwen2.5-72b": {"client": "openrouter", "model_id": "qwen/qwen-2.5-72b-instruct", "max_tokens": 32000},
}

# 3 judges (mantener Fleiss kappa con paper)
JUDGES = {
    "gpt-5.4": {"client": "openai", "model_id": "gpt-5.4", "max_tokens": 64000},
    "gemini-3.1-pro": {"client": "openrouter", "model_id": "google/gemini-3.1-pro-preview", "max_tokens": 64000},
    "qwen3-235b": {"client": "openrouter", "model_id": "qwen/qwen3-235b-a22b", "max_tokens": 64000},
}

# Verification model (mantener consistency con paper)
VERIFIER = {"client": "openai", "model_id": "gpt-5.4", "max_tokens": 64000}

# 3 weak canaries (K=8 samples each)
CANARIES = {
    "gpt-3.5-turbo": {"client": "openai", "model_id": "gpt-3.5-turbo", "max_tokens": 4096},
    "llama-3-8b-instruct": {"client": "openrouter", "model_id": "meta-llama/llama-3-8b-instruct", "max_tokens": 32000},
    "qwen2.5-7b-instruct": {"client": "openrouter", "model_id": "qwen/qwen-2.5-7b-instruct", "max_tokens": 32000},
}
N_CANARY_SAMPLES = 8

# Rubric v2 + geometry classifier
RUBRIC_GEMINI = {"client": "openrouter", "model_id": "google/gemini-3.1-pro-preview", "max_tokens": 64000}
GEOMETRY_GEMINI = {"client": "openrouter", "model_id": "google/gemini-3.1-pro-preview", "max_tokens": 4000}

# Per-stage worker caps. Bug #50 mitigated via explicit httpx.Limits above.
# 2026-05-08 retest: Gemini-3.1-Pro WORKS at workers=300 with 100% success (50/50 in 128s, 23 calls/min).
# Original "Gemini cap=5" diagnosis was confounded by wifi outage + httpx pool poisoning, not a real cap.
WORKERS_GEMINI_CAP = 300    # validated 2026-05-08, no failures
WORKERS_OPENAI_CAP = 300    # gpt-5.4, gpt-3.5 (with httpx limits)
WORKERS_OPENROUTER_OTHER = 300  # qwen3, llama, claude, mistral, etc.


# =============== API clients (one shared, thread-safe) ===============
# Bug #50 fix: explicit httpx.Limits to prevent pool poisoning at high concurrency.
# Allows 300 workers safely.

_HTTP_LIMITS = httpx.Limits(
    max_connections=600,        # 2x workers
    max_keepalive_connections=300,
    keepalive_expiry=15.0,
)
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
    """Unified API call with retry on rate limit, timeout, connection error. (bug #4 + #11)
    max_retries=5 with backoff [10, 20, 30, 60, 120]s = up to 4 min total per call.
    Robust to wifi outages of up to ~3-4 min.
    """
    client = openai_client if client_name == "openai" else openrouter_client
    backoff_schedule = [10, 20, 30, 60, 120]  # seconds
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if "gpt-5.4" in model_id or "gpt-5" in model_id:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            # If content empty but reasoning available (Gemini behavior), keep marker
            if not content:
                reasoning = getattr(msg, "reasoning", None) or ""
                if reasoning and len(reasoning) > 50:
                    # Use reasoning as fallback content (helps Gemini-style empty responses)
                    content = reasoning.strip()
            usage = resp.usage
            return {
                "content": content,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            err = str(e)
            err_l = err.lower()
            wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
            transient = (
                "429" in err or "rate" in err_l or
                "timeout" in err_l or "timed out" in err_l or
                "connection" in err_l or
                "503" in err or "502" in err or "504" in err
            )
            if transient and attempt < max_retries - 1:
                time.sleep(wait)
                continue
            return {"content": "", "error": f"{type(e).__name__}: {err[:200]}", "input_tokens": 0, "output_tokens": 0}
    return {"content": "", "error": f"exhausted {max_retries} retries", "input_tokens": 0, "output_tokens": 0}


# =============== Atomic JSON I/O (bug #1 + #12) ===============

def load_json(path, default=None):
    if Path(path).exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"  WARN: load failed {path}: {e}")
    return default if default is not None else {}


def save_atomic(data, path):
    """Atomic write: dump to .tmp then os.replace. Lock + deepcopy snapshot first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _save_lock:
        snapshot = copy.deepcopy(data)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# =============== Geometry classifier ===============

GEOMETRY_PROMPT = """Classify this competition math problem as GEOMETRY or NON-GEOMETRY.

GEOMETRY = primarily about: triangles, circles, polygons, angles, lengths, areas, points/lines in plane/space, coordinates of geometric figures.

NON-GEOMETRY = primarily about: number theory, algebra, combinatorics, sequences, functions, logic, probability without geometric setup.

Problem:
{problem}

Output exactly one word: GEOMETRY or NON-GEOMETRY"""


def classify_geometry(problem):
    resp = call_api(GEOMETRY_GEMINI["client"], GEOMETRY_GEMINI["model_id"],
                    GEOMETRY_PROMPT.format(problem=problem),
                    max_tokens=GEOMETRY_GEMINI["max_tokens"])
    content = (resp.get("content") or "").upper()
    if "NON-GEOMETRY" in content or "NON GEOMETRY" in content:
        return False
    if "GEOMETRY" in content:
        return True
    return None  # unclassified


# =============== Canary K=8 ===============

DIRECT_ANSWER_PROMPT = """Problem: {problem}

Provide ONLY the final answer in \\boxed{{}}. Do not show reasoning or work. Just the boxed answer.
"""


def verify_answer_async(problem, model_answer, reference, max_retries=2):
    if not model_answer:
        return False
    prompt = ANSWER_VERIFY_PROMPT.format(
        problem=problem, model_answer=model_answer, reference_answer=reference,
    )
    for attempt in range(max_retries):
        resp = call_api(VERIFIER["client"], VERIFIER["model_id"], prompt,
                        max_tokens=VERIFIER["max_tokens"])
        content = (resp.get("content") or "").upper()
        if "NOT_EQUIVALENT" in content or "NOT EQUIVALENT" in content:
            return False
        if "EQUIVALENT" in content:
            return True
        time.sleep(2)
    return None


def canary_one(problem, canary_name, cfg):
    prompt = DIRECT_ANSWER_PROMPT.format(problem=problem["problem"])
    samples = []
    for i in range(N_CANARY_SAMPLES):
        r = call_api(cfg["client"], cfg["model_id"], prompt,
                     max_tokens=cfg["max_tokens"], temperature=0.7)
        raw = r.get("content", "")
        # Bug fixed 2026-05-08: extract_boxed returns string, not list. Was extracted[-1] = last char!
        ans = extract_boxed_from_solution(raw)
        correct = False
        if ans:
            correct = verify_answer_async(problem["problem"], ans, problem["answer"]) or False
        samples.append({"i": i, "raw": raw[:300], "extracted": ans, "correct": bool(correct)})
    return {
        "canary": canary_name,
        "samples": samples,
        "any_correct": any(s["correct"] for s in samples),
        "n_correct": sum(1 for s in samples if s["correct"]),
    }


# =============== Stage runners ===============

def run_inference(problems, output_path, workers):
    """S1: 8 targets x N problems -> dict[(pid, model)] = {solution, extracted_answer}."""
    log(f"S1 inference: {len(problems)} problems x {len(TARGETS)} models = {len(problems)*len(TARGETS)} calls")
    state = load_json(output_path, {})

    def task(pid, problem_text, model, cfg):
        key = f"{pid}__{model}"
        if key in state and state[key].get("solution"):
            return None
        prompt = SOLVE_PROMPT.format(problem=problem_text)
        r = call_api(cfg["client"], cfg["model_id"], prompt, max_tokens=cfg["max_tokens"])
        sol = r.get("content", "")
        boxed = extract_boxed_from_solution(sol)
        # extract_boxed_from_solution returns the STRING content of last \boxed{} or None.
        # Bug fixed 2026-05-08: was boxed[-1] (last char of string!) causing 84% truncation.
        return key, {
            "pid": pid, "model": model,
            "solution": sol,
            "extracted_answer": boxed,
            "error": r.get("error"),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
        }

    tasks = [(p["post_neurips_id"], p["problem"], m, cfg) for p in problems for m, cfg in TARGETS.items()]
    # Retry empty solutions (resumable): include keys with empty/missing solution OR connection error
    todo = [t for t in tasks
            if f"{t[0]}__{t[2]}" not in state
            or not (state[f"{t[0]}__{t[2]}"].get("solution") or "").strip()]
    log(f"  {len(todo)}/{len(tasks)} pending (resume)")
    n_done = 0
    # S1 inference: 8 target models share load. With httpx limits, scale to user-specified workers (up to 300).
    eff_workers = min(workers, WORKERS_OPENAI_CAP)
    log(f"  workers={eff_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as ex:
        futs = [ex.submit(task, *t) for t in todo]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                key, entry = r
                state[key] = entry
            n_done += 1
            if n_done % max(50, len(todo) // 20) == 0 or n_done == len(todo):
                save_atomic(state, output_path)
                log(f"  S1 {n_done}/{len(todo)}")
    save_atomic(state, output_path)
    return state


def run_verification(problems_map, inference, output_path, workers):
    """S2: GPT-5.4 verifies answer correctness for each (pid, model)."""
    log(f"S2 verification: {len(inference)} pairs")
    state = load_json(output_path, {})

    def task(key, entry):
        if key in state and state[key].get("answer_correct") is not None:
            return None
        pid = entry["pid"]
        ma = entry.get("extracted_answer")
        if not ma:
            return key, {"pid": pid, "model": entry["model"], "answer_correct": False, "reason": "no_extracted_answer"}
        ref = problems_map[pid]["answer"]
        prompt = ANSWER_VERIFY_PROMPT.format(
            problem=problems_map[pid]["problem"], model_answer=ma, reference_answer=ref,
        )
        r = call_api(VERIFIER["client"], VERIFIER["model_id"], prompt, max_tokens=VERIFIER["max_tokens"])
        content = (r.get("content") or "").upper()
        if "NOT_EQUIVALENT" in content or "NOT EQUIVALENT" in content:
            ac = False
        elif "EQUIVALENT" in content:
            ac = True
        else:
            ac = None
        return key, {
            "pid": pid, "model": entry["model"], "answer_correct": ac,
            "model_answer": ma[:300], "reference_answer": ref, "raw": (r.get("content") or "")[:200],
        }

    todo = [(k, v) for k, v in inference.items() if k not in state or state[k].get("answer_correct") is None]
    log(f"  {len(todo)}/{len(inference)} pending")
    n_done = 0
    # S2 verification: GPT-5.4 only -> OpenAI cap 30
    eff_workers = min(workers, WORKERS_OPENAI_CAP)
    log(f"  workers={eff_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as ex:
        futs = [ex.submit(task, k, v) for k, v in todo]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                key, entry = r
                state[key] = entry
            n_done += 1
            if n_done % max(50, len(todo) // 20) == 0 or n_done == len(todo):
                save_atomic(state, output_path)
                log(f"  S2 {n_done}/{len(todo)}")
    save_atomic(state, output_path)
    return state


def run_static_rubrics(problems, output_path, workers):
    """S3: Gemini-3.1-Pro static rubric per problem."""
    log(f"S3 rubrics: {len(problems)} problems")
    state = load_json(output_path, {})

    def task(p):
        pid = p["post_neurips_id"]
        if pid in state and state[pid].get("rubric"):
            return None
        prompt = STATIC_RUBRIC_PROMPT.format(
            problem=p["problem"], answer=p["answer"], solution=(p.get("solution") or "")[:4000],
        )
        # rubric generated by Gemini-3.1-Pro (model-agnostic: independent of the target models)
        r = call_api(RUBRIC_GEMINI["client"], RUBRIC_GEMINI["model_id"], prompt,
                     max_tokens=RUBRIC_GEMINI["max_tokens"])
        return pid, {"pid": pid, "rubric": r.get("content", "")}

    todo = [p for p in problems if p["post_neurips_id"] not in state or not state[p["post_neurips_id"]].get("rubric", "").strip()]
    log(f"  {len(todo)}/{len(problems)} pending")
    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, WORKERS_GEMINI_CAP)) as ex:
        futs = [ex.submit(task, p) for p in todo]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                pid, entry = r
                state[pid] = entry
            n_done += 1
            if n_done % max(20, len(todo) // 20) == 0 or n_done == len(todo):
                save_atomic(state, output_path)
                log(f"  S3 {n_done}/{len(todo)}")
    save_atomic(state, output_path)
    return state


def run_grading(problems_map, inference, rubrics, output_path_template, workers):
    """S4: triple-judge grading, one file per judge."""
    state_per_judge = {}
    for judge_name, cfg in JUDGES.items():
        path = output_path_template.format(judge=judge_name)
        state = load_json(path, {})
        log(f"S4 {judge_name}: starting (existing entries: {len(state)})")
        def task(key, entry, judge_cfg=cfg, judge_name=judge_name):
            if key in state and state[key].get("score") is not None:
                return None
            pid = entry["pid"]
            if pid not in rubrics or not rubrics[pid].get("rubric"):
                return key, {"pid": pid, "model": entry["model"], "judge": judge_name,
                             "score": None, "reason": "no_rubric"}
            student_sol = entry.get("solution", "")
            prompt = GRADE_WITH_STATIC_PROMPT.format(
                problem=problems_map[pid]["problem"],
                answer=problems_map[pid]["answer"],
                rubric=rubrics[pid]["rubric"],
                student_solution=student_sol[:8000],
            )
            r = call_api(judge_cfg["client"], judge_cfg["model_id"], prompt, max_tokens=judge_cfg["max_tokens"])
            content = r.get("content", "")
            score = parse_score(content)
            return key, {
                "pid": pid, "model": entry["model"], "judge": judge_name,
                "score": score, "raw": content[:400],
            }

        todo = [(k, v) for k, v in inference.items() if k not in state or state[k].get("score") is None]
        log(f"  {judge_name}: {len(todo)} pending")
        n_done = 0
        # Per-judge worker cap: Gemini ≤5, OpenAI ≤30, Qwen3 ≤15
        if "gemini" in judge_name:
            eff_workers = min(workers, WORKERS_GEMINI_CAP)
        elif "gpt" in judge_name:
            eff_workers = min(workers, WORKERS_OPENAI_CAP)
        else:
            eff_workers = min(workers, WORKERS_OPENROUTER_OTHER)
        log(f"  {judge_name}: workers={eff_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as ex:
            futs = [ex.submit(task, k, v) for k, v in todo]
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                if r is not None:
                    key, entry = r
                    state[key] = entry
                n_done += 1
                if n_done % max(50, len(todo) // 20) == 0 or n_done == len(todo):
                    save_atomic(state, path)
                    log(f"  S4 {judge_name} {n_done}/{len(todo)}")
        save_atomic(state, path)
        state_per_judge[judge_name] = state
    return state_per_judge


def run_rubric_v2(problems, output_path, workers):
    """S5: Gemini-3.1-Pro EASY/HARD label."""
    log(f"S5 rubric v2 guessability: {len(problems)} problems")
    state = load_json(output_path, {})

    def task(p):
        pid = p["post_neurips_id"]
        if pid in state and state[pid].get("classification"):
            return None
        prompt = GUESSABILITY_RUBRIC_V2_PROMPT.format(
            problem=p["problem"], answer=p["answer"], solution=(p.get("solution") or "")[:3000],
        )
        r = call_api(RUBRIC_GEMINI["client"], RUBRIC_GEMINI["model_id"], prompt,
                     max_tokens=RUBRIC_GEMINI["max_tokens"])
        parsed = parse_rubric_v2_response(r.get("content", ""))
        return pid, {"pid": pid, "classification": parsed.get("classification"),
                     "primary_strategy": parsed.get("primary_strategy"),
                     "confidence": parsed.get("confidence"),
                     "raw": (r.get("content") or "")[:1000]}

    todo = [p for p in problems if p["post_neurips_id"] not in state or not state[p["post_neurips_id"]].get("classification")]
    log(f"  {len(todo)} pending")
    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, WORKERS_GEMINI_CAP)) as ex:
        futs = [ex.submit(task, p) for p in todo]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                pid, entry = r
                state[pid] = entry
            n_done += 1
            if n_done % max(20, len(todo) // 20) == 0 or n_done == len(todo):
                save_atomic(state, output_path)
                log(f"  S5 {n_done}/{len(todo)}")
    save_atomic(state, output_path)
    return state


def run_canary(problems, output_path, workers):
    """S6: K=8 direct-answer samples per problem per weak canary model."""
    log(f"S6 canary K={N_CANARY_SAMPLES} x {len(CANARIES)} models x {len(problems)} problems")
    state = load_json(output_path, {})

    def task(p, canary_name, cfg):
        pid = p["post_neurips_id"]
        key = f"{pid}__{canary_name}"
        if key in state and state[key].get("samples"):
            return None
        result = canary_one(p, canary_name, cfg)
        return key, {"pid": pid, **result}

    tasks = [(p, cn, cfg) for p in problems for cn, cfg in CANARIES.items()]
    todo = [t for t in tasks
            if f"{t[0]['post_neurips_id']}__{t[1]}" not in state
            or not state[f"{t[0]['post_neurips_id']}__{t[1]}"].get("samples")]
    log(f"  {len(todo)} pending")
    n_done = 0
    # Each canary task makes K=8 sequential calls + 8 GPT-5.4 verifications.
    # Cap at WORKERS_OPENAI_CAP because verifier is GPT-5.4 (the bottleneck).
    eff_workers = min(workers, WORKERS_OPENAI_CAP)
    log(f"  workers={eff_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as ex:
        futs = [ex.submit(task, *t) for t in todo]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                key, entry = r
                state[key] = entry
            n_done += 1
            if n_done % max(10, len(todo) // 20) == 0 or n_done == len(todo):
                save_atomic(state, output_path)
                log(f"  S6 {n_done}/{len(todo)}")
    save_atomic(state, output_path)
    return state


def run_geometry(problems, output_path, workers):
    """S7: Gemini geometry classifier."""
    log(f"S7 geometry: {len(problems)} problems")
    state = load_json(output_path, {})

    def task(p):
        pid = p["post_neurips_id"]
        if pid in state and state[pid].get("is_geometry") is not None:
            return None
        ig = classify_geometry(p["problem"])
        return pid, {"pid": pid, "is_geometry": ig}

    todo = [p for p in problems if p["post_neurips_id"] not in state or state[p["post_neurips_id"]].get("is_geometry") is None]
    log(f"  {len(todo)} pending")
    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, WORKERS_GEMINI_CAP)) as ex:
        futs = [ex.submit(task, p) for p in todo]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                pid, entry = r
                state[pid] = entry
            n_done += 1
            if n_done % max(10, len(todo) // 10) == 0 or n_done == len(todo):
                save_atomic(state, output_path)
                log(f"  S7 {n_done}/{len(todo)}")
    save_atomic(state, output_path)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="filename in data/, e.g. smoke_10.json")
    ap.add_argument("--run", required=True, help="run name prefix, e.g. smoke or stage_a")
    ap.add_argument("--workers", type=int, default=300)
    ap.add_argument("--stages", default="1,2,3,4,5,6,7", help="comma-separated stage numbers")
    args = ap.parse_args()

    in_path = OUT_DIR / args.input
    if not in_path.exists():
        log(f"ERROR: input {in_path} not found")
        sys.exit(1)

    with open(in_path, encoding="utf-8") as f:
        problems = json.load(f)
    problems_map = {p["post_neurips_id"]: p for p in problems}
    log(f"Loaded {len(problems)} problems from {args.input}")

    paths = {
        "inference": OUT_DIR / f"{args.run}_inference.json",
        "verification": OUT_DIR / f"{args.run}_verification.json",
        "rubrics": OUT_DIR / f"{args.run}_rubrics.json",
        "grading_template": str(OUT_DIR / f"{args.run}_grading_{{judge}}.json"),
        "rubric_v2": OUT_DIR / f"{args.run}_rubric_v2.json",
        "canary": OUT_DIR / f"{args.run}_canary.json",
        "geometry": OUT_DIR / f"{args.run}_geometry.json",
    }

    stages = [int(s) for s in args.stages.split(",")]
    t0 = time.time()

    inference = {}
    if 1 in stages:
        inference = run_inference(problems, paths["inference"], args.workers)
    else:
        inference = load_json(paths["inference"], {})

    if 2 in stages:
        run_verification(problems_map, inference, paths["verification"], args.workers)

    rubrics = {}
    if 3 in stages:
        rubrics = run_static_rubrics(problems, paths["rubrics"], args.workers)
    else:
        rubrics = load_json(paths["rubrics"], {})

    if 4 in stages and inference and rubrics:
        run_grading(problems_map, inference, rubrics, paths["grading_template"], args.workers)

    if 5 in stages:
        run_rubric_v2(problems, paths["rubric_v2"], args.workers)

    if 6 in stages:
        run_canary(problems, paths["canary"], args.workers)

    if 7 in stages:
        run_geometry(problems, paths["geometry"], args.workers)

    elapsed = time.time() - t0
    log(f"PIPELINE DONE in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
