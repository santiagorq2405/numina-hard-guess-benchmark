"""
PILOT: GuessProof-Bench GO/NO-GO Test
100 problems x 5 models x dual-judge guessability x dual-judge grading

Usage:
  python pilot_run.py --step 1   # Guessability labeling (GPT-5.4 + Gemini)
  python pilot_run.py --step 2   # Inference 5 models (all parallel)
  python pilot_run.py --step 3   # Answer verification + grading
  python pilot_run.py --step 4   # Analysis + GO/NO-GO decision
  python pilot_run.py --step all # Run everything sequentially
"""

import json
import os
import sys
import time
import argparse
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
from guessability_rubric import GUESSABILITY_PROMPT, parse_guessability_response

load_dotenv()

# API clients with explicit timeout to prevent indefinite hangs
openai_client = OpenAI(timeout=300.0)  # Uses OPENAI_API_KEY from .env, 5 min per call
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    timeout=300.0,  # 5 min per call, prevents indefinite hangs on OpenRouter stalls
)

# Model configs for the 5 pilot models
PILOT_MODELS = {
    "gpt-5.4": {
        "client": "openai",
        "model_id": "gpt-5.4",
        "max_completion_tokens": 16000,
    },
    "gemma-4-31b": {
        "client": "openrouter",
        "model_id": "google/gemma-4-31b-it",
        "max_completion_tokens": 16000,
    },
    "qwen-3.6-plus": {
        "client": "openrouter",
        "model_id": "qwen/qwen3.6-plus",
        "max_completion_tokens": 16000,
    },
    "llama-4-maverick": {
        "client": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
        "max_completion_tokens": 16000,
    },
    "qwen3-14b": {
        "client": "openrouter",
        "model_id": "qwen/qwen3-14b",
        "max_completion_tokens": 16000,
    },
}

# Judge models for guessability
GUESSABILITY_JUDGES = {
    "gpt-5.4": {"client": "openai", "model_id": "gpt-5.4"},
    "gemini-3.1-pro": {"client": "openrouter", "model_id": "google/gemini-3.1-pro-preview"},
}

SOLVE_PROMPT = """Solve the following mathematical competition problem. Show your complete solution with full justification. Put your final answer in \\boxed{{}}.

Problem:
{problem}

Important: You MUST provide your final answer inside \\boxed{{}}."""

ANSWER_VERIFY_PROMPT = """You are a mathematical answer checker. Given a problem, a model's answer, and the reference answer, determine if they are mathematically equivalent.

Problem: {problem}

Model's answer: {model_answer}
Reference answer: {reference_answer}

Are these answers mathematically equivalent? Consider different notations, orderings, and representations.

Answer ONLY with: EQUIVALENT or NOT_EQUIVALENT"""

GRADING_RUBRIC_PROMPT = """You are an expert mathematical judge. Generate a grading rubric for this olympiad problem based on the reference solution.

Problem:
{problem}

Reference solution:
{solution}

Reference answer: {answer}

Create a rubric on the IMO 0-7 scale:
- 0: No meaningful progress
- 1-2: Some relevant observations but no real progress
- 3-4: Significant progress, key ideas partially present
- 5-6: Nearly complete solution with minor gaps
- 7: Complete, rigorous solution

Output the rubric as a list of criteria for each score range."""

GRADING_PROMPT = """You are an expert mathematical judge. Grade this solution using the rubric below.

Problem:
{problem}

Reference solution (canonical, for comparison):
{reference_solution}

Reference answer: {answer}

Rubric:
{rubric}

Student solution to grade:
{student_solution}

Compare the student solution against the reference solution. Identify any arithmetic mistakes, missing logical steps, unjustified assumptions, or gaps relative to the canonical reasoning. Then assign a score using the rubric.

Output format:
SCORE: [0-7]
JUSTIFICATION: [2-3 sentences citing specific differences against the reference]"""


def load_pilot_problems():
    """Load the 100 pilot problems."""
    with open("data/pilot_100_problems.json", "r", encoding="utf-8") as f:
        return json.load(f)


def call_api(client_name, model_id, prompt, max_tokens=16000, temperature=0.0, max_retries=3):
    """Unified API call for both OpenAI and OpenRouter, with retry on rate limits."""
    client = openai_client if client_name == "openai" else openrouter_client

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            # GPT-5.4 uses max_completion_tokens (internal thinking)
            if "gpt-5.4" in model_id:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = msg.content or ""
            if not content:
                reasoning = getattr(msg, 'reasoning', None) or ""
                if reasoning:
                    content = "(reasoning only, no final content)"
            usage = response.usage
            return {
                "content": content,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait = (attempt + 1) * 10
                print(f"    Rate limited on {model_id}, waiting {wait}s (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait)
            elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
                print(f"    Timeout on {model_id} attempt {attempt+1}/{max_retries}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return {"content": "", "error": f"timeout after {max_retries} attempts", "input_tokens": 0, "output_tokens": 0}
            elif "connection" in error_str.lower():
                # APIConnectionError: transient network/TCP issue, retry with backoff
                wait = (attempt + 1) * 10
                print(f"    Connection error on {model_id} attempt {attempt+1}/{max_retries}, waiting {wait}s", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    return {"content": "", "error": f"connection error after {max_retries} retries", "input_tokens": 0, "output_tokens": 0}
            else:
                return {"content": "", "error": error_str, "input_tokens": 0, "output_tokens": 0}

    return {"content": "", "error": f"Rate limited after {max_retries} retries", "input_tokens": 0, "output_tokens": 0}


def save_incremental(data, filepath):
    """Save data incrementally to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# STEP 1: Guessability Labeling
# ============================================================
def step1_guessability(problems):
    """Label guessability with dual judges (GPT-5.4 + Gemini)."""
    print("\n" + "=" * 60)
    print("STEP 1: Guessability Labeling (dual judge)")
    print("=" * 60)

    results = []
    output_path = "data/pilot_guessability.json"

    for i, prob in enumerate(problems):
        pid = prob["pilot_id"]
        print(f"\n[{i+1}/100] {pid} (answer: {prob['answer'][:40]})")

        prob_result = {"pilot_id": pid, "answer": prob["answer"], "answer_type": prob["answer_type"]}

        for judge_name, judge_config in GUESSABILITY_JUDGES.items():
            prompt = GUESSABILITY_PROMPT.format(
                problem=prob["problem"],
                answer=prob["answer"],
                problem_type=prob.get("problem_type", ""),
                question_type=prob.get("question_type", "")
            )

            resp = call_api(judge_config["client"], judge_config["model_id"], prompt, max_tokens=2000)
            parsed = parse_guessability_response(resp["content"])

            prob_result[f"guessability_{judge_name}"] = {
                "score": parsed["score"],
                "reasoning": parsed["reasoning"],
                "tokens": resp["input_tokens"] + resp["output_tokens"],
            }
            print(f"  {judge_name}: score={parsed['score']}")

        results.append(prob_result)
        save_incremental(results, output_path)

    print(f"\nGuessability labels saved to {output_path}")
    return results


# ============================================================
# STEP 2: Model Inference (5 models in parallel)
# ============================================================
def run_single_model(model_name, model_config, problems, output_path):
    """Run a single model on all problems and save results."""
    print(f"\n  Starting {model_name}...")
    results = []

    for i, prob in enumerate(problems):
        prompt = SOLVE_PROMPT.format(problem=prob["problem"])
        resp = call_api(model_config["client"], model_config["model_id"], prompt,
                       max_tokens=model_config["max_completion_tokens"])

        results.append({
            "pilot_id": prob["pilot_id"],
            "model": model_name,
            "solution": resp["content"],
            "error": resp.get("error"),
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        })

        if (i + 1) % 10 == 0:
            print(f"    {model_name}: {i+1}/100 done")
            save_incremental(results, output_path)

    save_incremental(results, output_path)
    print(f"  {model_name}: COMPLETE ({len(results)} solutions)")
    return results


def step2_inference(problems):
    """Run all 5 models in parallel."""
    print("\n" + "=" * 60)
    print("STEP 2: Model Inference (5 models in parallel)")
    print("=" * 60)

    all_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for model_name, model_config in PILOT_MODELS.items():
            output_path = f"data/pilot_solutions_{model_name}.json"
            future = executor.submit(run_single_model, model_name, model_config, problems, output_path)
            futures[future] = model_name

        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                results = future.result()
                all_results[model_name] = results
                print(f"\n  {model_name}: finished with {len(results)} solutions")
            except Exception as e:
                print(f"\n  {model_name}: FAILED - {e}")

    return all_results


# ============================================================
# STEP 3: Answer Verification + Grading
# ============================================================
def extract_boxed_from_solution(solution_text):
    """Extract the last \\boxed{...} from a solution."""
    results = []
    i = 0
    while i < len(solution_text):
        idx = solution_text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + 7
        for j in range(start, len(solution_text)):
            if solution_text[j] == '{':
                depth += 1
            elif solution_text[j] == '}':
                if depth == 0:
                    results.append(solution_text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            i = idx + 1
            continue
        continue
    return results[-1].strip() if results else None


def step3_verify_and_grade(problems):
    """Verify answers and grade solutions."""
    print("\n" + "=" * 60)
    print("STEP 3: Answer Verification + Grading")
    print("=" * 60)

    # Load all solutions
    all_solutions = []
    for model_name in PILOT_MODELS:
        filepath = f"data/pilot_solutions_{model_name}.json"
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                solutions = json.load(f)
            all_solutions.extend(solutions)

    print(f"Loaded {len(all_solutions)} solutions from {len(PILOT_MODELS)} models")

    # Build problem lookup
    prob_lookup = {p["pilot_id"]: p for p in problems}

    results = []
    output_path = "data/pilot_verification_grading.json"

    for i, sol in enumerate(all_solutions):
        pid = sol["pilot_id"]
        prob = prob_lookup[pid]
        model_answer = extract_boxed_from_solution(sol.get("solution", ""))

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(all_solutions)}")

        # Answer verification with LLM
        if model_answer:
            verify_prompt = ANSWER_VERIFY_PROMPT.format(
                problem=prob["problem"][:500],
                model_answer=model_answer,
                reference_answer=prob["answer"]
            )
            verify_resp = call_api("openai", "gpt-5.4", verify_prompt, max_tokens=500)
            verify_text = verify_resp["content"].upper().strip()
            if "NOT_EQUIVALENT" in verify_text or "NOT EQUIVALENT" in verify_text:
                answer_correct = False
            elif "EQUIVALENT" in verify_text:
                answer_correct = True
            else:
                answer_correct = False
        else:
            answer_correct = False

        # Generate rubric (one per problem, cache it)
        rubric_key = f"rubric_{pid}"
        if not hasattr(step3_verify_and_grade, '_rubric_cache'):
            step3_verify_and_grade._rubric_cache = {}

        if pid not in step3_verify_and_grade._rubric_cache:
            rubric_prompt = GRADING_RUBRIC_PROMPT.format(
                problem=prob["problem"][:1000],
                solution=prob["solution"][:1500],
                answer=prob["answer"]
            )
            rubric_resp = call_api("openai", "gpt-5.4", rubric_prompt, max_tokens=2000)
            step3_verify_and_grade._rubric_cache[pid] = rubric_resp["content"]

        rubric = step3_verify_and_grade._rubric_cache[pid]

        # Grade solution
        grade_prompt = GRADING_PROMPT.format(
            problem=prob["problem"][:1000],
            answer=prob["answer"],
            rubric=rubric[:1500],
            student_solution=sol["solution"][:3000] if sol["solution"] else "(No solution produced)"
        )
        grade_resp = call_api("openai", "gpt-5.4", grade_prompt, max_tokens=1000)

        # Parse score
        score = None
        for line in grade_resp["content"].split("\n"):
            if line.strip().startswith("SCORE:"):
                try:
                    score = int(line.split(":")[1].strip().split("/")[0].strip())
                except (ValueError, IndexError):
                    pass

        results.append({
            "pilot_id": pid,
            "model": sol["model"],
            "model_answer": model_answer,
            "reference_answer": prob["answer"],
            "answer_correct": answer_correct,
            "solution_score": score,
            "answer_type": prob["answer_type"],
            "stratification_group": prob.get("stratification_group", ""),
        })

        if (i + 1) % 25 == 0:
            save_incremental(results, output_path)

    save_incremental(results, output_path)
    print(f"\nVerification + grading saved to {output_path}")
    return results


# ============================================================
# STEP 4: Analysis + GO/NO-GO
# ============================================================
def step4_analysis():
    """Analyze results and make GO/NO-GO decision."""
    print("\n" + "=" * 60)
    print("STEP 4: ANALYSIS + GO/NO-GO DECISION")
    print("=" * 60)

    # Load guessability labels
    with open("data/pilot_guessability.json", "r", encoding="utf-8") as f:
        guessability = json.load(f)

    # Load verification + grading
    with open("data/pilot_verification_grading.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    # Build guessability lookup (use GPT-5.4 scores)
    guess_scores = {}
    for g in guessability:
        pid = g["pilot_id"]
        gpt_score = g.get("guessability_gpt-5.4", {}).get("score")
        gemini_score = g.get("guessability_gemini-3.1-pro", {}).get("score")
        guess_scores[pid] = {
            "gpt": gpt_score,
            "gemini": gemini_score,
            "avg": (gpt_score + gemini_score) / 2 if gpt_score and gemini_score else gpt_score
        }

    # Split into guessable vs non-guessable (using GPT-5.4 score)
    guessable_pids = [pid for pid, gs in guess_scores.items() if gs["gpt"] and gs["gpt"] >= 4]
    non_guessable_pids = [pid for pid, gs in guess_scores.items() if gs["gpt"] and gs["gpt"] <= 2]
    borderline_pids = [pid for pid, gs in guess_scores.items() if gs["gpt"] and gs["gpt"] == 3]

    print(f"\nGuessability split (GPT-5.4 judge):")
    print(f"  Guessable (score 4-5): {len(guessable_pids)} problems")
    print(f"  Non-guessable (score 1-2): {len(non_guessable_pids)} problems")
    print(f"  Borderline (score 3): {len(borderline_pids)} problems")

    # Calculate FPR (False Positive Rate: answer correct but solution < threshold)
    for threshold in [2, 3, 4]:
        print(f"\n--- FPR at solution_score < {threshold} ---")

        for group_name, group_pids in [("Guessable", guessable_pids), ("Non-guessable", non_guessable_pids)]:
            group_results = [r for r in results if r["pilot_id"] in group_pids]
            n_total = len(group_results)
            n_answer_correct = sum(1 for r in group_results if r["answer_correct"])
            n_fp = sum(1 for r in group_results if r["answer_correct"] and r.get("solution_score") is not None and r["solution_score"] < threshold)
            n_correct_with_score = sum(1 for r in group_results if r["answer_correct"] and r.get("solution_score") is not None)

            fpr = (n_fp / n_correct_with_score * 100) if n_correct_with_score > 0 else 0
            acc = (n_answer_correct / n_total * 100) if n_total > 0 else 0

            print(f"  {group_name}: {n_total} pairs, answer_acc={acc:.1f}%, FP={n_fp}/{n_correct_with_score}, FPR={fpr:.1f}%")

    # Cohen's kappa for guessability labels
    print(f"\n--- Cohen's Kappa (Guessability Labels) ---")
    gpt_binary = []
    gemini_binary = []
    for g in guessability:
        gpt_s = g.get("guessability_gpt-5.4", {}).get("score")
        gem_s = g.get("guessability_gemini-3.1-pro", {}).get("score")
        if gpt_s and gem_s:
            gpt_binary.append(1 if gpt_s >= 3 else 0)
            gemini_binary.append(1 if gem_s >= 3 else 0)

    if gpt_binary:
        # Simple kappa calculation
        n = len(gpt_binary)
        agreement = sum(1 for a, b in zip(gpt_binary, gemini_binary) if a == b)
        p_o = agreement / n
        p_gpt = sum(gpt_binary) / n
        p_gem = sum(gemini_binary) / n
        p_e = p_gpt * p_gem + (1 - p_gpt) * (1 - p_gem)
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
        print(f"  Agreement: {agreement}/{n} = {p_o:.3f}")
        print(f"  Kappa: {kappa:.3f}")
    else:
        kappa = 0
        print(f"  No valid pairs for kappa calculation")

    # Per-model breakdown
    print(f"\n--- Per-Model Answer Accuracy ---")
    models = set(r["model"] for r in results)
    for model in sorted(models):
        model_results = [r for r in results if r["model"] == model]
        acc = sum(1 for r in model_results if r["answer_correct"]) / len(model_results) * 100
        avg_score = sum(r["solution_score"] for r in model_results if r["solution_score"] is not None) / max(1, sum(1 for r in model_results if r["solution_score"] is not None))
        print(f"  {model}: answer_acc={acc:.1f}%, avg_solution_score={avg_score:.2f}/7")

    # Per answer-type breakdown
    print(f"\n--- Per Answer-Type FPR (threshold < 4) ---")
    for atype in ["small_integer", "algebraic_expression", "function_expression", "medium_large_other"]:
        type_results = [r for r in results if r.get("stratification_group") == atype]
        n_correct = sum(1 for r in type_results if r["answer_correct"])
        n_fp = sum(1 for r in type_results if r["answer_correct"] and r.get("solution_score") is not None and r["solution_score"] < 4)
        fpr = (n_fp / n_correct * 100) if n_correct > 0 else 0
        print(f"  {atype}: {n_correct} correct answers, FPR={fpr:.1f}%")

    # GO/NO-GO DECISION
    print(f"\n{'='*60}")
    print("GO/NO-GO DECISION")
    print(f"{'='*60}")

    # Get FPR at threshold 4 for both groups
    g_results = [r for r in results if r["pilot_id"] in guessable_pids]
    ng_results = [r for r in results if r["pilot_id"] in non_guessable_pids]

    g_correct = [r for r in g_results if r["answer_correct"] and r.get("solution_score") is not None]
    ng_correct = [r for r in ng_results if r["answer_correct"] and r.get("solution_score") is not None]

    fpr_g = sum(1 for r in g_correct if r["solution_score"] < 4) / max(1, len(g_correct)) * 100
    fpr_ng = sum(1 for r in ng_correct if r["solution_score"] < 4) / max(1, len(ng_correct)) * 100
    fpr_diff = fpr_g - fpr_ng

    print(f"  FPR guessable: {fpr_g:.1f}%")
    print(f"  FPR non-guessable: {fpr_ng:.1f}%")
    print(f"  Difference: {fpr_diff:.1f}pp")
    print(f"  Kappa: {kappa:.3f}")

    if fpr_diff > 10 and kappa > 0.5:
        print(f"\n  >>> GO: FPR difference {fpr_diff:.1f}pp > 10pp AND kappa {kappa:.3f} > 0.5")
        print(f"  >>> The paper has empirical foundation. Proceed.")
    elif fpr_diff > 5:
        print(f"\n  >>> MARGINAL: FPR difference {fpr_diff:.1f}pp (5-10pp). Effect exists but small.")
        print(f"  >>> Consider EMNLP with more data.")
    else:
        print(f"\n  >>> NO-GO: FPR difference {fpr_diff:.1f}pp < 5pp. Guessability does NOT confound evaluation.")
        print(f"  >>> Rethink approach.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all", help="Step to run: 1, 2, 3, 4, or all")
    args = parser.parse_args()

    problems = load_pilot_problems()
    print(f"Loaded {len(problems)} pilot problems")

    if args.step in ("1", "all"):
        step1_guessability(problems)

    if args.step in ("2", "all"):
        step2_inference(problems)

    if args.step in ("3", "all"):
        step3_verify_and_grade(problems)

    if args.step in ("4", "all"):
        step4_analysis()


if __name__ == "__main__":
    main()
