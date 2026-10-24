"""
Guessability Assessment Rubric for NuminaMath problems.
Uses LLM-as-judge to evaluate how likely it is that a model could
guess the correct answer without genuinely solving the problem.
"""

# The guessability assessment prompt
GUESSABILITY_PROMPT = """You are evaluating whether an LLM could GUESS the correct answer to a math competition problem WITHOUT actually solving it.

Think about this concretely: imagine you give this problem to a language model and tell it "Output ONLY the final answer in \\boxed{{}}. Do NOT solve the problem." The model will try to guess based on:
- Statistical patterns from training data (common answer types for this problem category)
- The answer space size (yes/no = 50% chance, single digit integer = ~10% chance each)
- Problem structure cues (e.g., "find the smallest n" usually gives small integers)
- Famous results or well-known problems it may have memorized

Rate the probability that a strong LLM guesses correctly WITHOUT solving:

1 - VERY LOW (<5%): Answer is a specific complex expression, unusual number, non-trivial set, or requires precise computation. Examples: {{1, 2, 4, 12}}, 4n^2 - 3n + 7, f(x) = x^2 - 2x + \\sqrt{{3}}
2 - LOW (5-15%): Answer has some structure but many plausible alternatives exist. Examples: n = 2^k (plausible but many sequences possible), n(n-1)/2 (common formula but which one?)
3 - MODERATE (15-30%): Answer is from a small plausible set. Examples: answer is a single-digit integer (10 options), a common formula like n^2 or 2^n
4 - HIGH (30-60%): Answer space is very constrained. Examples: yes/no questions, answer is 0/1/2 (models default to these), standard functional equation answer f(x)=x
5 - VERY HIGH (>60%): Almost certainly guessable. Examples: answer is "yes" or "no", answer is 0, answer is a universally known constant

CRITICAL: A small integer like 2 or 3 for a "find all functions" or "find the smallest n" problem should be rated 3-4, NOT 1-2. LLMs have strong priors toward small integers. The question is NOT "could a human guess this" but "could a model with statistical priors guess this."

Problem:
{problem}

Correct answer: {answer}

Problem type: {problem_type}
Question type: {question_type}

Provide your assessment in this exact format:
GUESSABILITY_SCORE: [1-5]
ESTIMATED_PROBABILITY: [X%]
REASONING: [2-3 sentences explaining your probability estimate]
ANSWER_PATTERN: [Brief description, e.g., "common small integer", "unusual algebraic expression"]
"""

# For proof problems, we use a different assessment
PROOF_GUESSABILITY_PROMPT = """You are an expert mathematical competition evaluator. This is a PROOF problem, meaning the answer is not a specific value but a mathematical proof or demonstration.

For proof problems, "guessability" means: could a language model produce a convincing-looking proof that would fool an automated answer checker, WITHOUT actually understanding the mathematics?

Rate the proof-guessability on this scale:
1 - IMPOSSIBLE to fake: The proof requires highly specific, non-obvious arguments. A model cannot produce even a partially correct proof without genuine mathematical insight.
2 - VERY HARD to fake: The proof has a clear structure but requires specific technical steps that are hard to guess.
3 - MODERATE: Some standard proof techniques apply, and a model familiar with similar proofs might produce a partially correct argument.
4 - EASY to fake: The proof follows a very standard pattern (e.g., straightforward induction, direct computation) where a template could work.
5 - TRIVIALLY fakeable: The proof is essentially a standard exercise that follows a well-known template.

Problem:
{problem}

Problem type: {problem_type}

Provide your assessment in this exact format:
GUESSABILITY_SCORE: [1-5]
REASONING: [2-3 sentences explaining why you assigned this score]
PROOF_TECHNIQUE: [The main proof technique needed, e.g., "induction", "pigeonhole", "generating functions"]
"""


def parse_guessability_response(response_text):
    """Parse the LLM judge response into structured data."""
    result = {
        "score": None,
        "reasoning": None,
        "pattern_or_technique": None,
        "raw_response": response_text
    }

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("GUESSABILITY_SCORE:"):
            try:
                score_str = line.split(":", 1)[1].strip()
                # Handle cases like "3/5" or just "3"
                score_str = score_str.split("/")[0].strip()
                result["score"] = int(score_str)
            except (ValueError, IndexError):
                pass
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
        elif line.startswith("ANSWER_PATTERN:"):
            result["pattern_or_technique"] = line.split(":", 1)[1].strip()
        elif line.startswith("PROOF_TECHNIQUE:"):
            result["pattern_or_technique"] = line.split(":", 1)[1].strip()

    return result


# Quick heuristic for pre-filtering (before sending to LLM)
def heuristic_guessability(answer, answer_type):
    """Fast heuristic estimate of guessability. Returns 1-5 or None if uncertain."""
    if answer_type == "proof":
        return None  # Need LLM assessment

    if answer_type == "yes_no":
        return 5  # Binary answer, 50% chance of guessing

    if answer_type == "small_integer":
        try:
            val = int(answer)
            if val in [0, 1]:
                return 5  # Extremely common answers
            elif val in [2, 3, 4]:
                return 4  # Very common in competition math
            else:
                return 3  # Less common but still small
        except ValueError:
            return 3

    if answer_type == "no_answer":
        return None  # Can't assess

    if answer_type == "large_integer":
        return 2  # Hard to guess specific large number

    if answer_type == "set":
        return 2  # Hard to guess exact set

    if answer_type == "function_or_expression":
        # Some functional equation answers are very predictable
        if answer and any(std in answer for std in ["f(x)=x", "f(x)=0", "f(x)=c", "f(x) = x"]):
            return 4  # Standard functional equation answer
        return 2

    return None  # Uncertain, need LLM


if __name__ == "__main__":
    # Test the rubric on a few known IMO problems
    test_cases = [
        {
            "problem": "Determine all functions f: R -> R such that...",
            "answer": "f(x) = x",
            "problem_type": "Algebra",
            "question_type": "math_expression",
            "expected_guess": "high (standard functional equation answer)"
        },
        {
            "problem": "Find the smallest positive integer n such that...",
            "answer": "2",
            "problem_type": "Number Theory",
            "question_type": "math_expression",
            "expected_guess": "high (small integer)"
        },
        {
            "problem": "Find all values of x+y where...",
            "answer": "{0,3,4,5,8}",
            "problem_type": "Algebra",
            "question_type": "math_expression",
            "expected_guess": "low (specific non-trivial set)"
        },
        {
            "problem": "IMO 2024 N1: Find all (a,b,c,d) with a*b*c*d = (a+b+c+d)^2 = ...",
            "answer": "{1, 2, 4, 12}",
            "problem_type": "Number Theory",
            "question_type": "math_expression",
            "expected_guess": "low (non-obvious specific values)"
        },
    ]

    print("=== Guessability Rubric Test Cases ===\n")
    for tc in test_cases:
        h = heuristic_guessability(tc["answer"], "small_integer" if tc["answer"].isdigit() else "other")
        print(f"Problem: {tc['problem'][:80]}...")
        print(f"Answer: {tc['answer']}")
        print(f"Expected: {tc['expected_guess']}")
        print(f"Heuristic score: {h}")
        print()
