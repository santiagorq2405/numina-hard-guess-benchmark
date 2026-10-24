"""
Guessability Rubric v2 -- Rubric-based filtering with 5 concrete strategies.

Design notes:
"I think of alternative filtering strategy where Judge model checks against
the rubrics of most common answer-guessing strategies. Instead of simply asking
the model to grade guessability by 1-5 scale, we can ask to check whether any
of these hacks is possible."

Strategies:
1. Trivial/common answer (1, 1/2, 90 degrees, etc.)
2. Inductive guessing (brute force n=1,2,3,... and spot the pattern)
3. Brute force enumeration ("find all" with small candidate space)
4. Construction without optimality proof (extremal problems)
5. Other obvious shortcut (catch-all)

A problem is EASY_TO_GUESS if AT LEAST ONE strategy is APPLICABLE.
"""

import re


GUESSABILITY_RUBRIC_V2_PROMPT = """You are evaluating a math competition problem to determine if its answer can be reasonably guessed by applying common answer-guessing strategies, WITHOUT performing the actual mathematical proof or rigorous derivation.

Below you will see the problem, its reference answer, and the reference solution. IMPORTANT: the reference solution is the canonical elegant approach. A model could reach the answer through completely different "shortcut" paths that bypass the rigorous reasoning. Your job is to assess whether any such shortcut could plausibly work for this problem.

Check each of the following 5 strategies independently. For each, determine how likely it is to yield the correct answer without genuine solving:

STRATEGY 1 - TRIVIAL / COMMON ANSWER:
Is the answer one of the "usual suspects" that language models output by default for math problems? Specifically:
- Small integers: 0, 1, 2, 3
- Common fractions: 1/2, 1/3, 1/4
- Famous constants: pi, e, sqrt(2), phi
- Common angles: 30, 45, 60, 90, 180 degrees
- Yes/No or True/False
If the answer is exactly one of these, this strategy is APPLICABLE regardless of the problem content.

STRATEGY 2 - PATTERN FROM SMALL CASES (inductive guessing):
Does the problem involve a parameter (usually n) AND the answer follows a very SIMPLE closed form that a model could literally guess from seeing values at n=1, 2, 3? The answer must be in the form of: n, n^2, n!, 2^n, n(n-1)/2, or a similarly standard formula.

APPLICABLE only if ALL of these hold:
- There is a natural parameter n in the problem
- Computing f(1), f(2), f(3) is feasible by hand in minutes
- The resulting values would reveal the pattern (e.g., 1, 2, 6, 24 -> n!)
- The closed form is one of the standard ones above

NOT_APPLICABLE if:
- The closed form involves unusual constants (like "2016/2017^2") that don't follow from small cases
- Small cases are themselves hard to compute
- The pattern requires more than 5 small cases to recognize
- The answer is a complex multi-term expression like "(n-1)^n * n!" where the structure is not obvious from examples
- There is no natural parameter n

When in doubt, prefer NOT_APPLICABLE.

STRATEGY 3 - BRUTE FORCE ENUMERATION:
Does the problem ask for a specific value or set of values that fits a small, easily enumerable space? Examples:
- "Find all positive integers n less than 100 such that..."
- "Find all pairs (a, b) with a, b in {{1,...,10}}..."
- "Find the smallest n satisfying..."
If the candidate space is small enough that checking each candidate is feasible, brute force works.

STRATEGY 4 - CONSTRUCTION WITHOUT OPTIMALITY PROOF:
Is this an EXTREMAL problem (find the MAXIMUM / MINIMUM / BEST X) where the model could just construct a specific example achieving the answer, without proving optimality? The problem must EXPLICITLY ask for a max/min/best value (not just "find X"). For this to apply, a natural construction must exist that gives the right answer.

STRATEGY 5 - OTHER OBVIOUS SHORTCUT (reserved for CLEAR cases only, default is NOT_APPLICABLE):
This strategy exists ONLY for shortcuts that do not fit strategies 1-4 but are still concrete and specific. The default answer is NOT_APPLICABLE.

Only mark as APPLICABLE if ONE of these SPECIFIC conditions is met:
(a) The correct answer literally appears in the problem statement (e.g., problem mentions "the value 42" and the answer is 42)
(b) The problem is identical or nearly identical to a famous classical result whose answer is well-known in the math community
(c) A trivial symmetry argument (like "the answer must be symmetric in x and y") forces the exact answer

Do NOT mark Strategy 5 as APPLICABLE for any of these vague reasons:
- "A model might guess it"
- "There's a pattern a model could see"
- "It's similar to other problems"
- "The answer seems natural"
- "The problem has structure"
These are NOT Strategy 5. If you cannot name the exact shortcut in 5 words (e.g., "answer appears in statement" or "famous classical result"), mark NOT_APPLICABLE.

RULES FOR CLASSIFICATION:
- Mark a strategy as APPLICABLE only if a model has a CONCRETE and REALISTIC chance of reaching the EXACT correct answer through that shortcut. "Theoretically possible" is NOT enough.
- Default to NOT_APPLICABLE when in doubt. We want HIGH precision in detecting easy-to-guess problems.
- A problem is EASY_TO_GUESS if at least one strategy is APPLICABLE with CONFIDENCE >= MEDIUM
- A problem is HARD_TO_GUESS otherwise
- Key principle: the answer must be literally GUESSABLE (reachable via a concrete shortcut that yields the exact answer). If the shortcut gets you close but not exact, mark as NOT_APPLICABLE.
- Balance check: roughly 30-50% of typical olympiad problems should be EASY_TO_GUESS. If you mark everything as EASY, you are being too permissive.

Problem:
{problem}

Reference answer: {answer}

Reference solution (context only, may hide the real shortcuts):
{solution}

IMPORTANT: Output ONLY the structured format below. Do not add any analysis, preamble, or explanation before the output. Start directly with "STRATEGY_1_TRIVIAL:".

Output format (exactly this, one line per strategy, no extra text):

STRATEGY_1_TRIVIAL: APPLICABLE or NOT_APPLICABLE - one sentence reason
STRATEGY_2_PATTERN: APPLICABLE or NOT_APPLICABLE - one sentence reason
STRATEGY_3_BRUTE_FORCE: APPLICABLE or NOT_APPLICABLE - one sentence reason
STRATEGY_4_CONSTRUCTION: APPLICABLE or NOT_APPLICABLE - one sentence reason
STRATEGY_5_OTHER: APPLICABLE or NOT_APPLICABLE - one sentence reason or "none found"

CLASSIFICATION: EASY_TO_GUESS or HARD_TO_GUESS
PRIMARY_STRATEGY: 1 or 2 or 3 or 4 or 5 or none
CONFIDENCE: HIGH or MEDIUM or LOW
REASONING: two sentences justifying the final classification
"""


STRATEGY_KEYS = [
    ("strategy_1_trivial", ["STRATEGY_1_TRIVIAL", "STRATEGY 1 - TRIVIAL", "STRATEGY 1:", "STRATEGY 1 ", "STRATEGY_1:"]),
    ("strategy_2_pattern", ["STRATEGY_2_PATTERN", "STRATEGY 2 - PATTERN", "STRATEGY 2:", "STRATEGY 2 ", "STRATEGY_2:"]),
    ("strategy_3_brute_force", ["STRATEGY_3_BRUTE_FORCE", "STRATEGY 3 - BRUTE", "STRATEGY 3:", "STRATEGY 3 ", "STRATEGY_3:"]),
    ("strategy_4_construction", ["STRATEGY_4_CONSTRUCTION", "STRATEGY 4 - CONSTRUCTION", "STRATEGY 4:", "STRATEGY 4 ", "STRATEGY_4:"]),
    ("strategy_5_other", ["STRATEGY_5_OTHER", "STRATEGY 5 - OTHER", "STRATEGY 5:", "STRATEGY 5 ", "STRATEGY_5:"]),
]


def _find_applicable_nearby(text, start_idx, window=500):
    """Find APPLICABLE or NOT_APPLICABLE within a window of text after start_idx."""
    # Look in next `window` chars for the answer (NOT_APPLICABLE check first to avoid substring)
    snippet = text[start_idx:start_idx + window].upper()

    # Check for NOT_APPLICABLE first (with or without underscore)
    not_app_idx = -1
    for variant in ["NOT_APPLICABLE", "NOT APPLICABLE", "NON-APPLICABLE", "NON APPLICABLE"]:
        idx = snippet.find(variant)
        if idx != -1 and (not_app_idx == -1 or idx < not_app_idx):
            not_app_idx = idx

    # Check for APPLICABLE (must be a word boundary to avoid matching NOT_APPLICABLE substring)
    app_idx = -1
    for m in re.finditer(r'\bAPPLICABLE\b', snippet):
        idx = m.start()
        # Verify not preceded by "NOT " or "NOT_"
        prefix = snippet[max(0, idx - 5):idx]
        if "NOT" not in prefix:
            if app_idx == -1 or idx < app_idx:
                app_idx = idx
            break

    # Return whichever comes first
    if not_app_idx != -1 and (app_idx == -1 or not_app_idx < app_idx):
        return "NOT_APPLICABLE"
    elif app_idx != -1:
        return "APPLICABLE"
    return None


def parse_rubric_v2_response(text):
    """Parse the LLM response into structured data. Flexible to handle multi-line formats."""
    result = {
        "strategies": {},
        "classification": None,
        "primary_strategy": None,
        "confidence": None,
        "reasoning": None,
        "raw_response": text
    }

    if not text:
        return result

    text_upper = text.upper()

    # Parse each strategy by looking for the key pattern and then finding APPLICABLE nearby
    for key, prefixes in STRATEGY_KEYS:
        strategy_found = False
        for prefix in prefixes:
            idx = text_upper.find(prefix.upper())
            if idx != -1:
                # Find APPLICABLE/NOT_APPLICABLE within next 500 chars
                app_result = _find_applicable_nearby(text, idx)
                if app_result:
                    # Extract reason: look for the first sentence after the result
                    section_start = idx
                    section_end = len(text)
                    # Find next STRATEGY or CLASSIFICATION marker
                    for next_prefix in [p for _, ps in STRATEGY_KEYS for p in ps] + ["CLASSIFICATION:", "CLASSIFICATION "]:
                        next_idx = text_upper.find(next_prefix.upper(), idx + len(prefix))
                        if next_idx != -1 and next_idx < section_end:
                            section_end = next_idx
                    section = text[section_start:section_end]

                    # Reason: find text after APPLICABLE/NOT_APPLICABLE until newline or 200 chars
                    reason = ""
                    app_search = "NOT_APPLICABLE" if app_result == "NOT_APPLICABLE" else "APPLICABLE"
                    for variant in [app_search, app_search.replace("_", " ")]:
                        m = re.search(re.escape(variant) + r"[\s\-:,]*(.+?)(?:\n\n|\Z|STRATEGY)", section.upper(), re.DOTALL)
                        if m:
                            reason_start = section_start + section.upper().find(variant) + len(variant)
                            reason_text = text[reason_start:section_end].strip()
                            # Remove leading punctuation
                            reason_text = re.sub(r'^[\s\-:,\.]+', '', reason_text)
                            # Take first 200 chars or until double newline
                            reason = reason_text.split('\n\n')[0][:300].strip()
                            break

                    result["strategies"][key] = {
                        "result": app_result,
                        "reason": reason
                    }
                    strategy_found = True
                    break
        if not strategy_found:
            pass  # Strategy not in response

    # Parse classification
    for cls_prefix in ["CLASSIFICATION:", "CLASSIFICATION "]:
        idx = text_upper.find(cls_prefix.upper())
        if idx != -1:
            rest = text[idx + len(cls_prefix):idx + len(cls_prefix) + 200].upper()
            if "EASY_TO_GUESS" in rest or "EASY TO GUESS" in rest:
                result["classification"] = "EASY_TO_GUESS"
            elif "HARD_TO_GUESS" in rest or "HARD TO GUESS" in rest:
                result["classification"] = "HARD_TO_GUESS"
            break

    # Parse primary strategy
    for ps_prefix in ["PRIMARY_STRATEGY:", "PRIMARY STRATEGY:"]:
        idx = text_upper.find(ps_prefix.upper())
        if idx != -1:
            rest = text[idx + len(ps_prefix):idx + len(ps_prefix) + 100]
            match = re.search(r"\d", rest)
            if match:
                result["primary_strategy"] = int(match.group())
            elif "none" in rest.lower():
                result["primary_strategy"] = None
            break

    # Parse confidence
    for conf_prefix in ["CONFIDENCE:", "CONFIDENCE "]:
        idx = text_upper.find(conf_prefix.upper())
        if idx != -1:
            rest = text[idx + len(conf_prefix):idx + len(conf_prefix) + 100].upper()
            if "HIGH" in rest:
                result["confidence"] = "HIGH"
            elif "MEDIUM" in rest:
                result["confidence"] = "MEDIUM"
            elif "LOW" in rest:
                result["confidence"] = "LOW"
            break

    # Parse reasoning
    for reason_prefix in ["REASONING:", "REASONING "]:
        idx = text_upper.find(reason_prefix.upper())
        if idx != -1:
            rest = text[idx + len(reason_prefix):idx + len(reason_prefix) + 500]
            # Take first couple of sentences
            result["reasoning"] = rest.strip().split('\n\n')[0][:400]
            break

    # Fallback: if classification missing but we have strategies, infer from rule
    if result["classification"] is None and result["strategies"]:
        any_applicable = any(
            s.get("result") == "APPLICABLE"
            for s in result["strategies"].values()
        )
        result["classification"] = "EASY_TO_GUESS" if any_applicable else "HARD_TO_GUESS"

    return result


# Self-test on sample outputs
if __name__ == "__main__":
    test_responses = [
        # Case 1: clear EASY
        """STRATEGY_1_TRIVIAL: APPLICABLE - The answer 2 is in the list of trivial answers
STRATEGY_2_PATTERN: NOT_APPLICABLE - No parameter n in the problem
STRATEGY_3_BRUTE_FORCE: NOT_APPLICABLE - Not an enumeration problem
STRATEGY_4_CONSTRUCTION: NOT_APPLICABLE - Not an extremal problem
STRATEGY_5_OTHER: NOT_APPLICABLE - none found

CLASSIFICATION: EASY_TO_GUESS
PRIMARY_STRATEGY: 1
CONFIDENCE: HIGH
REASONING: The answer is a small integer which models default to. This is the classic trivial answer case.""",

        # Case 2: clear HARD
        """STRATEGY_1_TRIVIAL: NOT_APPLICABLE - The answer is a complex set {1, 2, 4, 12}
STRATEGY_2_PATTERN: NOT_APPLICABLE - No parameter
STRATEGY_3_BRUTE_FORCE: NOT_APPLICABLE - Large candidate space
STRATEGY_4_CONSTRUCTION: NOT_APPLICABLE - Not extremal
STRATEGY_5_OTHER: NOT_APPLICABLE - none found

CLASSIFICATION: HARD_TO_GUESS
PRIMARY_STRATEGY: none
CONFIDENCE: HIGH
REASONING: The answer is a very specific set of integers that cannot be guessed. No shortcut applies.""",

        # Case 3: brackets around answer
        """STRATEGY_1_TRIVIAL: [APPLICABLE] - Small integer
STRATEGY_2_PATTERN: [APPLICABLE] - Has parameter n
STRATEGY_3_BRUTE_FORCE: [NOT_APPLICABLE] - Infinite space
STRATEGY_4_CONSTRUCTION: [NOT_APPLICABLE] - Not extremal
STRATEGY_5_OTHER: [NOT_APPLICABLE] - none found

CLASSIFICATION: [EASY_TO_GUESS]
PRIMARY_STRATEGY: [1]
CONFIDENCE: [MEDIUM]
REASONING: Both strategies 1 and 2 are applicable.""",
    ]

    print("=== PARSER TESTS ===\n")
    for i, resp in enumerate(test_responses, 1):
        parsed = parse_rubric_v2_response(resp)
        print(f"--- Test {i} ---")
        print(f"Classification: {parsed['classification']}")
        print(f"Primary strategy: {parsed['primary_strategy']}")
        print(f"Confidence: {parsed['confidence']}")
        print(f"Strategies: {len(parsed['strategies'])}/5 parsed")
        for key, val in parsed['strategies'].items():
            print(f"  {key}: {val['result']}")
        print()
