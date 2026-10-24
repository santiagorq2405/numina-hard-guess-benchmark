"""
Static rubric module (replaces an earlier adaptive rubric).

Two stages:
1. generate_static_rubric(problem, reference_solution) -> str (rubric 0-7)
   Judge sees ONLY problem + reference solution. NOT student solution.
2. grade_with_static_rubric(rubric, problem, student_solution) -> (score, justification)
   Judge applies the static rubric to the student solution.

This avoids the risk that the adaptive rubric overfits to a flawed student solution.
"""

STATIC_RUBRIC_PROMPT = """You are an expert mathematical judge. Generate a grading rubric for this olympiad problem based on the reference solution.

Problem:
{problem}

Reference answer: {answer}

Reference solution:
{solution}

Create a rubric on the IMO 0-7 scale:
- 0: No meaningful progress
- 1-2: Some relevant observations but no real progress
- 3-4: Significant progress, key ideas partially present
- 5-6: Nearly complete solution with minor gaps
- 7: Complete, rigorous solution

Output a concise rubric with 5 score ranges (0, 1-2, 3-4, 5-6, 7), each as one line of criteria that describe what a solution at that level must demonstrate. Base the criteria on the reference solution's key ideas, lemmas, and logical steps.

IMPORTANT: Do NOT mention any specific "student solution". The rubric must be general for any solution that attempts this problem. Focus on required mathematical content and correctness, not on prose style.

Output format:
SCORE 0: <one line criteria>
SCORE 1-2: <one line criteria>
SCORE 3-4: <one line criteria>
SCORE 5-6: <one line criteria>
SCORE 7: <one line criteria>"""


GRADE_WITH_STATIC_PROMPT = """You are an expert mathematical judge. Grade this student solution using the rubric below.

Problem:
{problem}

Reference answer: {answer}

Grading rubric:
{rubric}

Student solution to grade:
{student_solution}

Analyze the student solution step by step, then assign a score from 0 to 7 using the rubric above.

Output format:
SCORE: [integer 0-7]
JUSTIFICATION: [2-3 sentences explaining the score]"""


def parse_score(response_text):
    """Extract integer score 0-7 from grader response. Returns None if not found."""
    if not response_text:
        return None
    import re
    m = re.search(r"SCORE\s*:\s*(\d+)", response_text, re.IGNORECASE)
    if m:
        s = int(m.group(1))
        if 0 <= s <= 7:
            return s
    # Fallback: look for digit after "score"
    m2 = re.search(r"\bscore\b[^0-9]*(\d)\b", response_text.lower())
    if m2:
        s = int(m2.group(1))
        if 0 <= s <= 7:
            return s
    return None


def parse_justification(response_text):
    """Extract justification text. Returns empty string if not found."""
    if not response_text:
        return ""
    import re
    m = re.search(r"JUSTIFICATION\s*:\s*(.+?)(?:\n\n|$)", response_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:500]
    return ""
