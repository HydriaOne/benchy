"""Grading suite: BFCL tool calling, tau-bench outcomes, IFEval constraints, GSM8K math, HumanEval code."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any


# --- 1. BFCL / tau-bench Tool Calling Grading ---

def parse_args(s: str) -> dict:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"__raw__": s}


def _norm_value(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm_value(val) for k, val in v.items()}
    return v


def args_subset(expected: dict, predicted: dict) -> bool:
    """Every expected key must be present with an equal (normalized) value."""
    e = _norm_value(expected) if isinstance(expected, dict) else {}
    p = _norm_value(predicted) if isinstance(predicted, dict) else {}
    for k, v in e.items():
        if k not in p:
            return False
        if p[k] != v:
            return False
    return True


def match_calls(expected_calls: list[dict], predicted_calls: list[dict]) -> bool:
    """Greedy one-to-one match of expected tool calls against predicted calls."""
    if len(expected_calls) == 0:
        return len(predicted_calls) == 0
    matched = [False] * len(expected_calls)
    for pc in predicted_calls:
        for i, ec in enumerate(expected_calls):
            if matched[i]:
                continue
            if pc.get("name") == ec.get("name") and args_subset(ec.get("arguments", {}), pc.get("arguments", {})):
                matched[i] = True
                break
    return all(matched)


def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def answer_correct(expected: str, predicted: str) -> bool:
    e = _norm_text(expected)
    p = _norm_text(predicted)
    if not e:
        return False
    return e in p


def grade_no_tool(calls: list[dict], response_text: str, expected_substring: str | None = None) -> tuple[bool, str]:
    """Grade no-tool / restraint scenarios: must NOT call any tools."""
    if calls:
        tool_names = ", ".join(c.get("name", "unknown") for c in calls)
        return False, f"hallucinated tool calls: {tool_names}"
    if not response_text.strip():
        return False, "empty response"
    if expected_substring and not answer_correct(expected_substring, response_text):
        return False, f"response missing expected info: '{expected_substring}'"
    return True, "correctly restrained (0 tools called)"


# --- 2. Google IFEval Constraint Grading ---

def grade_ifeval(rule_id: str, text: str) -> tuple[bool, str]:
    text_clean = text.strip()
    if rule_id == "json_schema":
        raw_json = text_clean
        if "```json" in raw_json:
            raw_json = raw_json.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            d = json.loads(raw_json)
            if not isinstance(d, dict):
                return False, "not a JSON object"
            if not isinstance(d.get("name"), str) or not d.get("name"):
                return False, "missing/invalid 'name'"
            if not isinstance(d.get("years_experience"), int) or isinstance(d.get("years_experience"), bool):
                return False, "missing/invalid integer 'years_experience'"
            if not isinstance(d.get("skills"), list) or len(d.get("skills")) < 3:
                return False, "missing or < 3 'skills'"
            if not isinstance(d.get("remote"), bool):
                return False, "missing/invalid boolean 'remote'"
            return True, "valid JSON schema adherence"
        except Exception as e:
            return False, f"invalid JSON: {str(e)[:30]}"

    elif rule_id == "no_comma":
        words = re.findall(r"\b\w+\b", text_clean)
        if len(words) < 40:
            return False, f"too short ({len(words)} < 40 words)"
        if "," in text_clean:
            return False, "contains comma ','"
        return True, f"passed ({len(words)} words, 0 commas)"

    elif rule_id == "keyword_freq":
        count = len(re.findall(r"\bneurons?\b", text_clean, re.IGNORECASE))
        if count < 4:
            return False, f"keyword 'neuron' count: {count} < 4"
        return True, f"keyword count: {count} >= 4"

    elif rule_id == "exact_paragraphs":
        paragraphs = [p.strip() for p in text_clean.split("\n\n") if p.strip()]
        if len(paragraphs) != 3:
            return False, f"got {len(paragraphs)} paragraphs (expected exactly 3)"
        if re.search(r"^\s*[-*•\d+\.]\s+", text_clean, re.MULTILINE):
            return False, "contains bullet points"
        return True, "exactly 3 paragraphs, no bullets"

    elif rule_id == "tags_and_bold":
        if "<response>" not in text_clean or "</response>" not in text_clean:
            return False, "missing <response> tags"
        bolds = re.findall(r"\*\*[^*]+\*\*", text_clean)
        if len(bolds) < 3:
            return False, f"found {len(bolds)} bold titles (< 3)"
        return True, f"has tags and {len(bolds)} bold titles"

    elif rule_id == "end_phrase":
        cleaned = re.sub(r'[\s"\']+$', "", text_clean)
        target = "The future is green."
        if cleaned.endswith(target) or cleaned.lower().endswith(target.lower()):
            return True, "correct end phrase"
        return False, f"did not end with '{target}'"

    return False, f"unknown rule {rule_id}"


# --- 3. GSM8K Math Reasoning Grading ---

def grade_gsm8k(expected_answer: int, response_text: str) -> tuple[bool, str]:
    # Match standard GSM8K delimiter #### 42
    m = re.findall(r"####\s*(-?\d+)", response_text)
    if m:
        val = int(m[-1])
        return val == expected_answer, f"got {val}, expected {expected_answer}"

    # Match LaTeX \boxed{42}
    m = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", response_text)
    if m:
        val = int(m[-1])
        return val == expected_answer, f"got {val}, expected {expected_answer}"

    # Match phrasing "answer is 42", "total is 42", "equals 42"
    m = re.findall(r"(?:answer is|total is|equals|result is|=)\s*[:\$]?\s*(-?\d+)", response_text, re.IGNORECASE)
    if m:
        val = int(m[-1])
        return val == expected_answer, f"got {val}, expected {expected_answer}"

    # Fallback: look at the last standalone integer in text
    nums = re.findall(r"(?<!\w)-?\d+(?!\w)", response_text)
    if nums:
        val = int(nums[-1])
        return val == expected_answer, f"got {val} (last integer), expected {expected_answer}"

    return False, "no integer answer found in response"


# --- 4. HumanEval Python Code Execution Grading ---

def grade_humaneval(prompt: str, test_code: str, response_text: str) -> tuple[bool, str]:
    code = ""
    if "```python" in response_text:
        parts = response_text.split("```python", 1)[1]
        code = parts.split("```", 1)[0]
    elif "```" in response_text:
        parts = response_text.split("```", 1)[1]
        code = parts.split("```", 1)[0]
    else:
        code = response_text

    full_program = f"""
import sys
import math
from typing import List, Tuple, Optional, Dict, Any

{prompt}
{code}

{test_code}
"""
    try:
        res = subprocess.run(
            [sys.executable, "-c", full_program],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if res.returncode == 0:
            return True, "passed unit tests"
        err_lines = (res.stderr or res.stdout).strip().splitlines()
        err_msg = err_lines[-1] if err_lines else f"exit code {res.returncode}"
        return False, f"assertion failed: {err_msg[:45]}"
    except subprocess.TimeoutExpired:
        return False, "execution timeout (5s)"
    except Exception as e:
        return False, f"exec error: {str(e)[:40]}"
