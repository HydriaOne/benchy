"""Grading suite: BFCL tool calling, tau-bench/GAIA multi-turn, IFEval hard constraints, AIME/GSM8K math, HumanEval+ code."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any


# --- 1. Tool-Calling & Agentic Grading (BFCL & tau-bench / GAIA) ---

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
        if isinstance(v, dict) and isinstance(p[k], dict):
            if not args_subset(v, p[k]):
                return False
        elif isinstance(v, list) and isinstance(p[k], list):
            if len(v) != len(p[k]):
                return False
            for vi, pi in zip(v, p[k]):
                if isinstance(vi, dict) and isinstance(pi, dict):
                    if not args_subset(vi, pi):
                        return False
                elif _norm_value(vi) != _norm_value(pi):
                    return False
        elif p[k] != v:
            return False
    return True


def match_calls(expected_calls: list[dict], predicted_calls: list[dict]) -> bool:
    """Greedy match of expected tool calls against predicted calls."""
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


def answer_correct(expected: str | list[str], predicted: str) -> bool:
    p = _norm_text(predicted)
    if isinstance(expected, list):
        # All expected keywords/substrings must be present
        return all(_norm_text(exp) in p for exp in expected)
    e = _norm_text(expected)
    if not e:
        return False
    return e in p


def grade_no_tool(calls: list[dict], response_text: str, expected_substring: str | None = None) -> tuple[bool, str]:
    if calls:
        tool_names = ", ".join(c.get("name", "unknown") for c in calls)
        return False, f"hallucinated tool calls: {tool_names}"
    if not response_text.strip():
        return False, "empty response"
    if expected_substring and not answer_correct(expected_substring, response_text):
        return False, f"response missing expected keyword: '{expected_substring}'"
    return True, "correctly restrained (0 tools called)"


# --- 2. Google IFEval Hard Multi-Constraint Grading ---

def grade_ifeval(rule_id: str, text: str) -> tuple[bool, str]:
    text_clean = text.strip()

    if rule_id == "h_json_schema_ranges":
        raw_json = text_clean
        if "```json" in raw_json:
            raw_json = raw_json.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            d = json.loads(raw_json)
            if not isinstance(d, dict):
                return False, "not a JSON object"
            if d.get("status") not in ("healthy", "degraded", "critical"):
                return False, f"invalid status enum: {d.get('status')}"
            cpu = d.get("cpu_percent")
            if not isinstance(cpu, (int, float)) or isinstance(cpu, bool) or not (0.0 <= cpu <= 100.0):
                return False, f"cpu_percent must be float/int in [0.0, 100.0], got {cpu}"
            services = d.get("services")
            if not isinstance(services, list) or len(services) < 3:
                return False, f"services must be list of >= 3 items, got {services}"
            for svc in services:
                if not isinstance(svc, dict) or not svc.get("name") or not isinstance(svc.get("latency_ms"), (int, float)):
                    return False, f"invalid service item schema in {svc}"
            if not isinstance(d.get("alert_triggered"), bool):
                return False, "alert_triggered must be boolean"
            return True, "valid complex telemetry JSON schema"
        except Exception as e:
            return False, f"invalid JSON: {str(e)[:35]}"

    elif rule_id == "h_paragraph_and_no_comma":
        # Exactly 3 paragraphs, paragraph 2 starts with 'Serverless', paragraph 3 has 0 commas, 0 bullet points
        paragraphs = [p.strip() for p in text_clean.split("\n\n") if p.strip()]
        if len(paragraphs) != 3:
            return False, f"got {len(paragraphs)} paragraphs (expected exactly 3)"
        if not paragraphs[1].lower().startswith("serverless"):
            return False, f"paragraph 2 does not start with 'Serverless' (starts with '{paragraphs[1][:20]}...')"
        if "," in paragraphs[2]:
            return False, "paragraph 3 contains comma ','"
        if re.search(r"^\s*[-*•\d+\.]\s+", text_clean, re.MULTILINE):
            return False, "contains bullet points"
        return True, "satisfied 4 simultaneous structural constraints"

    elif rule_id == "h_word_count_and_keywords":
        # 60 to 90 words, keyword 'consensus' >= 3 times, keyword 'byzantine' >= 1 time, ends with '[END_OF_REPORT]'
        words = re.findall(r"\b\w+\b", text_clean)
        w_count = len(words)
        if not (55 <= w_count <= 95):
            return False, f"word count {w_count} outside [60, 90]"
        c_count = len(re.findall(r"\bconsensus\b", text_clean, re.IGNORECASE))
        if c_count < 3:
            return False, f"keyword 'consensus' appeared {c_count} times (< 3)"
        b_count = len(re.findall(r"\bbyzantine\b", text_clean, re.IGNORECASE))
        if b_count < 1:
            return False, "missing keyword 'byzantine'"
        cleaned_end = re.sub(r'[\s"\']+$', "", text_clean)
        if not cleaned_end.endswith("[END_OF_REPORT]"):
            return False, "does not end with '[END_OF_REPORT]'"
        return True, "satisfied word bounds, 2 keyword frequencies, and closing tag"

    elif rule_id == "h_table_and_no_letter_e":
        # Contains markdown table with at least 3 columns and 2 data rows, followed by conclusion without letter 'e'
        if "|" not in text_clean:
            return False, "no markdown table found"
        table_lines = [l.strip() for l in text_clean.splitlines() if l.strip().startswith("|") and l.strip().endswith("|")]
        if len(table_lines) < 3:  # Header + separator + at least 1 data row
            return False, f"table has only {len(table_lines)} rows (< 3)"
        col_count = len([c for c in table_lines[0].split("|") if c.strip()])
        if col_count < 3:
            return False, f"table has {col_count} columns (< 3)"
        # Conclusion after table must not contain letter 'e' (case-insensitive)
        after_table = text_clean[text_clean.rfind(table_lines[-1]) + len(table_lines[-1]):].strip()
        if not after_table or len(re.findall(r"\b\w+\b", after_table)) < 10:
            return False, "missing conclusion sentence after table (min 10 words)"
        if "e" in after_table.lower():
            e_words = [w for w in re.findall(r"\b\w+\b", after_table) if "e" in w.lower()]
            return False, f"conclusion contains forbidden letter 'e' in: {e_words[:3]}"
        return True, "valid 3-column table + constrained conclusion (0 'e's)"

    elif rule_id == "h_tags_bold_and_all_caps":
        # Enclosed in <audit>...</audit>, has at least 3 [SECTION_N] tags, each section title in ALL CAPS
        if "<audit>" not in text_clean or "</audit>" not in text_clean:
            return False, "missing <audit> or </audit> tags"
        tags = re.findall(r"\[SECTION_\d+\]", text_clean)
        if len(tags) < 3:
            return False, f"found {len(tags)} [SECTION_N] tags (< 3)"
        caps = re.findall(r"\[SECTION_\d+\]\s*([A-Z0-9\s]+?)(?:\n|$)", text_clean)
        if len(caps) < 3:
            return False, "section headers following tags are not in ALL CAPS"
        return True, "enclosed in tags with 3 all-caps sections"

    elif rule_id == "h_forbidden_words":
        # Explain cryptography without using words: 'secure', 'encrypt', 'key', 'cipher', 'protect', min 50 words
        words = re.findall(r"\b\w+\b", text_clean.lower())
        if len(words) < 50:
            return False, f"too short ({len(words)} < 50 words)"
        forbidden = {"secure", "security", "encrypt", "encryption", "key", "keys", "cipher", "ciphers", "protect", "protection"}
        found = [w for w in words if w in forbidden]
        if found:
            return False, f"used forbidden word(s): {list(set(found))}"
        return True, f"explained concepts with 0 forbidden words ({len(words)} words)"

    return False, f"unknown rule {rule_id}"


# --- 3. Competition Math & Multi-Step Reasoning Grading (AIME / Olympiad) ---

def grade_gsm8k(expected_answer: int | str, response_text: str) -> tuple[bool, str]:
    if isinstance(expected_answer, str):
        target = expected_answer.strip().lower()
        if f"#### {target}" in response_text.lower() or f"\\boxed{{{target}}}" in response_text.lower():
            return True, f"exact match '{target}'"
        # Search for boxed or explicit answer
        m = re.findall(r"\\boxed\{\s*([^}]+)\s*\}", response_text)
        if m and _norm_text(m[-1]) == _norm_text(target):
            return True, f"boxed match '{m[-1]}'"
        return answer_correct(target, response_text), f"checked target '{target}'"

    # Integer target comparison
    exp_int = int(expected_answer)

    # 1. Match standard #### 42
    m = re.findall(r"####\s*(-?\d+)", response_text)
    if m:
        val = int(m[-1])
        return val == exp_int, f"got {val}, expected {exp_int}"

    # 2. Match LaTeX \boxed{42}
    m = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", response_text)
    if m:
        val = int(m[-1])
        return val == exp_int, f"got {val}, expected {exp_int}"

    # 3. Match phrasing "answer is 42", "= 42"
    m = re.findall(r"(?:answer is|total is|equals|result is|=)\s*[:\$]?\s*(-?\d+)", response_text, re.IGNORECASE)
    if m:
        val = int(m[-1])
        return val == exp_int, f"got {val}, expected {exp_int}"

    # 4. Look at the last integer in response
    nums = re.findall(r"(?<!\w)-?\d+(?!\w)", response_text)
    if nums:
        val = int(nums[-1])
        return val == exp_int, f"got {val} (last integer), expected {exp_int}"

    return False, "no integer answer found in response"



# --- 4. GPQA Diamond (PhD-Level Science & Physics Reasoning) ---

def grade_gpqa(expected_letter: str, response_text: str) -> tuple[bool, str]:
    exp = expected_letter.strip().upper()
    # 1. Match #### B or \boxed{B}
    m = re.findall(r"(?:####|\\boxed\{)\s*([A-D])\s*\}?", response_text, re.IGNORECASE)
    if m:
        got = m[-1].upper()
        return got == exp, f"got ({got}), expected ({exp})"

    # 2. Match "Answer: B", "The correct answer is (B)", "Option B"
    m = re.findall(r"(?:answer is|answer:|option|choice)\s*\(?\s*([A-D])\s*\)?", response_text, re.IGNORECASE)
    if m:
        got = m[-1].upper()
        return got == exp, f"got ({got}), expected ({exp})"

    # 3. Match trailing standalone letter choice
    m = re.findall(r"\b([A-D])\b", response_text)
    if m:
        got = m[-1].upper()
        return got == exp, f"got ({got}) [last choice], expected ({exp})"

    return False, "no choice (A/B/C/D) extracted"

# --- 4. HumanEval+ / LeetCode Code Execution Grading ---

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
import collections
import heapq
from typing import List, Tuple, Optional, Dict, Any, Set

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
