"""BFCL-style tool-call grading + tau-bench-style outcome grading."""

from __future__ import annotations

import json
import re


def parse_args(s: str) -> dict:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"__raw__": s}


def _norm_value(v):
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
