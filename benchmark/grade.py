"""Grading suite: BFCL tool calling, tau-bench/GAIA multi-turn, IFEval hard constraints, AIME/GSM8K math, HumanEval+ code, and Artificial Analysis Intelligence Index."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from typing import Any


# ==============================================================================
# --- 1. Tool-Calling & Agentic Grading (BFCL & tau-bench / GAIA) ---
# ==============================================================================

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
        return any(_norm_text(exp) in p for exp in expected)
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


# ==============================================================================
# --- 2. Google IFEval Hard Multi-Constraint Grading ---
# ==============================================================================

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
        if "|" not in text_clean:
            return False, "no markdown table found"
        table_lines = [l.strip() for l in text_clean.splitlines() if l.strip().startswith("|") and l.strip().endswith("|")]
        if len(table_lines) < 3:
            return False, f"table has only {len(table_lines)} rows (< 3)"
        col_count = len([c for c in table_lines[0].split("|") if c.strip()])
        if col_count < 3:
            return False, f"table has {col_count} columns (< 3)"
        after_table = text_clean[text_clean.rfind(table_lines[-1]) + len(table_lines[-1]):].strip()
        if not after_table or len(re.findall(r"\b\w+\b", after_table)) < 10:
            return False, "missing conclusion sentence after table (min 10 words)"
        if "e" in after_table.lower():
            e_words = [w for w in re.findall(r"\b\w+\b", after_table) if "e" in w.lower()]
            return False, f"conclusion contains forbidden letter 'e' in: {e_words[:3]}"
        return True, "valid 3-column table + constrained conclusion (0 'e's)"

    elif rule_id == "h_tags_bold_and_all_caps":
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
        words = re.findall(r"\b\w+\b", text_clean.lower())
        if len(words) < 50:
            return False, f"too short ({len(words)} < 50 words)"
        forbidden = {"secure", "security", "encrypt", "encryption", "key", "keys", "cipher", "ciphers", "protect", "protection"}
        found = [w for w in words if w in forbidden]
        if found:
            return False, f"used forbidden word(s): {list(set(found))}"
        return True, f"explained concepts with 0 forbidden words ({len(words)} words)"

    return False, f"unknown rule {rule_id}"


# ==============================================================================
# --- 3. Competition Math & Multi-Step Reasoning (AIME / Olympiad / CritPt) ---
# ==============================================================================

def grade_gsm8k(expected_answer: int | str, response_text: str) -> tuple[bool, str]:
    if isinstance(expected_answer, str):
        target = expected_answer.strip().lower()
        if f"#### {target}" in response_text.lower() or f"\\boxed{{{target}}}" in response_text.lower():
            return True, f"exact match '{target}'"
        m = re.findall(r"\\boxed\{\s*([^}]+)\s*\}", response_text)
        if m and _norm_text(m[-1]) == _norm_text(target):
            return True, f"boxed match '{m[-1]}'"
        return answer_correct(target, response_text), f"checked target '{target}'"

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


def grade_critpt(expected_answer: int | str, response_text: str) -> tuple[bool, str]:
    return grade_gsm8k(expected_answer, response_text)


# ==============================================================================
# --- 4. GPQA Diamond & Humanity's Last Exam (HLE) Multiple Choice / Exact ---
# ==============================================================================

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


def grade_hle(expected_answer: int | str, response_text: str) -> tuple[bool, str]:
    if isinstance(expected_answer, int):
        return grade_gsm8k(expected_answer, response_text)
    exp = str(expected_answer).strip().upper()
    if exp in ("A", "B", "C", "D", "E"):
        return grade_gpqa(exp, response_text)
    return answer_correct(str(expected_answer), response_text), f"checked '{expected_answer}'"


# ==============================================================================
# --- 5. HumanEval+ / SciCode Sandboxed Python Execution ---
# ==============================================================================

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


def grade_scicode(entry_point: str, prompt: str, test_code: str, response_text: str) -> tuple[bool, str]:
    return grade_humaneval(prompt, test_code, response_text)


# ==============================================================================
# --- 6. T3-Banking (tau-bench) Multi-Turn Stateful Grading ---
# ==============================================================================

def grade_banking(scenario: dict, first_calls: list[dict], final_answer: str, final_db: dict) -> tuple[bool, str]:
    # 1. Check expected transfers occurred in state
    exp_transfers = scenario.get("expected_transfers")
    if exp_transfers is not None:
        db_transfers = final_db.get("transfers", [])
        if len(exp_transfers) == 0 and len(db_transfers) > 0:
            return False, f"unexpected transfer executed: {db_transfers}"
        for exp in exp_transfers:
            matched = any(
                t["from"] == exp["from"] and t["to"] == exp["to"] and math.isclose(t["amount"], exp["amount"], abs_tol=0.01)
                for t in db_transfers
            )
            if not matched:
                return False, f"expected transfer {exp} not found in DB transfers: {db_transfers}"

    # 2. Check frozen cards in state
    exp_frozen = scenario.get("expected_frozen_cards")
    if exp_frozen:
        for cid in exp_frozen:
            if final_db["cards"].get(cid, {}).get("status") != "frozen":
                return False, f"card {cid} status is not 'frozen'"

    # 3. Check unfrozen cards
    exp_unfrozen = scenario.get("expected_unfrozen_cards")
    if exp_unfrozen:
        for cid in exp_unfrozen:
            if final_db["cards"].get(cid, {}).get("status") != "active":
                return False, f"card {cid} status is not 'active'"

    # 4. Check waived fee
    exp_waived = scenario.get("expected_waived_fees")
    if exp_waived:
        for fid in exp_waived:
            if final_db["transactions"].get(fid, {}).get("status") != "waived_refunded":
                return False, f"fee {fid} was not waived in DB"

    # 5. Check disputes
    exp_disputes = scenario.get("expected_disputes")
    if exp_disputes:
        dispute_txs = [d["transaction_id"] for d in final_db.get("disputes", {}).values()]
        for txid in exp_disputes:
            if txid not in dispute_txs:
                return False, f"dispute for transaction {txid} was not recorded in DB"

    # 6. Check final answer keyword assertions
    exp_ans = scenario.get("expected_answer")
    if exp_ans:
        if not answer_correct(exp_ans, final_answer):
            return False, f"final answer missing expected info: {exp_ans}"

    return True, "banking state and assertions verified"


# ==============================================================================
# --- 7. GDPval-AA v2 Structured Financial / Workflows Grading ---
# ==============================================================================

def grade_gdpval(rule_id: str, text: str) -> tuple[bool, str]:
    text_clean = text.strip()
    raw_json = text_clean
    if "```json" in raw_json:
        raw_json = raw_json.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw_json:
        raw_json = raw_json.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(raw_json)
    except Exception as e:
        return False, f"invalid JSON: {str(e)[:35]}"

    if rule_id == "gdpval_balance_sheet_reconciliation":
        # Assets = 450k + 280k + 190k + 1200k = 2,120,000
        # Liab = 210k + 90k + 800k = 1,100,000
        # Equity = 500k + 520k = 1,020,000
        # Total L+E = 2,120,000 (Balanced, variance 0)
        tot_a = data.get("total_assets")
        tot_l = data.get("total_liabilities")
        tot_e = data.get("total_equity")
        if not (tot_a and tot_l and tot_e):
            return False, "missing asset/liability/equity totals"
        if not (math.isclose(tot_a, 2120000, rel_tol=1e-3) and math.isclose(tot_l, 1100000, rel_tol=1e-3) and math.isclose(tot_e, 1020000, rel_tol=1e-3)):
            return False, f"incorrect totals: assets={tot_a}, liab={tot_l}, equity={tot_e}"
        if data.get("is_balanced") is not True:
            return False, "failed to recognize balanced balance sheet"
        return True, "verified balance sheet reconciliation"

    elif rule_id == "gdpval_vendor_sla_audit":
        # 120 min downtime -> SLA breached (99.72% < 99.9%), Tier 1 penalty (10%), credit = $5,000
        if data.get("sla_breached") is not True:
            return False, "failed to detect SLA breach (120 min downtime)"
        credit = data.get("credit_amount_usd")
        if not credit or not math.isclose(float(credit), 5000.0, rel_tol=0.05):
            return False, f"expected credit $5,000, got {credit}"
        return True, "accurate SLA breach credit calculation"

    elif rule_id == "gdpval_saas_metrics":
        # GRR = (1000k - 50k - 30k)/1000k = 92% (0.92 or 92)
        # NRR = (1000k - 50k - 30k + 120k)/1000k = 104% (1.04 or 104)
        grr = float(data.get("grr_percent", 0.0))
        nrr = float(data.get("nrr_percent", 0.0))
        grr_val = grr / 100.0 if grr > 1.0 else grr
        nrr_val = nrr / 100.0 if nrr > 1.0 else nrr
        if not math.isclose(grr_val, 0.92, abs_tol=0.02):
            return False, f"expected GRR ~92%, got {grr}"
        if not math.isclose(nrr_val, 1.04, abs_tol=0.02):
            return False, f"expected NRR ~104%, got {nrr}"
        return True, "accurate SaaS retention calculations"

    elif rule_id == "gdpval_payroll_withholding":
        # Gross = 10,000. SS (6.2%) = 620, Med (1.45%) = 145, Fed (22%) = 2200, State (5%) = 500
        # Deductions = 3465, Net = 6535
        net = float(data.get("net_take_home_pay", 0.0))
        if not math.isclose(net, 6535.0, rel_tol=0.01):
            return False, f"expected net pay $6,535.00, got {net}"
        return True, "accurate payroll tax deductions"

    elif rule_id == "gdpval_cloud_optimization":
        # Spend = 3971.20, 60% savings = 2382.72/mo, 36mo = 85777.92
        sav_mo = float(data.get("monthly_savings_usd", 0.0))
        if not math.isclose(sav_mo, 2382.72, rel_tol=0.05):
            return False, f"expected monthly savings ~$2,382, got {sav_mo}"
        return True, "accurate FinOps savings model"

    elif rule_id == "gdpval_msa_compliance_audit":
        if data.get("clause_compliant") is not False:
            return False, "failed to flag uncapped liability breach"
        cap = float(data.get("recommended_cap_usd", 0.0))
        if not math.isclose(cap, 200000.0, rel_tol=0.05):
            return False, f"expected recommended cap $200,000, got {cap}"
        return True, "contract compliance breach flagged"

    return False, f"unknown gdpval rule: {rule_id}"


# ==============================================================================
# --- 8. AA-Omniscience (Hallucination Restraint & Adversarial Traps) ---
# ==============================================================================

def grade_omniscience(scenario: dict, response_text: str) -> tuple[bool, str]:
    text_lower = response_text.lower()
    is_trap = scenario.get("expected_trap", False)

    if is_trap:
        req_denials = scenario.get("required_denial", [])
        # Model must express restraint / identify false premise
        if any(d.lower() in text_lower for d in req_denials):
            return True, "correctly recognized trap / false premise"
        return False, "hallucinated false premise instead of restraining"

    # Verifiable factual knowledge
    exp_facts = scenario.get("expected_facts", [])
    if all(f.lower() in text_lower for f in exp_facts):
        return True, "accurate factual recall"
    return False, f"missing expected facts: {exp_facts}"


# ==============================================================================
# --- 9. Terminal-Bench v4.0 (CLI / VFS Agent Verification) ---
# ==============================================================================

def grade_terminal(scenario: dict, first_calls: list[dict], final_answer: str, final_vfs: dict) -> tuple[bool, str]:
    # Check expected VFS keys created
    exp_keys = scenario.get("expected_vfs_keys", [])
    for k in exp_keys:
        if k not in final_vfs:
            return False, f"expected file '{k}' was not created in VFS"

    # Check expected content in VFS files
    exp_contains = scenario.get("expected_vfs_contains", {})
    for path, fragments in exp_contains.items():
        content = final_vfs.get(path, "")
        for frag in fragments:
            if frag not in content:
                return False, f"file '{path}' missing expected string: '{frag}'"

    # Check absent strings (e.g. conflict markers removed)
    exp_absent = scenario.get("expected_vfs_absent", {})
    for path, fragments in exp_absent.items():
        content = final_vfs.get(path, "")
        for frag in fragments:
            if frag in content:
                return False, f"file '{path}' still contains forbidden string: '{frag}'"

    # Check valid JSON
    exp_json = scenario.get("expected_json_valid", [])
    for path in exp_json:
        content = final_vfs.get(path, "")
        try:
            json.loads(content)
        except Exception as e:
            return False, f"file '{path}' is not valid JSON: {e}"

    # Check final answer keywords
    exp_ans = scenario.get("expected_answer")
    if exp_ans:
        if not answer_correct(exp_ans, final_answer):
            return False, f"terminal summary missing expected details: {exp_ans}"

    return True, "terminal agent tasks and VFS state verified"


# ==============================================================================
# --- 10. AA-LCR (Long-Context Retrieval & Reasoning) ---
# ==============================================================================

def grade_lcr(scenario: dict, response_text: str) -> tuple[bool, str]:
    expected = scenario.get("expected_answer", "")
    target = _norm_text(expected)
    if f"#### {expected.lower()}" in response_text.lower() or f"\\boxed{{{expected.lower()}}}" in response_text.lower():
        return True, f"exact needle match '{expected}'"
    if answer_correct(expected, response_text):
        return True, f"retrieved target '{expected}'"
    return False, f"failed to extract needle '{expected}'"
