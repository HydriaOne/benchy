"""Benchmark orchestrator: throughput (1x + 4x + 8x), tool calling, IFEval, AIME Math, GPQA Diamond, HumanEval+, and Artificial Analysis Intelligence Index.

Prints `METRIC <name>=<value>` lines on stdout; live traces on stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import Any

from .grade import (
    answer_correct,
    args_subset,
    grade_banking,
    grade_critpt,
    grade_gdpval,
    grade_gpqa,
    grade_gsm8k,
    grade_hle,
    grade_humaneval,
    grade_ifeval,
    grade_lcr,
    grade_no_tool,
    grade_omniscience,
    grade_scicode,
    grade_terminal,
    match_calls,
    parse_args,
)
from .live import LiveUI, ReqState, Tracker
from .scenarios import (
    BANKING_SCENARIOS,
    CRITPT_SCENARIOS,
    GDPVAL_SCENARIOS,
    GPQA_SCENARIOS,
    GSM8K_SCENARIOS,
    HLE_SCENARIOS,
    HUMANEVAL_SCENARIOS,
    IFEVAL_SCENARIOS,
    LCR_SCENARIOS,
    OMNISCIENCE_SCENARIOS,
    SCENARIOS,
    SCICODE_SCENARIOS,
    TERMINAL_SCENARIOS,
    THROUGHPUT_PROMPTS,
    TOOLS,
    create_banking_state,
    create_terminal_state,
    execute_banking_tool,
    execute_terminal_tool,
    execute_tool,
)
from .sglang_client import ChatClient, extract_quantization_from_text

DEFAULT_BASE_URL = "http://192.168.1.5:8888"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _detect_default_device() -> str:
    env_dev = os.environ.get("BENCH_DEVICE") or os.environ.get("BENCH_GPU")
    if env_dev:
        return env_dev.strip()
    if sys.platform == "darwin":
        try:
            import subprocess
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if chip:
                return re.sub(r"[^\w\-\.]+", "-", chip).strip("-")
        except Exception:
            pass
        return "Apple-Silicon"
    return "DGX-Spark"


def _fmt_time(seconds: float) -> str:
    if seconds >= 60:
        m = int(seconds // 60)
        rem = seconds % 60
        return f"{m}m {rem:.1f}s"
    return f"{seconds:.1f}s"


def _fmt_tokens(v: int | float | None) -> str:
    """Compact k/M formatting for token counts (leaderboard columns)."""
    if v is None:
        return "N/A"
    try:
        v = int(v)
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}k"
    return str(v)


def _sum_tokens(rows) -> tuple[int, int, int]:
    """Sum (prompt, completion, reasoning) tokens across result rows."""
    prompt = sum(r.prompt_tokens for r in rows if not r.error)
    completion = sum(r.completion_tokens for r in rows if not r.error)
    reasoning = sum(r.reasoning_tokens for r in rows if not r.error)
    return prompt, completion, reasoning


def _normalize_thinking(val: Any) -> str:
    if val is None:
        return "auto"
    if isinstance(val, bool):
        return "auto" if val else "off"
    s = str(val).strip().lower()
    if s in ("off", "false", "no", "0", "none"):
        return "off"
    if s in ("low", "min"):
        return "low"
    if s in ("medium", "med", "mid"):
        return "medium"
    if s in ("high", "max"):
        return "high"
    if s in ("xhigh", "extra-high", "very-high", "extra_high"):
        return "xhigh"
    if s in ("auto", "on", "true", "yes", "1"):
        return "auto"
    return s


def _env_thinking() -> str:
    raw = os.environ.get("BENCH_THINKING") or os.environ.get("BENCH_ENABLE_THINKING")
    if not raw:
        return "auto"
    return _normalize_thinking(raw)


def _detect_quant(model_name: str | None, raw_quant: str | None = None) -> str:
    if raw_quant:
        q = str(raw_quant).strip()
        if q.lower() in ("null", "none", ""):
            return "N/A"
        eq = extract_quantization_from_text(q)
        return eq if eq else q
    if not model_name:
        return "auto"
    eq = extract_quantization_from_text(model_name)
    return eq if eq else "auto"


def _env_quant() -> str | None:
    return os.environ.get("BENCH_QUANT") or os.environ.get("BENCH_QUANTIZATION") or None


@dataclass
class Config:
    base_url: str = field(default_factory=lambda: os.environ.get("BENCH_BASE_URL", DEFAULT_BASE_URL))
    model: str | None = field(default_factory=lambda: os.environ.get("BENCH_MODEL") or None)
    concurrency: int = field(default_factory=lambda: _env_int("BENCH_CONCURRENCY", 8))
    max_tokens: int = field(default_factory=lambda: _env_int("BENCH_MAX_TOKENS", 4096))
    tool_max_tokens: int = field(default_factory=lambda: _env_int("BENCH_TOOL_MAX_TOKENS", 4096))
    scenario_limit: int = field(default_factory=lambda: _env_int("BENCH_SCENARIOS", 0))
    temperature: float = 0.0
    thinking: str = field(default_factory=_env_thinking)
    system_prompt: str | None = field(default_factory=lambda: os.environ.get("BENCH_SYSTEM_PROMPT") or None)
    repeats: int = field(default_factory=lambda: _env_int("BENCH_REPEATS", 3))
    sweep: bool = field(default_factory=lambda: os.environ.get("BENCH_SWEEP", "").strip().lower() in ("1", "true", "yes", "on"))
    seed: int = field(default_factory=lambda: _env_int("BENCH_SEED", 42))
    api_key: str | None = field(default_factory=lambda: os.environ.get("BENCH_API_KEY") or None)
    device: str = field(default_factory=_detect_default_device)
    engine: str | None = field(default_factory=lambda: os.environ.get("BENCH_ENGINE") or None)
    quant: str | None = field(default_factory=_env_quant)
    results_dir: str = field(default_factory=lambda: os.environ.get("BENCH_RESULTS_DIR", "results"))
    eval_suites: str = field(default_factory=lambda: os.environ.get("BENCH_EVAL", "all"))
    no_record: bool = field(default_factory=lambda: os.environ.get("BENCH_NO_RECORD", "").strip().lower() in ("1", "true", "yes", "on"))

    @property
    def enable_thinking(self) -> bool:
        return self.thinking != "off"

    @property
    def thinking_kwargs(self) -> dict | None:
        if self.thinking == "off":
            return {"enable_thinking": False}
        effort = self.resolved_reasoning_effort
        if effort:
            return {"enable_thinking": True, "reasoning_effort": effort}
        if self.thinking == "auto":
            return {"enable_thinking": True}
        return {"enable_thinking": True}

    @property
    def resolved_reasoning_effort(self) -> str | None:
        t = (self.thinking or "").lower().strip()
        if not t or t in ("auto", "off"):
            return None
        # Qwen chat templates accept 'low', 'medium', 'xhigh' (and reject 'high' with HTTP 400)
        is_qwen = "qwen" in (self.model or "").lower()
        if t == "high" and is_qwen:
            return "xhigh"
        if t in ("low", "medium", "high", "xhigh"):
            return t
        return None

    @property
    def reasoning_effort(self) -> str | None:
        return self.resolved_reasoning_effort
    @property
    def resolved_quant(self) -> str:
        return _detect_quant(self.model, self.quant)

    def should_eval(self, suite: str) -> bool:
        s = self.eval_suites.strip().lower()
        if s in ("all", "*"):
            return True
        active = [x.strip().lower() for x in s.split(",") if x.strip()]
        suite_l = suite.lower()
        if "aa-index" in active or "aa_index" in active or "aa" in active:
            if suite_l in ("gpqa", "critpt", "hle", "banking", "gdpval", "omniscience", "scicode", "terminal", "lcr"):
                return True
        if "core" in active:
            if suite_l in ("tool", "ifeval", "gsm8k", "aime", "gpqa", "humaneval"):
                return True
        return suite_l in active


@dataclass
class ScenarioResult:
    id: str
    category: str
    ok: bool = False
    detail: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    error: str = ""
    reasoning_starved: bool = False

    @property
    def is_starved(self) -> bool:
        if self.reasoning_starved:
            return True
        ans_tok = max(self.completion_tokens - self.reasoning_tokens, 0)
        return (not self.ok) and self.completion_tokens > 0 and self.reasoning_tokens > 0 and (
            ans_tok < 32 or self.reasoning_tokens >= (self.completion_tokens * 0.75)
        )

def _on_chunk(state: ReqState, tracker: Tracker, live: LiveUI, chunk: dict) -> None:
    choices = chunk.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        rc = delta.get("reasoning_content") or delta.get("reasoning")
        ct = delta.get("content")
        tcs = delta.get("tool_calls")
        if state.status == "pending":
            state.status = "streaming"
        if (rc or ct or tcs) and state.ttft_s is None:
            state.ttft_s = time.monotonic() - state.started_at
        if rc:
            state.reasoning_chars += len(rc)
            state.trace += rc
        if ct:
            state.content_chars += len(ct)
            state.trace += ct
        for tc in tcs or []:
            fn = tc.get("function") or {}
            if fn.get("name"):
                state.tool_calls.append(fn["name"])
                state.trace += f"\n[TOOL] {fn['name']}("
            if fn.get("arguments"):
                state.trace += fn["arguments"]
        if len(state.trace) > 8000:
            state.trace = state.trace[-8000:]
    tracker.touch(state)
    live.update()


async def _run_stream(client, state, tracker, live, messages, *, tools, max_tokens, temperature, chat_template_kwargs=None, reasoning_effort=None):
    state.started_at = time.monotonic()
    res = await client.stream(
        messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        chat_template_kwargs=chat_template_kwargs,
        reasoning_effort=reasoning_effort,
        on_chunk=partial(_on_chunk, state, tracker, live),
    )
    state.elapsed_s = time.monotonic() - state.started_at
    state.completion_tokens = res.completion_tokens
    state.reasoning_tokens = res.reasoning_tokens
    if state.elapsed_s > 0:
        state.tps = res.completion_tokens / state.elapsed_s
    if res.error:
        state.status = "error"
        state.error = res.error
    elif state.status != "error":
        state.status = "done"
    state.finish_reason = res.finish_reason or ""
    res.elapsed_s = state.elapsed_s
    res.ttft_s = state.ttft_s
    ans_tokens = max(res.completion_tokens - res.reasoning_tokens, 0)
    if res.finish_reason == "length" and res.reasoning_tokens > 0:
        if ans_tokens < 32 or res.reasoning_tokens >= (res.completion_tokens * 0.75):
            state.reasoning_starved = True
    return res


async def _throughput(client, tracker, live, cfg, concurrency: int):
    states: list[ReqState] = []
    tasks = []
    count = concurrency if concurrency > 1 else len(THROUGHPUT_PROMPTS)
    prompts = (THROUGHPUT_PROMPTS * (count // len(THROUGHPUT_PROMPTS) + 1))[:count]
    phase_label = f"c{concurrency}" if concurrency > 1 else "single"
    for i, prompt in enumerate(prompts):
        name = f"c{concurrency}_{i + 1}" if concurrency > 1 else f"s{i + 1}"
        st = tracker.add(name, phase_label)
        states.append(st)
        messages = [{"role": "user", "content": prompt}]
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.system_prompt})
        tasks.append(
            _run_stream(
                client, st, tracker, live,
                messages,
                tools=None, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                chat_template_kwargs=cfg.thinking_kwargs,
                reasoning_effort=cfg.reasoning_effort,
            )
        )
    t0 = time.monotonic()
    if concurrency > 1:
        results = await asyncio.gather(*tasks)
    else:
        results = [await t for t in tasks]
    wall = time.monotonic() - t0
    for st in states:
        live.note_done(st)
    return results, wall


def _grade_tool(sc: dict, first_calls: list[dict], final_answer: str) -> tuple[bool, str]:
    cat = sc["category"]
    if cat == "no_tool":
        return grade_no_tool(first_calls, final_answer, sc.get("expected_answer"))
    if cat in ("simple", "parallel", "complex_args", "distractor_tools"):
        ok = match_calls(sc["expected_calls"], first_calls)
        return ok, "calls-ok" if ok else f"calls-mismatch got={len(first_calls)}/{len(sc['expected_calls'])}"
    ok = answer_correct(sc["expected_answer"], final_answer)
    return ok, "answer-ok" if ok else "answer-mismatch"


async def _run_scenario(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, f"tool/{sc['category']}")
    st.started_at = time.monotonic()
    tools = [TOOLS[t] for t in sc["tools"]]
    messages = [dict(m) for m in sc["messages"]]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    first_calls: list[dict] = []
    final_answer = ""
    tot_tokens = 0
    tot_reasoning = 0
    tot_prompt = 0
    try:
        for turn in range(4):
            res = await _run_stream(
                client, st, tracker, live, messages,
                tools=tools, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
                chat_template_kwargs=cfg.thinking_kwargs,
                reasoning_effort=cfg.reasoning_effort,
            )
            tot_tokens += res.completion_tokens
            tot_reasoning += res.reasoning_tokens
            tot_prompt += res.prompt_tokens
            if res.error:
                st.status = "error"
                return ScenarioResult(id=name, category=sc["category"], error=res.error,
                                      prompt_tokens=tot_prompt, completion_tokens=tot_tokens,
                                      reasoning_tokens=tot_reasoning)
            calls = [{"name": tc.name, "arguments": parse_args(tc.arguments)} for tc in res.tool_calls]
            if turn == 0:
                first_calls = calls
            assistant = {"role": "assistant", "content": res.content or ""}
            if res.tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id or f"call_{turn}_{i}", "type": "function",
                     "function": {"name": tc.name, "arguments": tc.arguments}}
                    for i, tc in enumerate(res.tool_calls)
                ]
            messages.append(assistant)
            if res.finish_reason == "tool_calls" and res.tool_calls:
                for i, tc in enumerate(res.tool_calls):
                    out = execute_tool(tc.name, parse_args(tc.arguments))
                    messages.append({"role": "tool", "tool_call_id": assistant["tool_calls"][i]["id"], "content": out})
                continue
            final_answer = res.content
            break
        ok, detail = _grade_tool(sc, first_calls, final_answer)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category=sc["category"], ok=ok, detail=detail,
                              prompt_tokens=tot_prompt, completion_tokens=tot_tokens,
                              reasoning_tokens=tot_reasoning)
    finally:
        st.elapsed_s = time.monotonic() - st.started_at
        st.tps = tot_tokens / st.elapsed_s if st.elapsed_s > 0 else 0.0
        st.completion_tokens = tot_tokens
        st.reasoning_tokens = tot_reasoning
        live.note_done(st)


async def _run_eval_suite(client, tracker, live, cfg, scenarios, runner_fn) -> list[ScenarioResult]:
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def _one(sc):
        async with sem:
            return await runner_fn(client, tracker, live, cfg, sc)

    return list(await asyncio.gather(*[_one(sc) for sc in scenarios]))


# --- Suite 1: Tool Calling ---
async def _toolcalls(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = SCENARIOS if cfg.scenario_limit <= 0 else SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _run_scenario)


# --- Suite 2: IFEval ---
async def _one_ifeval(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "ifeval")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="ifeval", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_ifeval(sc["rule_id"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="ifeval", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_ifeval(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = IFEVAL_SCENARIOS if cfg.scenario_limit <= 0 else IFEVAL_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_ifeval)


# --- Suite 3: GSM8K / AIME Math ---
async def _one_gsm8k(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "aime")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="aime", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_gsm8k(sc["expected_answer"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="aime", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_gsm8k(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = GSM8K_SCENARIOS if cfg.scenario_limit <= 0 else GSM8K_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_gsm8k)


# --- Suite 4: GPQA Diamond ---
async def _one_gpqa(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "gpqa")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="gpqa", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_gpqa(sc["expected_answer"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="gpqa", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_gpqa(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = GPQA_SCENARIOS if cfg.scenario_limit <= 0 else GPQA_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_gpqa)


# --- Suite 5: HumanEval+ / LeetCode ---
async def _one_humaneval(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "humaneval")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": f"Complete the following Python code:\n\n{sc['prompt']}"}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="humaneval", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_humaneval(sc["prompt"], sc["test"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="humaneval", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_humaneval(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = HUMANEVAL_SCENARIOS if cfg.scenario_limit <= 0 else HUMANEVAL_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_humaneval)


# --- Suite 6: CritPt (Competition Physics & Reasoning) ---
async def _one_critpt(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "critpt")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="critpt", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_critpt(sc["expected_answer"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="critpt", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_critpt(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = CRITPT_SCENARIOS if cfg.scenario_limit <= 0 else CRITPT_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_critpt)


# --- Suite 7: Humanity's Last Exam (HLE) ---
async def _one_hle(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "hle")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="hle", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_hle(sc["expected_answer"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="hle", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_hle(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = HLE_SCENARIOS if cfg.scenario_limit <= 0 else HLE_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_hle)


# --- Suite 8: T3-Banking (tau-bench Multi-Turn Stateful Agentic) ---
async def _run_banking_scenario(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "banking")
    st.started_at = time.monotonic()
    db = create_banking_state()
    tools = [TOOLS[t] for t in sc["tools"]]
    messages = [dict(m) for m in sc["messages"]]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    first_calls: list[dict] = []
    final_answer = ""
    tot_tokens = 0
    tot_reasoning = 0
    tot_prompt = 0
    try:
        for turn in range(4):
            res = await _run_stream(
                client, st, tracker, live, messages,
                tools=tools, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
                chat_template_kwargs=cfg.thinking_kwargs,
                reasoning_effort=cfg.reasoning_effort,
            )
            tot_tokens += res.completion_tokens
            tot_reasoning += res.reasoning_tokens
            tot_prompt += res.prompt_tokens
            if res.error:
                st.status = "error"
                return ScenarioResult(id=name, category="banking", error=res.error,
                                      prompt_tokens=tot_prompt, completion_tokens=tot_tokens,
                                      reasoning_tokens=tot_reasoning)
            calls = [{"name": tc.name, "arguments": parse_args(tc.arguments)} for tc in res.tool_calls]
            if turn == 0:
                first_calls = calls
            assistant = {"role": "assistant", "content": res.content or ""}
            if res.tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id or f"call_bk_{turn}_{i}", "type": "function",
                     "function": {"name": tc.name, "arguments": tc.arguments}}
                    for i, tc in enumerate(res.tool_calls)
                ]
            messages.append(assistant)
            if res.finish_reason == "tool_calls" and res.tool_calls:
                for i, tc in enumerate(res.tool_calls):
                    out = execute_banking_tool(db, tc.name, parse_args(tc.arguments))
                    messages.append({"role": "tool", "tool_call_id": assistant["tool_calls"][i]["id"], "content": out})
                continue
            final_answer = res.content
            break
        ok, detail = grade_banking(sc, first_calls, final_answer, db)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="banking", ok=ok, detail=detail,
                              prompt_tokens=tot_prompt, completion_tokens=tot_tokens,
                              reasoning_tokens=tot_reasoning)
    finally:
        st.elapsed_s = time.monotonic() - st.started_at
        st.tps = tot_tokens / st.elapsed_s if st.elapsed_s > 0 else 0.0
        st.completion_tokens = tot_tokens
        st.reasoning_tokens = tot_reasoning
        live.note_done(st)


async def _run_banking(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = BANKING_SCENARIOS if cfg.scenario_limit <= 0 else BANKING_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _run_banking_scenario)


# --- Suite 9: GDPval-AA v2 ---
async def _one_gdpval(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "gdpval")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="gdpval", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_gdpval(sc["rule_id"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="gdpval", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_gdpval(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = GDPVAL_SCENARIOS if cfg.scenario_limit <= 0 else GDPVAL_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_gdpval)


# --- Suite 10: AA-Omniscience ---
async def _one_omniscience(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "omniscience")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="omniscience", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_omniscience(sc, res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="omniscience", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_omniscience(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = OMNISCIENCE_SCENARIOS if cfg.scenario_limit <= 0 else OMNISCIENCE_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_omniscience)


# --- Suite 11: SciCode ---
async def _one_scicode(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "scicode")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": f"Implement the following scientific Python function:\n\n{sc['prompt']}"}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="scicode", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_scicode(sc["entry_point"], sc["prompt"], sc["test"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="scicode", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_scicode(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = SCICODE_SCENARIOS if cfg.scenario_limit <= 0 else SCICODE_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_scicode)


# --- Suite 12: Terminal-Bench v4.0 ---
async def _run_terminal_scenario(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "terminal")
    st.started_at = time.monotonic()
    vfs = create_terminal_state()
    tools = [TOOLS[t] for t in sc["tools"]]
    messages = [dict(m) for m in sc["messages"]]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    first_calls: list[dict] = []
    final_answer = ""
    tot_tokens = 0
    tot_reasoning = 0
    tot_prompt = 0
    try:
        for turn in range(4):
            res = await _run_stream(
                client, st, tracker, live, messages,
                tools=tools, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
                chat_template_kwargs=cfg.thinking_kwargs,
                reasoning_effort=cfg.reasoning_effort,
            )
            tot_tokens += res.completion_tokens
            tot_reasoning += res.reasoning_tokens
            tot_prompt += res.prompt_tokens
            if res.error:
                st.status = "error"
                return ScenarioResult(id=name, category="terminal", error=res.error,
                                      prompt_tokens=tot_prompt, completion_tokens=tot_tokens,
                                      reasoning_tokens=tot_reasoning)
            calls = [{"name": tc.name, "arguments": parse_args(tc.arguments)} for tc in res.tool_calls]
            if turn == 0:
                first_calls = calls
            assistant = {"role": "assistant", "content": res.content or ""}
            if res.tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id or f"call_term_{turn}_{i}", "type": "function",
                     "function": {"name": tc.name, "arguments": tc.arguments}}
                    for i, tc in enumerate(res.tool_calls)
                ]
            messages.append(assistant)
            if res.finish_reason == "tool_calls" and res.tool_calls:
                for i, tc in enumerate(res.tool_calls):
                    out = execute_terminal_tool(vfs, tc.name, parse_args(tc.arguments))
                    messages.append({"role": "tool", "tool_call_id": assistant["tool_calls"][i]["id"], "content": out})
                continue
            final_answer = res.content
            break
        ok, detail = grade_terminal(sc, first_calls, final_answer, vfs)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="terminal", ok=ok, detail=detail,
                              prompt_tokens=tot_prompt, completion_tokens=tot_tokens,
                              reasoning_tokens=tot_reasoning)
    finally:
        st.elapsed_s = time.monotonic() - st.started_at
        st.tps = tot_tokens / st.elapsed_s if st.elapsed_s > 0 else 0.0
        st.completion_tokens = tot_tokens
        st.reasoning_tokens = tot_reasoning
        live.note_done(st)


async def _run_terminal(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = TERMINAL_SCENARIOS if cfg.scenario_limit <= 0 else TERMINAL_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _run_terminal_scenario)


# --- Suite 13: AA-LCR (Long Context Reasoning) ---
async def _one_lcr(client, tracker, live, cfg, sc) -> ScenarioResult:
    name = sc["id"]
    st = tracker.add(name, "lcr")
    st.started_at = time.monotonic()
    messages = [{"role": "user", "content": sc["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    try:
        res = await _run_stream(
            client, st, tracker, live, messages,
            tools=None, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
            chat_template_kwargs=cfg.thinking_kwargs,
            reasoning_effort=cfg.reasoning_effort,
        )
        if res.error:
            st.status = "error"
            return ScenarioResult(id=name, category="lcr", error=res.error,
                                  prompt_tokens=res.prompt_tokens,
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_lcr(sc, res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="lcr", ok=ok, detail=detail,
                              prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_lcr(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = LCR_SCENARIOS if cfg.scenario_limit <= 0 else LCR_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_lcr)


def _sanitize_name(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def _save_report_md(
    cfg: Config,
    single_tps: float,
    s_tok: int,
    conc4_tps: float,
    conc4_reps: list,
    lo4: float,
    hi4: float,
    conc8_tps: float,
    conc8_reps: list,
    lo8: float,
    hi8: float,
    ttft_ms: float,
    ratio: float,
    tool_results: list[ScenarioResult] | None,
    ifeval_results: list[ScenarioResult] | None,
    gsm8k_results: list[ScenarioResult] | None,
    gpqa_results: list[ScenarioResult] | None,
    humaneval_results: list[ScenarioResult] | None,
    critpt_results: list[ScenarioResult] | None,
    hle_results: list[ScenarioResult] | None,
    banking_results: list[ScenarioResult] | None,
    gdpval_results: list[ScenarioResult] | None,
    omniscience_results: list[ScenarioResult] | None,
    scicode_results: list[ScenarioResult] | None,
    terminal_results: list[ScenarioResult] | None,
    lcr_results: list[ScenarioResult] | None,
    composite_score: float | None,
    aa_index_score: float | None,
    total_duration_s: float = 0.0,
    sweep_data: list | None = None,
    token_rows: list | None = None,
    token_totals: tuple | None = None,
    quality_per_time: float | None = None,
) -> str:
    os.makedirs(cfg.results_dir, exist_ok=True)
    dev_clean = _sanitize_name(cfg.device)
    model_clean = _sanitize_name(cfg.model or "unknown-model")
    filename = f"{dev_clean}-{model_clean}.md"
    filepath = os.path.join(cfg.results_dir, filename)

    now_utc = datetime.now(timezone.utc)
    date_iso = now_utc.isoformat()
    date_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Compute individual accuracies
    def _calc_acc(rs):
        return (sum(1 for r in rs if r.ok) / len(rs)) if rs else None

    tool_acc = _calc_acc(tool_results)
    ifeval_acc = _calc_acc(ifeval_results)
    gsm8k_acc = _calc_acc(gsm8k_results)
    gpqa_acc = _calc_acc(gpqa_results)
    he_acc = _calc_acc(humaneval_results)
    critpt_acc = _calc_acc(critpt_results)
    hle_acc = _calc_acc(hle_results)
    banking_acc = _calc_acc(banking_results)
    gdpval_acc = _calc_acc(gdpval_results)
    omni_acc = _calc_acc(omniscience_results)
    scicode_acc = _calc_acc(scicode_results)
    term_acc = _calc_acc(terminal_results)
    lcr_acc = _calc_acc(lcr_results)

    token_rows = token_rows or []
    t_in, t_out, t_reas = token_totals if token_totals is not None else (0, 0, 0)

    lines = [
        "---",
        f'model: "{cfg.model}"',
        f'device: "{cfg.device}"',
        f'engine: "{cfg.engine or "auto"}"',
        f'quant: "{cfg.resolved_quant}"',
        f'endpoint: "{cfg.base_url}"',
        f'thinking: "{cfg.thinking}"',
        f'date: "{date_iso}"',
        f"tokens_per_second: {conc8_tps:.3f}",
        f"conc8_tps: {conc8_tps:.3f}",
        f"conc4_tps: {conc4_tps:.3f}",
        f"single_stream_tps: {single_tps:.3f}",
        f"time_to_first_token_ms: {ttft_ms:.3f}",
        f"total_duration_seconds: {total_duration_s:.3f}",
        f"smart_composite_score: {composite_score:.4f}" if composite_score is not None else "smart_composite_score: null",
        f"aa_intelligence_index: {aa_index_score:.4f}" if aa_index_score is not None else "aa_intelligence_index: null",
        f"tool_call_accuracy: {tool_acc:.4f}" if tool_acc is not None else "tool_call_accuracy: null",
        f"ifeval_accuracy: {ifeval_acc:.4f}" if ifeval_acc is not None else "ifeval_accuracy: null",
        f"gsm8k_accuracy: {gsm8k_acc:.4f}" if gsm8k_acc is not None else "gsm8k_accuracy: null",
        f"gpqa_accuracy: {gpqa_acc:.4f}" if gpqa_acc is not None else "gpqa_accuracy: null",
        f"humaneval_accuracy: {he_acc:.4f}" if he_acc is not None else "humaneval_accuracy: null",
        f"critpt_accuracy: {critpt_acc:.4f}" if critpt_acc is not None else "critpt_accuracy: null",
        f"hle_accuracy: {hle_acc:.4f}" if hle_acc is not None else "hle_accuracy: null",
        f"banking_accuracy: {banking_acc:.4f}" if banking_acc is not None else "banking_accuracy: null",
        f"gdpval_accuracy: {gdpval_acc:.4f}" if gdpval_acc is not None else "gdpval_accuracy: null",
        f"omniscience_accuracy: {omni_acc:.4f}" if omni_acc is not None else "omniscience_accuracy: null",
        f"scicode_accuracy: {scicode_acc:.4f}" if scicode_acc is not None else "scicode_accuracy: null",
        f"terminal_accuracy: {term_acc:.4f}" if term_acc is not None else "terminal_accuracy: null",
        f"lcr_accuracy: {lcr_acc:.4f}" if lcr_acc is not None else "lcr_accuracy: null",
        f"reasoning_ratio: {ratio:.4f}",
        f"quality_per_time: {quality_per_time:.4f}" if quality_per_time is not None else "quality_per_time: null",
        f"total_tokens: {t_in + t_out}",
        f"input_tokens: {t_in}",
        f"output_tokens: {t_out}",
        "---",
        "",
        f"# Benchmark Report: {cfg.model} on {cfg.device}",
        "",
        f"- **Date:** {date_str}",
        f"- **Device / GPU:** `{cfg.device}`",
        f"- **Serving Engine:** `{cfg.engine or 'OpenAI-Compatible'}`",
        f"- **Quantization:** `{cfg.resolved_quant}`",
        f"- **Endpoint:** `{cfg.base_url}`",
        f"- **Model:** `{cfg.model}`",
        f"- **Thinking Mode:** `{cfg.thinking}`",
        f"- **Total Execution Time:** **`{_fmt_time(total_duration_s)}`** ({total_duration_s:.1f}s)",
        f"- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `{len(conc8_reps)}`)",
        f"- **Seed:** `{cfg.seed}`",
    ]

    if composite_score is not None:
        lines.append(f"- **Composite Intelligence Score:** **`{composite_score * 100:.1f}%`**")
    if aa_index_score is not None:
        lines.append(f"- **Artificial Analysis Intelligence Index:** **`{aa_index_score * 100:.1f}%`**")
    all_eval_scenarios = []
    for rs in (
        tool_results, ifeval_results, gsm8k_results, gpqa_results, humaneval_results,
        critpt_results, hle_results, banking_results, gdpval_results, omniscience_results,
        scicode_results, terminal_results, lcr_results,
    ):
        if rs:
            all_eval_scenarios.extend(rs)

    starved = [r for r in all_eval_scenarios if getattr(r, "is_starved", False) or getattr(r, "reasoning_starved", False)]
    if starved:
        lines.extend([
            "",
            "> ⚠️ **Warning: Reasoning Token Starvation Detected**  ",
            f"> **{len(starved)} scenario(s)** (`{', '.join(r.id for r in starved)}`) burned their token budget inside `<think>` (`finish_reason: length` with near-zero answer tokens).  ",
            "> The model was truncated before producing a final answer. Increase `--tool-max-tokens` / `--max-tokens` or evaluate with `--no-thinking` / `--thinking low` to prevent answer truncation.",
        ])

    lines.extend([
        "",
        "## Throughput Performance",
        "",
        "| Metric | Value | Details |",
        "|---|---|---|",
        f"| **8-Concurrent Throughput** | **`{conc8_tps:.2f} tok/s`** | median of {len(conc8_reps)} reps (spread: {lo8:.1f}–{hi8:.1f} tok/s) |",
        f"| **4-Concurrent Throughput** | **`{conc4_tps:.2f} tok/s`** | median of {len(conc4_reps)} reps (spread: {lo4:.1f}–{hi4:.1f} tok/s) |",
        f"| **Single-Stream Throughput** | **`{single_tps:.2f} tok/s`** | {s_tok} tokens generated |",
        f"| **Mean TTFT (8-Concurrent)** | **`{ttft_ms:.1f} ms`** | time to first token |",
        f"| **Total Execution Time** | **`{_fmt_time(total_duration_s)}`** | total benchmark wall-clock time ({total_duration_s:.1f}s) |",
        f"| **Reasoning Ratio** | **`{ratio:.3f}`** | {ratio * 100:.1f}% of generated tokens spent reasoning |",
        f"| **Quality / Time Efficiency** | **`{quality_per_time:.1f} pts`** | intelligence × time efficiency × token economy (0-100 scale) |" if quality_per_time is not None else "",
        "",
    ])

    lines.extend([
        "## Token Consumption",
        "",
        "| Phase | Input (prompt) | Output (completion) | Reasoning | Total |",
        "|---|---|---|---|---|",
    ])
    for label, r_in, r_out, r_reas in token_rows:
        lines.append(f"| {label} | {r_in:,} | {r_out:,} | {r_reas:,} | {r_in + r_out:,} |")
    lines.append(f"| **Total** | **{t_in:,}** | **{t_out:,}** | **{t_reas:,}** | **{t_in + t_out:,}** |")
    lines.append("")

    # Helper for suite tables
    def _add_suite_section(title: str, results: list[ScenarioResult], benchmark_label: str, details_label: str):
        if not results:
            return
        tot = len(results)
        cor = sum(1 for r in results if r.ok)
        failed = [f"{r.id} ({r.detail})" if r.detail else r.id for r in results if not r.ok]
        lines.extend([
            f"## {title}",
            "",
            "| Benchmark | Accuracy | Correct / Total | Details |",
            "|---|---|---|---|",
            f"| **{benchmark_label}** | **`{(cor / tot) * 100:.1f}%`** | {cor} / {tot} | {details_label} |",
            "",
            f"**Failed Scenarios:** `{', '.join(failed) if failed else 'none'}`",
            "",
        ])

    if tool_results:
        t_tot = len(tool_results)
        t_cor = sum(1 for r in tool_results if r.ok)
        sp = [r for r in tool_results if r.category in ("simple", "parallel", "complex_args", "no_tool", "distractor_tools")]
        sp_cor = sum(1 for r in sp if r.ok)
        ag = [r for r in tool_results if r.category in ("multi_turn", "error_recovery")]
        ag_cor = sum(1 for r in ag if r.ok)
        failed_tools = [r.id for r in tool_results if not r.ok]
        lines.extend([
            "## Tool-Calling & Agentic Evaluation (BFCL & tau-bench)",
            "",
            "| Category | Accuracy | Correct / Total | Details |",
            "|---|---|---|---|",
            f"| **Overall Tool Accuracy** | **`{(t_cor / t_tot) * 100:.1f}%`** | {t_cor} / {t_tot} | BFCL exact-match, distractor selection & multi-turn |",
        ])
        if sp:
            lines.append(f"| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`{(sp_cor / len(sp)) * 100:.1f}%`** | {sp_cor} / {len(sp)} | Tool selection, args, restraint, distractors & schemas |")
        if ag:
            lines.append(f"| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`{(ag_cor / len(ag)) * 100:.1f}%`** | {ag_cor} / {len(ag)} | Multi-step dependency chains & stateful rollback |")
        lines.extend(["", f"**Failed Scenarios:** `{', '.join(failed_tools) if failed_tools else 'none'}`", ""])

    _add_suite_section("Instruction Following (Google IFEval Hard)", ifeval_results, "IFEval Hard Constraints", "Multi-constraint conjunctions, JSON ranges, negative constraints")
    _add_suite_section("Math Reasoning (AIME & Competition Math)", gsm8k_results, "AIME / Competition Math", "Modular arithmetic, combinatorics, algebra & geometry proofs")
    _add_suite_section("PhD Science Reasoning (GPQA Diamond)", gpqa_results, "GPQA Diamond (Physics / Chem / Bio)", "Google-proof PhD-level deduction & domain reasoning")
    _add_suite_section("Code Intelligence (HumanEval+ Data Structures)", humaneval_results, "HumanEval+ Code & Data Structures", "LRUCache, MinStack, Trie, interval merging with test execution")
    _add_suite_section("Advanced Physics & Math Reasoning (CritPt)", critpt_results, "CritPt Competition Physics", "Phase transitions, relativistic Doppler, thermodynamics, harmonic oscillator")
    _add_suite_section("Frontier Multidisciplinary PhD Exam (Humanity's Last Exam)", hle_results, "Humanity's Last Exam (HLE)", "Game theory, algebraic topology, provability logic, black holes, genetics")
    _add_suite_section("Stateful Banking Agent (T3-Banking / tau-bench)", banking_results, "T3-Banking Agent", "Multi-turn bank DB mutations, fee waivers, card freeze & dispute workflows")
    _add_suite_section("White-Collar Economic Audits (GDPval-AA v2)", gdpval_results, "GDPval-AA v2 Workflows", "Balance sheet reconciliation, vendor SLA audit, SaaS metrics, payroll tax")
    _add_suite_section("Hallucination Restraint & Adversarial Traps (AA-Omniscience)", omniscience_results, "AA-Omniscience Traps", "Counterfactual false premises, fictional entities, precise scientific recall")
    _add_suite_section("Scientific Python Computing (SciCode)", scicode_results, "SciCode Scientific Programming", "Quantum purity, Lennard-Jones, RK4 integrator, diffusion & matrix math")
    _add_suite_section("Interactive CLI & Terminal Agent (Terminal-Bench v4.0)", terminal_results, "Terminal-Bench CLI Agent", "VFS log triage, nginx syntax repair, git merge conflict resolution, JSON migration")
    _add_suite_section("Long-Context Reasoning & Retrieval (AA-LCR)", lcr_results, "AA-LCR Long Context", "Multi-document timeline synthesis, incident root cause, procurement liability")

    if sweep_data:
        lines.extend([
            "## Concurrency Scaling",
            "",
            "| Concurrency | Throughput (tok/s) |",
            "|---|---|",
        ])
        for level, tps, *_ in sweep_data:
            lines.append(f"| {level} streams | {tps:.1f} tok/s |")
        lines.append("")

    lines.extend([
        "## Machine-Readable Metrics",
        "",
        "```",
        f"METRIC tokens_per_second={conc8_tps:.3f}",
        f"METRIC conc8_tps={conc8_tps:.3f}",
        f"METRIC conc4_tps={conc4_tps:.3f}",
        f"METRIC single_stream_tps={single_tps:.3f}",
        f"METRIC time_to_first_token_ms={ttft_ms:.3f}",
        f"METRIC total_duration_seconds={total_duration_s:.3f}",
    ])
    if composite_score is not None:
        lines.append(f"METRIC smart_composite_score={composite_score:.4f}")
    if aa_index_score is not None:
        lines.append(f"METRIC aa_intelligence_index={aa_index_score:.4f}")
    if tool_acc is not None:
        lines.append(f"METRIC tool_call_accuracy={tool_acc:.4f}")
    if ifeval_acc is not None:
        lines.append(f"METRIC ifeval_accuracy={ifeval_acc:.4f}")
    if gsm8k_acc is not None:
        lines.append(f"METRIC gsm8k_accuracy={gsm8k_acc:.4f}")
    if gpqa_acc is not None:
        lines.append(f"METRIC gpqa_accuracy={gpqa_acc:.4f}")
    if he_acc is not None:
        lines.append(f"METRIC humaneval_accuracy={he_acc:.4f}")
    if critpt_acc is not None:
        lines.append(f"METRIC critpt_accuracy={critpt_acc:.4f}")
    if hle_acc is not None:
        lines.append(f"METRIC hle_accuracy={hle_acc:.4f}")
    if banking_acc is not None:
        lines.append(f"METRIC banking_accuracy={banking_acc:.4f}")
    if gdpval_acc is not None:
        lines.append(f"METRIC gdpval_accuracy={gdpval_acc:.4f}")
    if omni_acc is not None:
        lines.append(f"METRIC omniscience_accuracy={omni_acc:.4f}")
    if scicode_acc is not None:
        lines.append(f"METRIC scicode_accuracy={scicode_acc:.4f}")
    if term_acc is not None:
        lines.append(f"METRIC terminal_accuracy={term_acc:.4f}")
    if lcr_acc is not None:
        lines.append(f"METRIC lcr_accuracy={lcr_acc:.4f}")
    if quality_per_time is not None:
        lines.append(f"METRIC quality_per_time={quality_per_time:.4f}")
    lines.extend([
        f"METRIC reasoning_ratio={ratio:.4f}",
        f"METRIC total_tokens={t_in + t_out}",
        "```",
        "",
    ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def _parse_frontmatter(content: str) -> dict | None:
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    yaml_text = parts[1]
    data: dict[str, Any] = {}
    for line in yaml_text.strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lower() in ("null", "none"):
                data[key] = None
                continue
            try:
                if "." in val:
                    data[key] = float(val)
                else:
                    data[key] = int(val)
            except ValueError:
                data[key] = val
    if "quant" not in data:
        match = re.search(r"\*\*Quantization:\*\*\s*`([^`]+)`", content)
        if match:
            data["quant"] = match.group(1).strip()
        else:
            data["quant"] = _detect_quant(data.get("model"))
    elif data["quant"] is not None:
        data["quant"] = str(data["quant"]).strip()
    if "thinking" not in data:
        match = re.search(r"\*\*Thinking Mode:\*\*\s*`([^`]+)`", content)
        if match:
            data["thinking"] = _normalize_thinking(match.group(1))
        elif "- **Thinking Mode:** `off`" in content:
            data["thinking"] = "off"
        elif "- **Thinking Mode:** `on`" in content:
            data["thinking"] = "auto"
    else:
        data["thinking"] = _normalize_thinking(data["thinking"])

    if "quality_per_time" not in data or data["quality_per_time"] is None:
        comp = data.get("smart_composite_score")
        dur = data.get("total_duration_seconds")
        toks = data.get("total_tokens")
        if comp is not None and dur is not None and toks is not None and float(dur) > 0 and float(toks) > 0:
            e_time = (600.0 / max(300.0, float(dur))) ** 0.5
            e_tok = (170000.0 / max(50000.0, float(toks))) ** 0.3
            data["quality_per_time"] = min(100.0, (float(comp) * 100.0) * e_time * e_tok)
    return data


def _format_quant(m: dict) -> str:
    raw = m.get("quant")
    if raw:
        return str(raw).strip()[:8]
    return _detect_quant(m.get("model"))[:8]


def _format_thinking(m: dict) -> str:
    raw = m.get("thinking")
    if raw is None:
        return "N/A"
    return _normalize_thinking(raw)


ALL_SUITE_METRIC_KEYS = (
    "tool_call_accuracy", "ifeval_accuracy", "gsm8k_accuracy", "gpqa_accuracy", "humaneval_accuracy",
    "critpt_accuracy", "hle_accuracy", "banking_accuracy", "gdpval_accuracy", "omniscience_accuracy",
    "scicode_accuracy", "terminal_accuracy", "lcr_accuracy"
)


def _count_evaluated_suites(m: dict) -> int:
    return sum(1 for k in ALL_SUITE_METRIC_KEYS if m.get(k) is not None)


def _print_leaderboard(results_dir: str, concurrency: int = 8) -> None:
    if not os.path.isdir(results_dir):
        return
    entries: list[dict] = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            data = _parse_frontmatter(content)
            if data and "model" in data:
                data["_filename"] = fname
                entries.append(data)
        except Exception:
            continue

    if not entries:
        return

    # Deduplicate by (model, device, quant, thinking) keeping the entry with highest composite score or aa index
    unique: dict[tuple[str, str, str, str], dict] = {}
    for e in entries:
        m_name = str(e.get("model") or "")
        d_name = str(e.get("device") or "")
        q_name = _format_quant(e)
        th_name = _format_thinking(e)
        key = (m_name, d_name, q_name, th_name)
        if key not in unique:
            unique[key] = e
        else:
            existing = unique[key]
            e_comp = float(e.get("smart_composite_score") or e.get("aa_intelligence_index") or e.get("tool_call_accuracy") or 0)
            ex_comp = float(existing.get("smart_composite_score") or existing.get("aa_intelligence_index") or existing.get("tool_call_accuracy") or 0)
            if e_comp >= ex_comp:
                unique[key] = e

    all_records = list(unique.values())
    def _smartest_sort_key(m: dict):
        comp = m.get("smart_composite_score")
        aa_idx = m.get("aa_intelligence_index")
        suites_cnt = _count_evaluated_suites(m)
        tps = float(m.get("tokens_per_second") or m.get("conc8_tps") or 0.0)

        # Pure intelligence ranking: prioritize holistic Composite Score, with AA-Index as tiebreaker
        primary_score = float(comp) if comp is not None else float(aa_idx or 0.0)
        secondary_score = float(aa_idx) if aa_idx is not None else float(comp or 0.0)
        return (primary_score, secondary_score, suites_cnt, tps)

    smartest = sorted(all_records, key=_smartest_sort_key, reverse=True)[:3]

    # Top 3 Fastest: by tokens_per_second desc, single_stream_tps desc, smart_composite_score desc
    fastest = sorted(
        all_records,
        key=lambda x: (
            float(x.get("tokens_per_second") or x.get("conc8_tps") or x.get("conc4_tps") or 0),
            float(x.get("conc4_tps") or 0),
            float(x.get("single_stream_tps") or 0),
            float(x.get("smart_composite_score") if x.get("smart_composite_score") is not None else (x.get("aa_intelligence_index") or 0)),
        ),
        reverse=True,
    )[:3]

    print()
    print("=" * 143)
    print("Top 3 Smartest Models (Composite Intelligence & Artificial Analysis Index)")
    print("=" * 143)
    print(f" {'#':<3} {'Model':<22} {'Engine':<10} {'Device':<12} {'Quant':<8} {'Composite':<11} {'AA-Index':<10} {'Q/Time':<8} {'Tool Acc':<10} {'GPQA':<8} {'HLE':<8} {'Thinking':<8} {'Tokens'}")
    print(f" {'-':<3} {'-'*22:<22} {'-'*10:<10} {'-'*12:<12} {'-'*8:<8} {'-'*11:<11} {'-'*10:<10} {'-'*8:<8} {'-'*10:<10} {'-'*8:<8} {'-'*8:<8} {'-'*8:<8} {'-'*6}")
    for i, m in enumerate(smartest, 1):
        model_str = str(m.get("model", "unknown"))[:22]
        eng_str = str(m.get("engine", "unknown"))[:10]
        dev_str = str(m.get("device", "unknown"))[:12]
        q_str = _format_quant(m)
        th_str = _format_thinking(m)
        raw_comp = m.get("smart_composite_score")
        comp_val = f"{float(raw_comp) * 100:.1f}%" if raw_comp is not None else "N/A"
        raw_aa = m.get("aa_intelligence_index")
        aa_val = f"{float(raw_aa) * 100:.1f}%" if raw_aa is not None else "N/A"
        q_time_val = f"{float(m['quality_per_time']):.1f} pts" if m.get("quality_per_time") is not None else "N/A"
        tool_acc = f"{float(m['tool_call_accuracy']) * 100:.1f}%" if m.get("tool_call_accuracy") is not None else "N/A"
        gpqa_acc = f"{float(m['gpqa_accuracy']) * 100:.1f}%" if m.get("gpqa_accuracy") is not None else "N/A"
        hle_acc = f"{float(m['hle_accuracy']) * 100:.1f}%" if m.get("hle_accuracy") is not None else "N/A"
        tok_str = _fmt_tokens(m.get("total_tokens"))
        print(f" {i:<3} {model_str:<22} {eng_str:<10} {dev_str:<12} {q_str:<8} {comp_val:<11} {aa_val:<10} {q_time_val:<8} {tool_acc:<10} {gpqa_acc:<8} {hle_acc:<8} {th_str:<8} {tok_str}")
    print()
    print("=" * 143)
    print("Top 3 Fastest Models (Generation Throughput: 8-Conc / 4-Conc / Single)")
    print("=" * 143)
    print(f" {'#':<3} {'Model':<22} {'Engine':<10} {'Device':<12} {'Quant':<8} {'8-Conc t/s':<13} {'4-Conc t/s':<13} {'Single t/s':<13} {'Composite':<11} {'AA-Index':<10} {'Q/Time':<8} {'Thinking':<8} {'Tokens'}")
    print(f" {'-':<3} {'-'*22:<22} {'-'*10:<10} {'-'*12:<12} {'-'*8:<8} {'-'*13:<13} {'-'*13:<13} {'-'*13:<13} {'-'*11:<11} {'-'*10:<10} {'-'*8:<8} {'-'*8:<8} {'-'*6}")
    for i, m in enumerate(fastest, 1):
        model_str = str(m.get("model", "unknown"))[:22]
        eng_str = str(m.get("engine", "unknown"))[:10]
        dev_str = str(m.get("device", "unknown"))[:12]
        q_str = _format_quant(m)
        th_str = _format_thinking(m)
        c8_val = m.get("conc8_tps") or m.get("tokens_per_second")
        c8_str = f"{float(c8_val):.1f} tok/s" if c8_val is not None else "N/A"
        c4_val = m.get("conc4_tps")
        c4_str = f"{float(c4_val):.1f} tok/s" if c4_val is not None else "N/A"
        s_val = m.get("single_stream_tps")
        s_str = f"{float(s_val):.1f} tok/s" if s_val is not None else "N/A"
        raw_comp = m.get("smart_composite_score")
        comp_val = f"{float(raw_comp) * 100:.1f}%" if raw_comp is not None else "N/A"
        raw_aa = m.get("aa_intelligence_index")
        aa_val = f"{float(raw_aa) * 100:.1f}%" if raw_aa is not None else "N/A"
        q_time_val = f"{float(m['quality_per_time']):.1f} pts" if m.get("quality_per_time") is not None else "N/A"
        tok_str = _fmt_tokens(m.get("total_tokens"))
        print(f" {i:<3} {model_str:<22} {eng_str:<10} {dev_str:<12} {q_str:<8} {c8_str:<13} {c4_str:<13} {s_str:<13} {comp_val:<11} {aa_val:<10} {q_time_val:<8} {th_str:<8} {tok_str}")
    print("=" * 143)

    def _find_champion(keys: list[str], detail_formatters: list[tuple[str, str]], mode: str = "pure_accuracy") -> str | None:
        best_model = None
        best_tuple = (-1.0, -1.0, -1.0)
        for r in all_records:
            vals = [float(r[k]) for k in keys if r.get(k) is not None]
            if vals:
                base_score = (sum(vals) / len(vals)) * (1.0 + 0.05 * (len(vals) - 1))
                if mode == "agentic_speed":
                    s_tps = float(r.get("single_stream_tps") or 30.0)
                    speed_factor = 0.60 + 0.40 * min(1.0, s_tps / 75.0)
                    score = base_score * speed_factor
                else:
                    score = base_score
                comp_val = float(r.get("smart_composite_score") or r.get("aa_intelligence_index") or 0.0)
                tps_val = float(r.get("tokens_per_second") or r.get("conc8_tps") or 0.0)
                # Deterministic tiebreaker: primary domain score -> composite intelligence -> serving speed
                tie_tuple = (round(score, 5), comp_val, tps_val)
                if tie_tuple > best_tuple:
                    best_tuple = tie_tuple
                    best_model = r
        if not best_model:
            return None
        details = []
        for key, label in detail_formatters:
            if best_model.get(key) is not None:
                val = float(best_model[key])
                if key in ("tokens_per_second", "conc8_tps", "conc4_tps", "single_stream_tps"):
                    details.append(f"{val:.1f} {label}")
                elif "quality" in key or "efficiency" in key:
                    details.append(f"{val:.1f} {label}")
                else:
                    details.append(f"{val * 100:.1f}% {label}")
        m_name = str(best_model.get("model", "unknown"))[:22]
        eng = str(best_model.get("engine", "unknown"))[:10]
        q = _format_quant(best_model)
        return f"{m_name:<22} [{eng:<8} {q:<6}] — {' • '.join(details)}"

    agentic_champ = _find_champion(["tool_call_accuracy", "banking_accuracy", "terminal_accuracy"], [("tool_call_accuracy", "Tool Acc"), ("banking_accuracy", "Banking"), ("terminal_accuracy", "Terminal")], mode="agentic_speed")
    science_champ = _find_champion(["gpqa_accuracy", "critpt_accuracy", "gsm8k_accuracy"], [("critpt_accuracy", "CritPt"), ("gpqa_accuracy", "GPQA"), ("gsm8k_accuracy", "AIME")], mode="pure_accuracy")
    reason_champ = _find_champion(["hle_accuracy", "gdpval_accuracy", "ifeval_accuracy"], [("ifeval_accuracy", "IFEval"), ("gdpval_accuracy", "GDPval"), ("hle_accuracy", "HLE")], mode="pure_accuracy")
    code_champ = _find_champion(["humaneval_accuracy", "scicode_accuracy"], [("humaneval_accuracy", "HumanEval+"), ("scicode_accuracy", "SciCode")], mode="pure_accuracy")
    speed_champ = _find_champion(["tokens_per_second", "conc8_tps"], [("tokens_per_second", "8-Conc tok/s"), ("single_stream_tps", "Single tok/s")], mode="pure_accuracy")
    eff_champ = _find_champion(["quality_per_time"], [("quality_per_time", "pts Quality/Time")], mode="pure_accuracy")
    print()
    print("=" * 143)
    print("Domain Excellence Champions & Badges")
    print("=" * 143)
    if agentic_champ:
        print(f" • [Agentic & Banking Master] : {agentic_champ}")
    if science_champ:
        print(f" • [Science & Physics Leader] : {science_champ}")
    if reason_champ:
        print(f" • [Frontier PhD Reasoning]   : {reason_champ}")
    if code_champ:
        print(f" • [Code Intelligence Leader] : {code_champ}")
    if speed_champ:
        print(f" • [Raw Throughput Speed King]: {speed_champ}")
    if eff_champ:
        print(f" • [Quality/Time Efficiency]  : {eff_champ}")
    print("=" * 143)


def _report(
    cfg: Config,
    single: list,
    conc4_reps: list,
    conc8_reps: list,
    tool_results: list[ScenarioResult] | None,
    ifeval_results: list[ScenarioResult] | None,
    gsm8k_results: list[ScenarioResult] | None,
    gpqa_results: list[ScenarioResult] | None,
    humaneval_results: list[ScenarioResult] | None,
    critpt_results: list[ScenarioResult] | None = None,
    hle_results: list[ScenarioResult] | None = None,
    banking_results: list[ScenarioResult] | None = None,
    gdpval_results: list[ScenarioResult] | None = None,
    omniscience_results: list[ScenarioResult] | None = None,
    scicode_results: list[ScenarioResult] | None = None,
    terminal_results: list[ScenarioResult] | None = None,
    lcr_results: list[ScenarioResult] | None = None,
    total_duration_s: float = 0.0,
    sweep_data: list | None = None,
) -> int:
    def tok_sum(rs):
        return sum(r.completion_tokens for r in rs if not r.error)

    def reas_sum(rs):
        return sum(r.reasoning_tokens for r in rs if not r.error)

    s_tok = tok_sum(single)
    s_el = sum(r.elapsed_s for r in single if not r.error)
    single_tps = s_tok / s_el if s_el > 0 else 0.0

    # 4-concurrent stats
    rep4_stats = []
    for results, wall in conc4_reps:
        tok = tok_sum(results)
        rep4_stats.append((tok / wall if wall > 0 else 0.0, tok, wall))
    rep4_stats.sort(key=lambda x: x[0])
    conc4_tps, c4_tok, conc4_wall = rep4_stats[len(rep4_stats) // 2]
    lo4 = min(x[0] for x in rep4_stats)
    hi4 = max(x[0] for x in rep4_stats)

    # 8-concurrent stats (primary)
    rep8_stats = []
    for results, wall in conc8_reps:
        tok = tok_sum(results)
        rep8_stats.append((tok / wall if wall > 0 else 0.0, tok, wall))
    rep8_stats.sort(key=lambda x: x[0])
    conc8_tps, c8_tok, conc8_wall = rep8_stats[len(rep8_stats) // 2]
    lo8 = min(x[0] for x in rep8_stats)
    hi8 = max(x[0] for x in rep8_stats)

    all_conc8 = [r for results, _ in conc8_reps for r in results]
    ttfts = [r.ttft_s for r in all_conc8 if r.ttft_s is not None]
    ttft_ms = (sum(ttfts) / len(ttfts) * 1000.0) if ttfts else 0.0

    # Accuracies
    def _calc_acc(rs):
        return (sum(1 for r in rs if r.ok) / len(rs)) if rs else None

    tool_acc = _calc_acc(tool_results)
    ifeval_acc = _calc_acc(ifeval_results)
    gsm8k_acc = _calc_acc(gsm8k_results)
    gpqa_acc = _calc_acc(gpqa_results)
    he_acc = _calc_acc(humaneval_results)
    critpt_acc = _calc_acc(critpt_results)
    hle_acc = _calc_acc(hle_results)
    banking_acc = _calc_acc(banking_results)
    gdpval_acc = _calc_acc(gdpval_results)
    omni_acc = _calc_acc(omniscience_results)
    scicode_acc = _calc_acc(scicode_results)
    term_acc = _calc_acc(terminal_results)
    lcr_acc = _calc_acc(lcr_results)

    # Composite intelligence score (mean of all evaluated suites)
    all_accuracies = [a for a in (tool_acc, ifeval_acc, gsm8k_acc, gpqa_acc, he_acc,
                                  critpt_acc, hle_acc, banking_acc, gdpval_acc,
                                  omni_acc, scicode_acc, term_acc, lcr_acc) if a is not None]
    composite_score = (sum(all_accuracies) / len(all_accuracies)) if all_accuracies else None

    # Artificial Analysis Index score (mean of the 9 AA suites)
    aa_accuracies = [a for a in (gpqa_acc, critpt_acc, hle_acc, banking_acc, gdpval_acc,
                                omni_acc, scicode_acc, term_acc, lcr_acc) if a is not None]
    aa_index_score = (sum(aa_accuracies) / len(aa_accuracies)) if aa_accuracies else None

    # Reasoning ratio across all runs
    all_eval_scenarios = (
        (tool_results or []) + (ifeval_results or []) + (gsm8k_results or []) +
        (gpqa_results or []) + (humaneval_results or []) + (critpt_results or []) +
        (hle_results or []) + (banking_results or []) + (gdpval_results or []) +
        (omniscience_results or []) + (scicode_results or []) +
        (terminal_results or []) + (lcr_results or [])
    )
    c_tok_total = sum(tok_sum(results) for results, _ in conc4_reps) + sum(tok_sum(results) for results, _ in conc8_reps)
    c_reas_total = sum(reas_sum(results) for results, _ in conc4_reps) + sum(reas_sum(results) for results, _ in conc8_reps)
    all_tok = s_tok + c_tok_total + sum(r.completion_tokens for r in all_eval_scenarios)
    all_reas = reas_sum(single) + c_reas_total + sum(r.reasoning_tokens for r in all_eval_scenarios)
    ratio = min(1.0, all_reas / all_tok) if all_tok else 0.0

    # Token accounting across the whole run (input / output / reasoning)
    tp1 = _sum_tokens(single)
    tp4 = _sum_tokens([r for results, _ in conc4_reps for r in results])
    tp8 = _sum_tokens([r for results, _ in conc8_reps for r in results])
    ev_in, ev_out, ev_reas = _sum_tokens(all_eval_scenarios)
    sw_in = sum(x[2] for x in sweep_data) if sweep_data else 0
    sw_out = sum(x[3] for x in sweep_data) if sweep_data else 0
    sw_reas = sum(x[4] for x in sweep_data) if sweep_data else 0
    total_in = tp1[0] + tp4[0] + tp8[0] + ev_in + sw_in
    total_out = tp1[1] + tp4[1] + tp8[1] + ev_out + sw_out
    total_reas = tp1[2] + tp4[2] + tp8[2] + ev_reas + sw_reas
    total_tokens = total_in + total_out

    # Quality / Time Efficiency Score: intelligence achieved scaled by wall-clock time & token conciseness (0-100 pts)
    quality_per_time = (
        min(100.0, (composite_score * 100.0) * ((600.0 / max(300.0, total_duration_s)) ** 0.5) * ((170000.0 / max(50000.0, total_tokens)) ** 0.3))
        if (composite_score is not None and total_tokens > 0 and total_duration_s > 0)
        else None
    )

    token_rows: list[tuple[str, int, int, int]] = [
        ("Throughput single (1x)", tp1[0], tp1[1], tp1[2]),
        (f"Throughput 4x ({len(conc4_reps)} reps)", tp4[0], tp4[1], tp4[2]),
        (f"Throughput {cfg.concurrency}x ({len(conc8_reps)} reps)", tp8[0], tp8[1], tp8[2]),
    ]
    if all_eval_scenarios:
        token_rows.append(("Intelligence suites", ev_in, ev_out, ev_reas))
    if sweep_data:
        token_rows.append(("Concurrency sweep", sw_in, sw_out, sw_reas))

    print()
    print("=" * 66)
    print("AGENTIC & INTELLIGENCE BENCHMARK SUMMARY")
    print("=" * 66)
    print(f"endpoint : {cfg.base_url}")
    print(f"engine   : {cfg.engine or 'auto'}")
    print(f"device   : {cfg.device}")
    print(f"model    : {cfg.model}")
    print(f"thinking : {'on' if cfg.enable_thinking else 'off'}")

    print()
    print("--- Throughput ---")
    print(f"throughput single stream : {single_tps:.2f} tok/s ({s_tok} tok)")
    print(f"throughput 4-concurrent  : {conc4_tps:.2f} tok/s (median of {len(conc4_reps)} reps; spread {lo4:.1f}-{hi4:.1f})")
    print(f"throughput 8-concurrent  : {conc8_tps:.2f} tok/s (median of {len(conc8_reps)} reps; spread {lo8:.1f}-{hi8:.1f})")
    print(f"mean TTFT (8-concurrent) : {ttft_ms:.1f} ms")
    print(f"total execution time     : {_fmt_time(total_duration_s)} ({total_duration_s:.1f}s)")
    print(f"reasoning ratio          : {ratio:.3f} ({ratio * 100:.1f}%)")
    print(f"tokens used              : {total_tokens:,} total ({total_in:,} input / {total_out:,} output / {total_reas:,} reasoning)")
    if quality_per_time is not None:
        print(f"quality / time efficiency: {quality_per_time:.1f} pts ({composite_score * 100:.1f}% / {total_tokens:,} tok / {_fmt_time(total_duration_s)})")

    print()
    print("--- Intelligence Breakdown ---")
    if composite_score is not None:
        print(f"Composite Intelligence   : {composite_score * 100:.1f}%")
    if aa_index_score is not None:
        print(f"Artificial Analysis Index: {aa_index_score * 100:.1f}%")
    if tool_acc is not None:
        print(f"Tool-Call Accuracy       : {sum(1 for r in tool_results if r.ok)}/{len(tool_results)} = {tool_acc * 100:.1f}%")
    if ifeval_acc is not None:
        print(f"IFEval (Hard Rules)      : {sum(1 for r in ifeval_results if r.ok)}/{len(ifeval_results)} = {ifeval_acc * 100:.1f}%")
    if gsm8k_acc is not None:
        print(f"AIME / Competition Math  : {sum(1 for r in gsm8k_results if r.ok)}/{len(gsm8k_results)} = {gsm8k_acc * 100:.1f}%")
    if gpqa_acc is not None:
        print(f"GPQA Diamond (Science)   : {sum(1 for r in gpqa_results if r.ok)}/{len(gpqa_results)} = {gpqa_acc * 100:.1f}%")
    if he_acc is not None:
        print(f"HumanEval+ (Data Struct) : {sum(1 for r in humaneval_results if r.ok)}/{len(humaneval_results)} = {he_acc * 100:.1f}%")
    if critpt_acc is not None:
        print(f"CritPt (Olympiad Physics): {sum(1 for r in critpt_results if r.ok)}/{len(critpt_results)} = {critpt_acc * 100:.1f}%")
    if hle_acc is not None:
        print(f"Humanity's Last Exam     : {sum(1 for r in hle_results if r.ok)}/{len(hle_results)} = {hle_acc * 100:.1f}%")
    if banking_acc is not None:
        print(f"T3-Banking (tau-bench)   : {sum(1 for r in banking_results if r.ok)}/{len(banking_results)} = {banking_acc * 100:.1f}%")
    if gdpval_acc is not None:
        print(f"GDPval-AA v2 (Workflows) : {sum(1 for r in gdpval_results if r.ok)}/{len(gdpval_results)} = {gdpval_acc * 100:.1f}%")
    if omni_acc is not None:
        print(f"AA-Omniscience (Traps)   : {sum(1 for r in omniscience_results if r.ok)}/{len(omniscience_results)} = {omni_acc * 100:.1f}%")
    if scicode_acc is not None:
        print(f"SciCode (Scientific Py)  : {sum(1 for r in scicode_results if r.ok)}/{len(scicode_results)} = {scicode_acc * 100:.1f}%")
    if term_acc is not None:
        print(f"Terminal-Bench v4.0 (CLI): {sum(1 for r in terminal_results if r.ok)}/{len(terminal_results)} = {term_acc * 100:.1f}%")
    if lcr_acc is not None:
        print(f"AA-LCR (Long Context)    : {sum(1 for r in lcr_results if r.ok)}/{len(lcr_results)} = {lcr_acc * 100:.1f}%")

    # Failures list
    all_failed = [r.id for r in all_eval_scenarios if not r.ok]
    starved = [r for r in all_eval_scenarios if getattr(r, "is_starved", False) or getattr(r, "reasoning_starved", False)]
    if starved:
        print()
        print("!" * 78)
        print(f"⚠️  WARNING: {len(starved)} scenario(s) suffered REASONING TOKEN STARVATION!")
        print("   The model burned max_tokens inside <think> and was cut off before emitting a full answer.")
        print(f"   Affected scenarios: {', '.join(r.id for r in starved)}")
        print("   Remedy: Increase --tool-max-tokens / --max-tokens, or use --no-thinking / --thinking low.")
        print("!" * 78)
    if all_failed:
        print(f"failed items             : {', '.join(all_failed)}")

    if sweep_data:
        print()
        print("=== Concurrency scaling (t/s) ===")
        for level, tps, *_ in sweep_data:
            print(f"  {level:>2} concurrent : {tps:7.1f} tok/s")

    if not cfg.no_record:
        saved_path = _save_report_md(
            cfg,
            single_tps,
            s_tok,
            conc4_tps,
            conc4_reps,
            lo4,
            hi4,
            conc8_tps,
            conc8_reps,
            lo8,
            hi8,
            ttft_ms,
            ratio,
            tool_results,
            ifeval_results,
            gsm8k_results,
            gpqa_results,
            humaneval_results,
            critpt_results,
            hle_results,
            banking_results,
            gdpval_results,
            omniscience_results,
            scicode_results,
            terminal_results,
            lcr_results,
            composite_score,
            aa_index_score,
            total_duration_s,
            sweep_data,
            token_rows,
            (total_in, total_out, total_reas),
            quality_per_time,
        )
        print()
        print(f"Results saved to {saved_path}")

    _print_leaderboard(cfg.results_dir, cfg.concurrency)

    print()
    print(f"METRIC tokens_per_second={conc8_tps:.3f}")
    print(f"METRIC conc8_tps={conc8_tps:.3f}")
    print(f"METRIC conc4_tps={conc4_tps:.3f}")
    print(f"METRIC single_stream_tps={single_tps:.3f}")
    print(f"METRIC time_to_first_token_ms={ttft_ms:.3f}")
    print(f"METRIC total_duration_seconds={total_duration_s:.3f}")
    if composite_score is not None:
        print(f"METRIC smart_composite_score={composite_score:.4f}")
    if aa_index_score is not None:
        print(f"METRIC aa_intelligence_index={aa_index_score:.4f}")
    if tool_acc is not None:
        print(f"METRIC tool_call_accuracy={tool_acc:.4f}")
    if ifeval_acc is not None:
        print(f"METRIC ifeval_accuracy={ifeval_acc:.4f}")
    if gsm8k_acc is not None:
        print(f"METRIC gsm8k_accuracy={gsm8k_acc:.4f}")
    if gpqa_acc is not None:
        print(f"METRIC gpqa_accuracy={gpqa_acc:.4f}")
    if he_acc is not None:
        print(f"METRIC humaneval_accuracy={he_acc:.4f}")
    if critpt_acc is not None:
        print(f"METRIC critpt_accuracy={critpt_acc:.4f}")
    if hle_acc is not None:
        print(f"METRIC hle_accuracy={hle_acc:.4f}")
    if banking_acc is not None:
        print(f"METRIC banking_accuracy={banking_acc:.4f}")
    if gdpval_acc is not None:
        print(f"METRIC gdpval_accuracy={gdpval_acc:.4f}")
    if omni_acc is not None:
        print(f"METRIC omniscience_accuracy={omni_acc:.4f}")
    if scicode_acc is not None:
        print(f"METRIC scicode_accuracy={scicode_acc:.4f}")
    if term_acc is not None:
        print(f"METRIC terminal_accuracy={term_acc:.4f}")
    if lcr_acc is not None:
        print(f"METRIC lcr_accuracy={lcr_acc:.4f}")
    print(f"METRIC reasoning_ratio={ratio:.4f}")
    print(f"METRIC total_tokens={total_tokens}")
    if quality_per_time is not None:
        print(f"METRIC quality_per_time={quality_per_time:.4f}")
    return 0


async def _sweep(client, tracker, live, cfg):
    out = []
    for level in (1, 2, 4, 8, 16):
        prompts = (THROUGHPUT_PROMPTS * (level // 4 + 1))[:level]
        tasks = []
        states = []
        for i, prompt in enumerate(prompts):
            st = tracker.add(f"w{level}-{i + 1}", f"sweep/{level}")
            states.append(st)
            messages = [{"role": "user", "content": prompt}]
            if cfg.system_prompt:
                messages.insert(0, {"role": "system", "content": cfg.system_prompt})
            tasks.append(_run_stream(client, st, tracker, live, messages, tools=None,
                                     max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                                     chat_template_kwargs=cfg.thinking_kwargs,
                                     reasoning_effort=cfg.reasoning_effort))
        t0 = time.monotonic()
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - t0
        total = sum(r.completion_tokens for r in results if not r.error)
        tps = total / wall if wall > 0 else 0.0
        prompt_t = sum(r.prompt_tokens for r in results if not r.error)
        reas_t = sum(r.reasoning_tokens for r in results if not r.error)
        out.append((level, tps, prompt_t, total, reas_t))
        for st in states:
            live.note_done(st)
    return out


async def _main(cfg: Config) -> int:
    t_start = time.monotonic()
    random.seed(cfg.seed)
    client = ChatClient(cfg.base_url, cfg.model or "", api_key=cfg.api_key)
    try:
        models, detected_engine, detected_quant = await client.check()
        if not cfg.engine:
            cfg.engine = detected_engine
            print(f"auto-detected engine: {cfg.engine}", file=sys.stderr)
        if not cfg.quant and detected_quant:
            cfg.quant = detected_quant
            print(f"auto-detected quantization: {cfg.quant}", file=sys.stderr)
    except Exception as exc:
        print(f"FATAL: cannot reach endpoint {cfg.base_url}: {exc}", file=sys.stderr)
        return 2
    if not cfg.model:
        if not models:
            print(f"FATAL: endpoint {cfg.base_url} serves no models", file=sys.stderr)
            await client.aclose()
            return 2
        cfg.model = models[0]
        client.model = models[0]
        print(f"auto-detected model: {cfg.model}", file=sys.stderr)
    tracker = Tracker()
    live = LiveUI(
        tracker=tracker,
        header=f"Agentic benchmark — {cfg.model} [{cfg.engine}] on {cfg.device} @ {cfg.base_url} (thinking={cfg.thinking}, temp={cfg.temperature}, seed={cfg.seed})",
        enabled=sys.stderr.isatty(),
    )
    live.start()
    try:
        print(
            f"benchmark: {cfg.model} [{cfg.engine}] on {cfg.device} @ {cfg.base_url} throughput=(1x, 4x, 8x) "
            f"max_tokens={cfg.max_tokens} thinking={'on' if cfg.enable_thinking else 'off'} seed={cfg.seed}",
            file=sys.stderr,
        )
        if not sys.stderr.isatty():
            print("hint: run in a terminal for the live trace panel", file=sys.stderr)

        # 1. Throughput: Single stream (1x)
        single, _ = await _throughput(client, tracker, live, cfg, concurrency=1)

        # 2. Throughput: 4-Concurrent (4x)
        conc4_reps = []
        for _ in range(max(1, cfg.repeats)):
            conc4_reps.append(await _throughput(client, tracker, live, cfg, concurrency=4))

        # 3. Throughput: 8-Concurrent (8x)
        conc8_reps = []
        for _ in range(max(1, cfg.repeats)):
            conc8_reps.append(await _throughput(client, tracker, live, cfg, concurrency=cfg.concurrency))

        # --- Core Evaluations ---
        tool_results = None
        if cfg.should_eval("tool"):
            tool_results = await _toolcalls(client, tracker, live, cfg)

        ifeval_results = None
        if cfg.should_eval("ifeval"):
            ifeval_results = await _run_ifeval(client, tracker, live, cfg)

        gsm8k_results = None
        if cfg.should_eval("gsm8k") or cfg.should_eval("aime"):
            gsm8k_results = await _run_gsm8k(client, tracker, live, cfg)

        gpqa_results = None
        if cfg.should_eval("gpqa"):
            gpqa_results = await _run_gpqa(client, tracker, live, cfg)

        humaneval_results = None
        if cfg.should_eval("humaneval"):
            humaneval_results = await _run_humaneval(client, tracker, live, cfg)

        # --- Artificial Analysis Intelligence Index Suites ---
        critpt_results = None
        if cfg.should_eval("critpt"):
            critpt_results = await _run_critpt(client, tracker, live, cfg)

        hle_results = None
        if cfg.should_eval("hle"):
            hle_results = await _run_hle(client, tracker, live, cfg)

        banking_results = None
        if cfg.should_eval("banking") or cfg.should_eval("t3-banking"):
            banking_results = await _run_banking(client, tracker, live, cfg)

        gdpval_results = None
        if cfg.should_eval("gdpval") or cfg.should_eval("gdpval-aa"):
            gdpval_results = await _run_gdpval(client, tracker, live, cfg)

        omniscience_results = None
        if cfg.should_eval("omniscience") or cfg.should_eval("omni"):
            omniscience_results = await _run_omniscience(client, tracker, live, cfg)

        scicode_results = None
        if cfg.should_eval("scicode"):
            scicode_results = await _run_scicode(client, tracker, live, cfg)

        terminal_results = None
        if cfg.should_eval("terminal") or cfg.should_eval("terminal-bench"):
            terminal_results = await _run_terminal(client, tracker, live, cfg)

        lcr_results = None
        if cfg.should_eval("lcr") or cfg.should_eval("aa-lcr"):
            lcr_results = await _run_lcr(client, tracker, live, cfg)

        sweep_data = await _sweep(client, tracker, live, cfg) if cfg.sweep else []
    finally:
        live.stop()
        await client.aclose()
    total_duration_s = time.monotonic() - t_start
    return _report(
        cfg, single, conc4_reps, conc8_reps,
        tool_results, ifeval_results, gsm8k_results, gpqa_results, humaneval_results,
        critpt_results, hle_results, banking_results, gdpval_results,
        omniscience_results, scicode_results, terminal_results, lcr_results,
        total_duration_s, sweep_data
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tool-eval-bench",
        description="High-Throughput (1x, 4x, 8x) + Frontier Agentic & Artificial Analysis Intelligence Index Benchmark for OpenAI-compatible (vLLM/SGLang/MLX/llama.cpp) endpoints.",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: http://192.168.1.5:8888 or BENCH_BASE_URL)")
    parser.add_argument("--model", default=None, help="Model name (auto-detected from endpoint if omitted)")
    parser.add_argument("--device", "--gpu", default=None, help="Hardware / GPU label (default: DGX-Spark / Apple-Silicon or BENCH_DEVICE)")
    parser.add_argument("--engine", default=None, help="Serving engine override (vLLM, SGLang, MLX, llama.cpp, Ollama)")
    parser.add_argument("--quant", "--quantization", default=None, help="Quantization label override (NVFP4, EXL3, FP8, AWQ, BF16)")
    parser.add_argument("--results-dir", default=None, help="Directory to save markdown reports (default: results)")
    parser.add_argument("--eval", default=None, dest="eval_suites", help="Evaluations: 'all', 'aa-index', 'core', or comma-separated ('tool,ifeval,gsm8k,gpqa,humaneval,critpt,hle,banking,gdpval,omniscience,scicode,terminal,lcr')")
    parser.add_argument("--concurrency", type=int, default=None, help="Concurrent streams for primary throughput tier (default: 8)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max generation tokens for throughput tests (default: 4096)")
    parser.add_argument("--tool-max-tokens", type=int, default=None, help="Max output tokens for intelligence suites (default: 4096)")
    parser.add_argument("--scenarios", type=int, default=None, dest="scenario_limit", help="Scenario limit per suite (0 = all)")
    parser.add_argument("--repeats", type=int, default=None, help="Concurrent throughput measurement rounds (default: 3, median reported)")
    parser.add_argument("--thinking", default=None, help="Thinking mode: 'off', 'low', 'medium', 'high', 'xhigh', 'auto' (default: auto)")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking blocks (alias for --thinking off)")
    parser.add_argument("--reasoning-effort", default=None, help="Reasoning effort level ('low', 'medium', 'high', 'xhigh')")
    parser.add_argument("--system-prompt", default=None, help="System message prepended to requests")
    parser.add_argument("--sweep", action="store_true", help="Run 1/2/4/8/16 concurrency scaling sweep")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for deterministic test suite (default: 42)")
    parser.add_argument("--api-key", default=None, help="Bearer token for authenticated endpoints")
    parser.add_argument("--no-record", action="store_true", help="Do not write report .md file to results/")

    args = parser.parse_args()

    cfg = Config()
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    if args.device:
        cfg.device = args.device
    if args.engine:
        cfg.engine = args.engine
    if args.quant:
        cfg.quant = args.quant
    if args.results_dir:
        cfg.results_dir = args.results_dir
    if args.eval_suites:
        cfg.eval_suites = args.eval_suites
    if args.concurrency is not None:
        cfg.concurrency = args.concurrency
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.tool_max_tokens is not None:
        cfg.tool_max_tokens = args.tool_max_tokens
    if args.scenario_limit is not None:
        cfg.scenario_limit = args.scenario_limit
    if args.repeats is not None:
        cfg.repeats = args.repeats
    if args.thinking:
        cfg.thinking = _normalize_thinking(args.thinking)
    elif args.no_thinking:
        cfg.thinking = "off"
    elif args.reasoning_effort:
        cfg.thinking = _normalize_thinking(args.reasoning_effort)
    if args.system_prompt:
        cfg.system_prompt = args.system_prompt
    if args.sweep:
        cfg.sweep = True
    if args.seed is not None:
        cfg.seed = args.seed
    if args.api_key:
        cfg.api_key = args.api_key
    if args.no_record:
        cfg.no_record = True

    try:
        return asyncio.run(_main(cfg))
    except KeyboardInterrupt:
        print("\naborted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
