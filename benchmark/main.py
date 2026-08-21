"""Benchmark orchestrator: throughput (1x + 4x + 8x), tool calling, IFEval, AIME Math, GPQA Diamond, and HumanEval+.

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
    grade_gpqa,
    grade_gsm8k,
    grade_humaneval,
    grade_ifeval,
    grade_no_tool,
    match_calls,
    parse_args,
)
from .live import LiveUI, ReqState, Tracker
from .scenarios import (
    GPQA_SCENARIOS,
    GSM8K_SCENARIOS,
    HUMANEVAL_SCENARIOS,
    IFEVAL_SCENARIOS,
    SCENARIOS,
    THROUGHPUT_PROMPTS,
    TOOLS,
    execute_tool,
)
from .sglang_client import ChatClient

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


@dataclass
class Config:
    base_url: str = field(default_factory=lambda: os.environ.get("BENCH_BASE_URL", DEFAULT_BASE_URL))
    model: str | None = field(default_factory=lambda: os.environ.get("BENCH_MODEL") or None)
    concurrency: int = field(default_factory=lambda: _env_int("BENCH_CONCURRENCY", 8))
    max_tokens: int = field(default_factory=lambda: _env_int("BENCH_MAX_TOKENS", 2048))
    tool_max_tokens: int = field(default_factory=lambda: _env_int("BENCH_TOOL_MAX_TOKENS", 1536))
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
        if self.thinking == "auto":
            return {"enable_thinking": True}
        return {"enable_thinking": True, "reasoning_effort": self.thinking}

    @property
    def reasoning_effort(self) -> str | None:
        if self.thinking in ("low", "medium", "high", "xhigh"):
            return self.thinking
        return None
    def should_eval(self, suite: str) -> bool:
        if self.eval_suites.lower() in ("all", "*"):
            return True
        active = [s.strip().lower() for s in self.eval_suites.split(",") if s.strip()]
        return suite.lower() in active


@dataclass
class ScenarioResult:
    id: str
    category: str
    ok: bool = False
    detail: str = ""
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    error: str = ""


def _on_chunk(state: ReqState, tracker: Tracker, live: LiveUI, chunk: dict) -> None:
    choices = chunk.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        rc = delta.get("reasoning_content")
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


def _grade(sc: dict, first_calls: list[dict], final_answer: str) -> tuple[bool, str]:
    cat = sc["category"]
    if cat == "no_tool":
        return grade_no_tool(first_calls, final_answer, sc.get("expected_answer"))
    if cat in ("simple", "parallel", "complex_args", "distractor_tools"):
        ok = match_calls(sc["expected_calls"], first_calls)
        return ok, "calls-ok" if ok else f"calls-mismatch got={len(first_calls)}/{len(sc['expected_calls'])}"
    # multi_turn, error_recovery
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
            if res.error:
                st.status = "error"
                return ScenarioResult(id=name, category=sc["category"], error=res.error,
                                      completion_tokens=tot_tokens, reasoning_tokens=tot_reasoning)
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
        ok, detail = _grade(sc, first_calls, final_answer)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category=sc["category"], ok=ok, detail=detail,
                              completion_tokens=tot_tokens, reasoning_tokens=tot_reasoning)
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


async def _toolcalls(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = SCENARIOS if cfg.scenario_limit <= 0 else SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _run_scenario)


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
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_ifeval(sc["rule_id"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="ifeval", ok=ok, detail=detail,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_ifeval(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = IFEVAL_SCENARIOS if cfg.scenario_limit <= 0 else IFEVAL_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_ifeval)


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
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_gsm8k(sc["expected_answer"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="aime", ok=ok, detail=detail,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_gsm8k(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = GSM8K_SCENARIOS if cfg.scenario_limit <= 0 else GSM8K_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_gsm8k)


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
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_gpqa(sc["expected_answer"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="gpqa", ok=ok, detail=detail,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_gpqa(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = GPQA_SCENARIOS if cfg.scenario_limit <= 0 else GPQA_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_gpqa)


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
                                  completion_tokens=res.completion_tokens,
                                  reasoning_tokens=res.reasoning_tokens)
        ok, detail = grade_humaneval(sc["prompt"], sc["test"], res.content)
        st.status = "ok" if ok else "fail"
        st.finish_reason = detail
        return ScenarioResult(id=name, category="humaneval", ok=ok, detail=detail,
                              completion_tokens=res.completion_tokens,
                              reasoning_tokens=res.reasoning_tokens)
    finally:
        live.note_done(st)


async def _run_humaneval(client, tracker, live, cfg) -> list[ScenarioResult]:
    scenarios = HUMANEVAL_SCENARIOS if cfg.scenario_limit <= 0 else HUMANEVAL_SCENARIOS[: cfg.scenario_limit]
    return await _run_eval_suite(client, tracker, live, cfg, scenarios, _one_humaneval)


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
    composite_score: float | None,
    total_duration_s: float = 0.0,
    sweep_data: list | None = None,
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
    tool_acc = (sum(1 for r in tool_results if r.ok) / len(tool_results)) if tool_results else None
    ifeval_acc = (sum(1 for r in ifeval_results if r.ok) / len(ifeval_results)) if ifeval_results else None
    gsm8k_acc = (sum(1 for r in gsm8k_results if r.ok) / len(gsm8k_results)) if gsm8k_results else None
    gpqa_acc = (sum(1 for r in gpqa_results if r.ok) / len(gpqa_results)) if gpqa_results else None
    he_acc = (sum(1 for r in humaneval_results if r.ok) / len(humaneval_results)) if humaneval_results else None

    lines = [
        "---",
        f'model: "{cfg.model}"',
        f'device: "{cfg.device}"',
        f'engine: "{cfg.engine or "auto"}"',
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
        f"tool_call_accuracy: {tool_acc:.4f}" if tool_acc is not None else "tool_call_accuracy: null",
        f"ifeval_accuracy: {ifeval_acc:.4f}" if ifeval_acc is not None else "ifeval_accuracy: null",
        f"gsm8k_accuracy: {gsm8k_acc:.4f}" if gsm8k_acc is not None else "gsm8k_accuracy: null",
        f"gpqa_accuracy: {gpqa_acc:.4f}" if gpqa_acc is not None else "gpqa_accuracy: null",
        f"humaneval_accuracy: {he_acc:.4f}" if he_acc is not None else "humaneval_accuracy: null",
        f"reasoning_ratio: {ratio:.4f}",
        "---",
        "",
        f"# Benchmark Report: {cfg.model} on {cfg.device}",
        "",
        f"- **Date:** {date_str}",
        f"- **Device / GPU:** `{cfg.device}`",
        f"- **Serving Engine:** `{cfg.engine or 'OpenAI-Compatible'}`",
        f"- **Endpoint:** `{cfg.base_url}`",
        f"- **Model:** `{cfg.model}`",
        f"- **Thinking Mode:** `{cfg.thinking}`",
        f"- **Total Execution Time:** **`{_fmt_time(total_duration_s)}`** ({total_duration_s:.1f}s)",
        f"- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `{len(conc8_reps)}`)",
        f"- **Seed:** `{cfg.seed}`",
    ]

    if composite_score is not None:
        lines.append(f"- **🧠 Composite Intelligence Score:** **`{composite_score * 100:.1f}%`**")

    lines.extend([
        "",
        "## ⚡ Throughput Performance",
        "",
        "| Metric | Value | Details |",
        "|---|---|---|",
        f"| **8-Concurrent Throughput** | **`{conc8_tps:.2f} tok/s`** | median of {len(conc8_reps)} reps (spread: {lo8:.1f}–{hi8:.1f} tok/s) |",
        f"| **4-Concurrent Throughput** | **`{conc4_tps:.2f} tok/s`** | median of {len(conc4_reps)} reps (spread: {lo4:.1f}–{hi4:.1f} tok/s) |",
        f"| **Single-Stream Throughput** | **`{single_tps:.2f} tok/s`** | {s_tok} tokens generated |",
        f"| **Mean TTFT (8-Concurrent)** | **`{ttft_ms:.1f} ms`** | time to first token |",
        f"| **Total Execution Time** | **`{_fmt_time(total_duration_s)}`** | total benchmark wall-clock time ({total_duration_s:.1f}s) |",
        f"| **Reasoning Ratio** | **`{ratio:.3f}`** | {ratio * 100:.1f}% of generated tokens spent reasoning |",
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
            "## 🛠️ Tool-Calling & Agentic Evaluation (BFCL & tau-bench)",
            "",
            "| Category | Accuracy | Correct / Total | Details |",
            "|---|---|---|---|",
            f"| **Overall Tool Accuracy** | **`{(t_cor / t_tot) * 100:.1f}%`** | {t_cor} / {t_tot} | BFCL exact-match, distractor selection & multi-turn |",
        ])
        if sp:
            lines.append(f"| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`{(sp_cor / len(sp)) * 100:.1f}%`** | {sp_cor} / {len(sp)} | Tool selection, args, restraint, distractors & schemas |")
        if ag:
            lines.append(f"| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`{(ag_cor / len(ag)) * 100:.1f}%`** | {ag_cor} / {len(ag)} | Multi-step dependency chains & stateful rollback |")
        lines.extend([
            "",
            f"**Failed Scenarios:** `{', '.join(failed_tools) if failed_tools else 'none'}`",
            "",
        ])

    if ifeval_results:
        i_tot = len(ifeval_results)
        i_cor = sum(1 for r in ifeval_results if r.ok)
        failed_ifeval = [f"{r.id} ({r.detail})" for r in ifeval_results if not r.ok]
        lines.extend([
            "## 📋 Instruction Following (Google IFEval Hard)",
            "",
            "| Benchmark | Accuracy | Correct / Total | Details |",
            "|---|---|---|---|",
            f"| **IFEval Hard Constraints** | **`{(i_cor / i_tot) * 100:.1f}%`** | {i_cor} / {i_tot} | Multi-constraint conjunctions, JSON ranges, negative constraints |",
            "",
            f"**Failed Constraints:** `{', '.join(failed_ifeval) if failed_ifeval else 'none'}`",
            "",
        ])

    if gsm8k_results:
        g_tot = len(gsm8k_results)
        g_cor = sum(1 for r in gsm8k_results if r.ok)
        failed_gsm8k = [f"{r.id} ({r.detail})" for r in gsm8k_results if not r.ok]
        lines.extend([
            "## 🔢 Math Reasoning (AIME & Competition Math)",
            "",
            "| Benchmark | Accuracy | Correct / Total | Details |",
            "|---|---|---|---|",
            f"| **AIME / Competition Math** | **`{(g_cor / g_tot) * 100:.1f}%`** | {g_cor} / {g_tot} | Modular arithmetic, combinatorics, algebra & geometry proofs |",
            "",
            f"**Failed Problems:** `{', '.join(failed_gsm8k) if failed_gsm8k else 'none'}`",
            "",
        ])

    if gpqa_results:
        gp_tot = len(gpqa_results)
        gp_cor = sum(1 for r in gpqa_results if r.ok)
        failed_gpqa = [f"{r.id} ({r.detail})" for r in gpqa_results if not r.ok]
        lines.extend([
            "## 🔬 PhD Science Reasoning (GPQA Diamond)",
            "",
            "| Benchmark | Accuracy | Correct / Total | Details |",
            "|---|---|---|---|",
            f"| **GPQA Diamond (Physics / Chem / Bio)** | **`{(gp_cor / gp_tot) * 100:.1f}%`** | {gp_cor} / {gp_tot} | Google-proof PhD-level deduction & domain reasoning |",
            "",
            f"**Failed Questions:** `{', '.join(failed_gpqa) if failed_gpqa else 'none'}`",
            "",
        ])

    if humaneval_results:
        h_tot = len(humaneval_results)
        h_cor = sum(1 for r in humaneval_results if r.ok)
        failed_he = [f"{r.id} ({r.detail})" for r in humaneval_results if not r.ok]
        lines.extend([
            "## 💻 Code Intelligence (HumanEval+ Data Structures)",
            "",
            "| Benchmark | Accuracy | Correct / Total | Details |",
            "|---|---|---|---|",
            f"| **HumanEval+ Code & Data Structures** | **`{(h_cor / h_tot) * 100:.1f}%`** | {h_cor} / {h_tot} | LRUCache, MinStack, Trie, interval merging with test execution |",
            "",
            f"**Failed Unit Tests:** `{', '.join(failed_he) if failed_he else 'none'}`",
            "",
        ])

    if sweep_data:
        lines.extend([
            "## 📈 Concurrency Scaling",
            "",
            "| Concurrency | Throughput (tok/s) |",
            "|---|---|",
        ])
        for level, tps in sweep_data:
            lines.append(f"| {level} streams | {tps:.1f} tok/s |")
        lines.append("")

    lines.extend([
        "## 📊 Machine-Readable Metrics",
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
    lines.extend([
        f"METRIC reasoning_ratio={ratio:.4f}",
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
    return data


def _format_thinking(m: dict) -> str:
    raw = m.get("thinking")
    if raw is None:
        return "N/A"
    return _normalize_thinking(raw)


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

    # Deduplicate by (model, device, thinking) keeping the entry with highest composite score or tool_call_accuracy
    unique: dict[tuple[str, str, str], dict] = {}
    for e in entries:
        m_name = str(e.get("model") or "")
        d_name = str(e.get("device") or "")
        th_name = _format_thinking(e)
        key = (m_name, d_name, th_name)
        if key not in unique:
            unique[key] = e
        else:
            existing = unique[key]
            e_comp = float(e.get("smart_composite_score") or e.get("tool_call_accuracy") or 0)
            ex_comp = float(existing.get("smart_composite_score") or existing.get("tool_call_accuracy") or 0)
            if e_comp >= ex_comp:
                unique[key] = e

    all_records = list(unique.values())

    # Top 3 Smartest: by smart_composite_score desc, tool_call_accuracy desc, tokens_per_second desc
    smartest = sorted(
        all_records,
        key=lambda x: (
            float(x.get("smart_composite_score") if x.get("smart_composite_score") is not None else (x.get("tool_call_accuracy") or 0)),
            float(x.get("tool_call_accuracy") or 0),
            float(x.get("tokens_per_second") or 0),
        ),
        reverse=True,
    )[:3]

    # Top 3 Fastest: by tokens_per_second desc, single_stream_tps desc, smart_composite_score desc
    fastest = sorted(
        all_records,
        key=lambda x: (
            float(x.get("tokens_per_second") or x.get("conc8_tps") or x.get("conc4_tps") or 0),
            float(x.get("conc4_tps") or 0),
            float(x.get("single_stream_tps") or 0),
            float(x.get("smart_composite_score") if x.get("smart_composite_score") is not None else (x.get("tool_call_accuracy") or 0)),
        ),
        reverse=True,
    )[:3]

    print()
    print("=" * 125)
    print("🏆 Top 3 Smartest Models (Composite Intelligence Score: Tool, IFEval, AIME Math, GPQA, HumanEval+)")
    print("=" * 125)
    print(f" {'#':<3} {'Model':<22} {'Engine':<10} {'Device':<12} {'Composite':<11} {'Tool Acc':<10} {'IFEval':<9} {'AIME':<8} {'GPQA':<8} {'HumanEval+':<11} {'Thinking'}")
    print(f" {'-':<3} {'-'*22:<22} {'-'*10:<10} {'-'*12:<12} {'-'*11:<11} {'-'*10:<10} {'-'*9:<9} {'-'*8:<8} {'-'*8:<8} {'-'*11:<11} {'-'*8}")
    for i, m in enumerate(smartest, 1):
        model_str = str(m.get("model", "unknown"))[:22]
        eng_str = str(m.get("engine", "unknown"))[:10]
        dev_str = str(m.get("device", "unknown"))[:12]
        th_str = _format_thinking(m)
        raw_comp = m.get("smart_composite_score") if m.get("smart_composite_score") is not None else m.get("tool_call_accuracy")
        comp_val = f"{float(raw_comp) * 100:.1f}%" if raw_comp is not None else "N/A"
        tool_acc = f"{float(m['tool_call_accuracy']) * 100:.1f}%" if m.get("tool_call_accuracy") is not None else "N/A"
        ifeval_acc = f"{float(m['ifeval_accuracy']) * 100:.1f}%" if m.get("ifeval_accuracy") is not None else "N/A"
        gsm8k_acc = f"{float(m['gsm8k_accuracy']) * 100:.1f}%" if m.get("gsm8k_accuracy") is not None else "N/A"
        gpqa_acc = f"{float(m['gpqa_accuracy']) * 100:.1f}%" if m.get("gpqa_accuracy") is not None else "N/A"
        he_acc = f"{float(m['humaneval_accuracy']) * 100:.1f}%" if m.get("humaneval_accuracy") is not None else "N/A"
        print(f" {i:<3} {model_str:<22} {eng_str:<10} {dev_str:<12} {comp_val:<11} {tool_acc:<10} {ifeval_acc:<9} {gsm8k_acc:<8} {gpqa_acc:<8} {he_acc:<11} {th_str}")
    print()
    print("=" * 125)
    print("⚡ Top 3 Fastest Models (Generation Throughput: 8-Conc / 4-Conc / Single)")
    print("=" * 125)
    print(f" {'#':<3} {'Model':<22} {'Engine':<10} {'Device':<12} {'8-Conc t/s':<13} {'4-Conc t/s':<13} {'Single t/s':<13} {'Composite':<11} {'Tool Acc':<10} {'Thinking'}")
    print(f" {'-':<3} {'-'*22:<22} {'-'*10:<10} {'-'*12:<12} {'-'*13:<13} {'-'*13:<13} {'-'*13:<13} {'-'*11:<11} {'-'*10:<10} {'-'*8}")
    for i, m in enumerate(fastest, 1):
        model_str = str(m.get("model", "unknown"))[:22]
        eng_str = str(m.get("engine", "unknown"))[:10]
        dev_str = str(m.get("device", "unknown"))[:12]
        th_str = _format_thinking(m)
        c8_val = m.get("conc8_tps") or m.get("tokens_per_second")
        c8_str = f"{float(c8_val):.1f} tok/s" if c8_val is not None else "N/A"
        c4_val = m.get("conc4_tps")
        c4_str = f"{float(c4_val):.1f} tok/s" if c4_val is not None else "N/A"
        s_val = m.get("single_stream_tps")
        s_str = f"{float(s_val):.1f} tok/s" if s_val is not None else "N/A"
        raw_comp = m.get("smart_composite_score") if m.get("smart_composite_score") is not None else m.get("tool_call_accuracy")
        comp_val = f"{float(raw_comp) * 100:.1f}%" if raw_comp is not None else "N/A"
        tool_acc = f"{float(m['tool_call_accuracy']) * 100:.1f}%" if m.get("tool_call_accuracy") is not None else "N/A"
        print(f" {i:<3} {model_str:<22} {eng_str:<10} {dev_str:<12} {c8_str:<13} {c4_str:<13} {s_str:<13} {comp_val:<11} {tool_acc:<10} {th_str}")
    print("=" * 125)


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
    tool_acc = (sum(1 for r in tool_results if r.ok) / len(tool_results)) if tool_results else None
    sp_items = [r for r in tool_results if r.category in ("simple", "parallel", "complex_args", "no_tool", "distractor_tools")] if tool_results else []
    sp_acc = (sum(1 for r in sp_items if r.ok) / len(sp_items)) if sp_items else None
    ag_items = [r for r in tool_results if r.category in ("multi_turn", "error_recovery")] if tool_results else []
    ag_acc = (sum(1 for r in ag_items if r.ok) / len(ag_items)) if ag_items else None

    ifeval_acc = (sum(1 for r in ifeval_results if r.ok) / len(ifeval_results)) if ifeval_results else None
    gsm8k_acc = (sum(1 for r in gsm8k_results if r.ok) / len(gsm8k_results)) if gsm8k_results else None
    gpqa_acc = (sum(1 for r in gpqa_results if r.ok) / len(gpqa_results)) if gpqa_results else None
    he_acc = (sum(1 for r in humaneval_results if r.ok) / len(humaneval_results)) if humaneval_results else None

    # Composite intelligence score (mean of all evaluated suites)
    smart_scores = [a for a in (tool_acc, ifeval_acc, gsm8k_acc, gpqa_acc, he_acc) if a is not None]
    composite_score = (sum(smart_scores) / len(smart_scores)) if smart_scores else None

    # Reasoning ratio across all runs
    all_eval_scenarios = (tool_results or []) + (ifeval_results or []) + (gsm8k_results or []) + (gpqa_results or []) + (humaneval_results or [])
    c_tok_total = sum(tok_sum(results) for results, _ in conc4_reps) + sum(tok_sum(results) for results, _ in conc8_reps)
    c_reas_total = sum(reas_sum(results) for results, _ in conc4_reps) + sum(reas_sum(results) for results, _ in conc8_reps)
    all_tok = s_tok + c_tok_total + sum(r.completion_tokens for r in all_eval_scenarios)
    all_reas = reas_sum(single) + c_reas_total + sum(r.reasoning_tokens for r in all_eval_scenarios)
    ratio = min(1.0, all_reas / all_tok) if all_tok else 0.0

    print()
    print("=" * 66)
    print("📊 AGENTIC & INTELLIGENCE BENCHMARK SUMMARY")
    print("=" * 66)
    print(f"endpoint : {cfg.base_url}")
    print(f"engine   : {cfg.engine or 'auto'}")
    print(f"device   : {cfg.device}")
    print(f"model    : {cfg.model}")
    print(f"thinking : {'on' if cfg.enable_thinking else 'off'}")

    print()
    print("--- ⚡ Throughput ---")
    print(f"throughput single stream : {single_tps:.2f} tok/s ({s_tok} tok)")
    print(f"throughput 4-concurrent  : {conc4_tps:.2f} tok/s (median of {len(conc4_reps)} reps; spread {lo4:.1f}-{hi4:.1f})")
    print(f"throughput 8-concurrent  : {conc8_tps:.2f} tok/s (median of {len(conc8_reps)} reps; spread {lo8:.1f}-{hi8:.1f})")
    print(f"mean TTFT (8-concurrent) : {ttft_ms:.1f} ms")
    print(f"total execution time     : {_fmt_time(total_duration_s)} ({total_duration_s:.1f}s)")
    print(f"reasoning ratio          : {ratio:.3f} ({ratio * 100:.1f}%)")

    print()
    print("--- 🧠 Intelligence Breakdown ---")
    if composite_score is not None:
        print(f"Composite Intelligence   : {composite_score * 100:.1f}%")
    if tool_acc is not None:
        t_cor = sum(1 for r in tool_results if r.ok)
        print(f"Tool-Call Accuracy       : {t_cor}/{len(tool_results)} = {tool_acc * 100:.1f}%")
        if sp_acc is not None:
            print(f"  single-turn restraint  : {sp_acc * 100:.1f}%")
        if ag_acc is not None:
            print(f"  agentic multi-turn     : {ag_acc * 100:.1f}%")
    if ifeval_acc is not None:
        i_cor = sum(1 for r in ifeval_results if r.ok)
        print(f"IFEval (Hard Rules)      : {i_cor}/{len(ifeval_results)} = {ifeval_acc * 100:.1f}%")
    if gsm8k_acc is not None:
        g_cor = sum(1 for r in gsm8k_results if r.ok)
        print(f"AIME / Competition Math  : {g_cor}/{len(gsm8k_results)} = {gsm8k_acc * 100:.1f}%")
    if gpqa_acc is not None:
        gp_cor = sum(1 for r in gpqa_results if r.ok)
        print(f"GPQA Diamond (Science)   : {gp_cor}/{len(gpqa_results)} = {gpqa_acc * 100:.1f}%")
    if he_acc is not None:
        h_cor = sum(1 for r in humaneval_results if r.ok)
        print(f"HumanEval+ (Data Struct) : {h_cor}/{len(humaneval_results)} = {he_acc * 100:.1f}%")

    # Failures list
    all_failed = [r.id for r in all_eval_scenarios if not r.ok]
    if all_failed:
        print(f"failed items             : {', '.join(all_failed)}")

    if sweep_data:
        print()
        print("=== Concurrency scaling (t/s) ===")
        for level, tps in sweep_data:
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
            composite_score,
            total_duration_s,
            sweep_data,
        )
        print()
        print(f"💾 Results saved to {saved_path}")

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
    print(f"METRIC reasoning_ratio={ratio:.4f}")
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
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - t0
        total = sum(r.completion_tokens for r in results if not r.error)
        tps = total / wall if wall > 0 else 0.0
        out.append((level, tps))
        for st in states:
            live.note_done(st)
    return out


async def _main(cfg: Config) -> int:
    t_start = time.monotonic()
    random.seed(cfg.seed)
    client = ChatClient(cfg.base_url, cfg.model or "", api_key=cfg.api_key)
    try:
        models, detected_engine = await client.check()
        if not cfg.engine:
            cfg.engine = detected_engine
            print(f"auto-detected engine: {cfg.engine}", file=sys.stderr)
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

        sweep_data = await _sweep(client, tracker, live, cfg) if cfg.sweep else []
    finally:
        live.stop()
        await client.aclose()
    total_duration_s = time.monotonic() - t_start
    return _report(cfg, single, conc4_reps, conc8_reps, tool_results, ifeval_results, gsm8k_results, gpqa_results, humaneval_results, total_duration_s, sweep_data)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tool-eval-bench",
        description="t/s (1x, 4x, 8x) + frontier intelligence benchmark (BFCL, tau-bench, IFEval Hard, AIME Math, GPQA Diamond, HumanEval+) for OpenAI-compatible endpoints",
    )
    parser.add_argument("--base-url", default=None, help="endpoint URL (default: BENCH_BASE_URL, else http://192.168.1.5:8888)")
    parser.add_argument("--api-key", default=None, help="bearer token for endpoints that require auth (default: BENCH_API_KEY)")
    parser.add_argument("--device", "--gpu", dest="device", default=None, help="device/GPU name for result filenames (default: BENCH_DEVICE or DGX-Spark)")
    parser.add_argument("--results-dir", default=None, help="directory to store result markdown files (default: results)")
    parser.add_argument("--engine", default=None, help="serving engine name/override (default: auto-detect e.g. vLLM, SGLang, llama.cpp)")
    parser.add_argument("--model", default=None, help="model id (default: auto-detect from /v1/models)")
    parser.add_argument("--eval", dest="eval_suites", default=None, help="evaluation suites to run: all (default) or comma-separated e.g. tool,ifeval,gsm8k,gpqa,humaneval")
    parser.add_argument("--concurrency", type=int, default=None, help="concurrency tier for primary metric (default 8)")
    parser.add_argument("--max-tokens", type=int, default=None, help="output-token cap for throughput prompts (default 2048)")
    parser.add_argument("--tool-max-tokens", type=int, default=None, help="output-token cap per eval problem/turn (default 1536)")
    parser.add_argument("--scenarios", type=int, default=None, help="limit scenarios per suite, 0=all (default 0)")
    parser.add_argument("--repeats", type=int, default=None, help="concurrent rounds; median is reported (default 3)")
    parser.add_argument("--seed", type=int, default=None, help="harness RNG seed (default 42; workload is fixed at temperature 0)")
    parser.add_argument(
        "--thinking",
        nargs="?",
        const="auto",
        default=None,
        choices=["off", "low", "medium", "high", "xhigh", "auto", "on"],
        help="thinking / reasoning mode: off, low, medium, high, xhigh, auto (default: auto)",
    )
    parser.add_argument(
        "--no-thinking",
        dest="no_thinking",
        action="store_true",
        default=None,
        help="disable reasoning (equivalent to --thinking off)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["off", "low", "medium", "high", "xhigh", "auto"],
        help="reasoning effort level (alias for --thinking)",
    )
    parser.add_argument("--system-prompt", default=None, help="prepend this system message to every request")
    parser.add_argument("--sweep", action="store_true", default=None, help="report 1/2/4/8/16 concurrency scaling curve")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (default 0.0)")
    parser.add_argument("--no-record", action="store_true", default=None, help="do not record/save results to results/ markdown file")
    args = parser.parse_args()

    cfg = Config()
    for attr, value in (
        ("base_url", args.base_url),
        ("model", args.model),
        ("device", args.device),
        ("engine", args.engine),
        ("results_dir", args.results_dir),
        ("eval_suites", args.eval_suites),
        ("api_key", args.api_key),
        ("concurrency", args.concurrency),
        ("max_tokens", args.max_tokens),
        ("tool_max_tokens", args.tool_max_tokens),
        ("scenario_limit", args.scenarios),
        ("repeats", args.repeats),
        ("seed", args.seed),

        ("system_prompt", args.system_prompt),
        ("sweep", args.sweep),
        ("temperature", args.temperature),
        ("no_record", args.no_record),
    ):
        if value is not None:
            setattr(cfg, attr, value)
    thinking_val = None
    if getattr(args, "no_thinking", False):
        thinking_val = "off"
    elif getattr(args, "reasoning_effort", None) is not None:
        thinking_val = _normalize_thinking(args.reasoning_effort)
    elif getattr(args, "thinking", None) is not None:
        thinking_val = _normalize_thinking(args.thinking)

    if thinking_val is not None:
        cfg.thinking = thinking_val

    try:
        return asyncio.run(_main(cfg))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
