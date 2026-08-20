"""Benchmark orchestrator: throughput (1x + Nx) and agentic tool calling.

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

from .grade import answer_correct, match_calls, parse_args
from .live import LiveUI, ReqState, Tracker
from .scenarios import SCENARIOS, THROUGHPUT_PROMPTS, TOOLS, execute_tool
from .sglang_client import ChatClient

DEFAULT_BASE_URL = "http://192.168.1.5:8888"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


@dataclass
class Config:
    base_url: str = field(default_factory=lambda: os.environ.get("BENCH_BASE_URL", DEFAULT_BASE_URL))
    model: str | None = field(default_factory=lambda: os.environ.get("BENCH_MODEL") or None)
    concurrency: int = field(default_factory=lambda: _env_int("BENCH_CONCURRENCY", 4))
    max_tokens: int = field(default_factory=lambda: _env_int("BENCH_MAX_TOKENS", 2048))
    tool_max_tokens: int = field(default_factory=lambda: _env_int("BENCH_TOOL_MAX_TOKENS", 1536))
    scenario_limit: int = field(default_factory=lambda: _env_int("BENCH_SCENARIOS", 0))
    temperature: float = 0.0
    enable_thinking: bool = field(default_factory=lambda: os.environ.get("BENCH_ENABLE_THINKING", "true").strip().lower() in ("1", "true", "yes", "on"))
    system_prompt: str | None = field(default_factory=lambda: os.environ.get("BENCH_SYSTEM_PROMPT") or None)
    repeats: int = field(default_factory=lambda: _env_int("BENCH_REPEATS", 3))
    sweep: bool = field(default_factory=lambda: os.environ.get("BENCH_SWEEP", "").strip().lower() in ("1", "true", "yes", "on"))
    seed: int = field(default_factory=lambda: _env_int("BENCH_SEED", 42))
    api_key: str | None = field(default_factory=lambda: os.environ.get("BENCH_API_KEY") or None)
    device: str = field(default_factory=lambda: os.environ.get("BENCH_DEVICE") or os.environ.get("BENCH_GPU") or "DGX-Spark")
    results_dir: str = field(default_factory=lambda: os.environ.get("BENCH_RESULTS_DIR", "results"))

    @property
    def thinking_kwargs(self) -> dict | None:
        return None if self.enable_thinking else {"enable_thinking": False}


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


async def _run_stream(client, state, tracker, live, messages, *, tools, max_tokens, temperature, chat_template_kwargs=None):
    state.started_at = time.monotonic()
    res = await client.stream(
        messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        chat_template_kwargs=chat_template_kwargs,
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


async def _throughput(client, tracker, live, cfg, concurrent):
    states: list[ReqState] = []
    tasks = []
    for i, prompt in enumerate(THROUGHPUT_PROMPTS):
        name = f"c{i + 1}" if concurrent else f"s{i + 1}"
        st = tracker.add(name, "throughput")
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
            )
        )
    t0 = time.monotonic()
    if concurrent:
        results = await asyncio.gather(*tasks)
    else:
        results = [await t for t in tasks]
    wall = time.monotonic() - t0
    for st in states:
        live.note_done(st)
    return results, wall


async def _run_scenario(client, tracker, live, cfg, sc):
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
        for turn in range(3):
            res = await _run_stream(
                client, st, tracker, live, messages,
                tools=tools, max_tokens=cfg.tool_max_tokens, temperature=cfg.temperature,
                chat_template_kwargs=cfg.thinking_kwargs,
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


def _grade(sc, first_calls, final_answer):
    cat = sc["category"]
    if cat in ("simple", "parallel"):
        ok = match_calls(sc["expected_calls"], first_calls)
        return ok, "calls-ok" if ok else f"calls-mismatch got={len(first_calls)}/{len(sc['expected_calls'])}"
    ok = answer_correct(sc["expected_answer"], final_answer)
    return ok, "answer-ok" if ok else "answer-mismatch"


async def _toolcalls(client, tracker, live, cfg):
    scenarios = SCENARIOS if cfg.scenario_limit <= 0 else SCENARIOS[: cfg.scenario_limit]
    results = []
    for sc in scenarios:
        results.append(await _run_scenario(client, tracker, live, cfg, sc))
    return results


def _sanitize_name(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def _save_report_md(
    cfg: Config,
    single_tps: float,
    s_tok: int,
    conc_tps: float,
    conc_reps: list,
    lo: float,
    hi: float,
    ttft_ms: float,
    total: int,
    correct: int,
    acc: float | None,
    sp_acc: float | None,
    ag_acc: float | None,
    ratio: float,
    failed: list[str],
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
    failed_str = ", ".join(failed) if failed else "none"

    lines = [
        "---",
        f'model: "{cfg.model}"',
        f'device: "{cfg.device}"',
        f'endpoint: "{cfg.base_url}"',
        f'date: "{date_iso}"',
        f"tokens_per_second: {conc_tps:.3f}",
        f"single_stream_tps: {single_tps:.3f}",
        f"time_to_first_token_ms: {ttft_ms:.3f}",
        f"tool_call_accuracy: {acc:.4f}" if acc is not None else "tool_call_accuracy: null",
        f"agentic_accuracy: {ag_acc:.4f}" if ag_acc is not None else "agentic_accuracy: null",
        f"reasoning_ratio: {ratio:.4f}",
        "---",
        "",
        f"# Benchmark Report: {cfg.model} on {cfg.device}",
        "",
        f"- **Date:** {date_str}",
        f"- **Device / GPU:** `{cfg.device}`",
        f"- **Endpoint:** `{cfg.base_url}`",
        f"- **Model:** `{cfg.model}`",
        f"- **Thinking Mode:** `{'on' if cfg.enable_thinking else 'off'}`",
        f"- **Concurrency:** `{cfg.concurrency}` (repeats: `{len(conc_reps)}`)",
        f"- **Seed:** `{cfg.seed}`",
        "",
        "## Throughput Performance",
        "",
        "| Metric | Value | Details |",
        "|---|---|---|",
        f"| **{cfg.concurrency}-Concurrent Throughput** | **`{conc_tps:.2f} tok/s`** | median of {len(conc_reps)} reps (spread: {lo:.1f}–{hi:.1f} tok/s) |",
        f"| **Single-Stream Throughput** | **`{single_tps:.2f} tok/s`** | {s_tok} tokens generated |",
        f"| **Mean TTFT (Concurrent)** | **`{ttft_ms:.1f} ms`** | time to first token |",
        f"| **Reasoning Ratio** | **`{ratio:.3f}`** | {ratio * 100:.1f}% of generated tokens spent reasoning |",
        "",
        "## Tool-Calling & Agentic Accuracy",
        "",
        "| Category | Accuracy | Correct / Total | Details |",
        "|---|---|---|---|",
    ]

    if acc is not None:
        lines.append(f"| **Overall Tool-Call Accuracy** | **`{acc * 100:.1f}%`** | {correct} / {total} | BFCL-style exact match & tau-bench evaluation |")
    if sp_acc is not None:
        lines.append(f"| **Simple + Parallel** | **`{sp_acc * 100:.1f}%`** | - | Single-call arg extraction + parallel multi-call |")
    if ag_acc is not None:
        lines.append(f"| **Agentic Multi-Turn** | **`{ag_acc * 100:.1f}%`** | - | Multi-step tool execution & final answer grading |")

    lines.extend([
        "",
        f"**Failed Scenarios:** `{failed_str}`",
        "",
    ])

    if sweep_data:
        lines.extend([
            "## Concurrency Scaling",
            "",
            "| Concurrency | Throughput (tok/s) |",
            "|---|---|",
        ])
        for level, tps in sweep_data:
            lines.append(f"| {level} streams | {tps:.1f} tok/s |")
        lines.append("")

    lines.extend([
        "## Machine-Readable Metrics",
        "",
        "```",
        f"METRIC tokens_per_second={conc_tps:.3f}",
        f"METRIC single_stream_tps={single_tps:.3f}",
        f"METRIC time_to_first_token_ms={ttft_ms:.3f}",
    ])
    if acc is not None:
        lines.append(f"METRIC tool_call_accuracy={acc:.4f}")
    if ag_acc is not None:
        lines.append(f"METRIC agentic_accuracy={ag_acc:.4f}")
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
    return data


def _print_leaderboard(results_dir: str) -> None:
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

    # Deduplicate by (model, device) keeping the highest scoring entry
    unique: dict[tuple[str, str], dict] = {}
    for e in entries:
        m_name = str(e.get("model") or "")
        d_name = str(e.get("device") or "")
        key = (m_name, d_name)
        if key not in unique:
            unique[key] = e
        else:
            existing = unique[key]
            e_score = (
                float(e.get("tool_call_accuracy") or 0),
                float(e.get("tokens_per_second") or 0),
            )
            ex_score = (
                float(existing.get("tool_call_accuracy") or 0),
                float(existing.get("tokens_per_second") or 0),
            )
            if e_score > ex_score:
                unique[key] = e

    all_records = list(unique.values())

    # Top 3 Smartest: tool_call_accuracy desc, agentic_accuracy desc, tokens_per_second desc
    smartest = sorted(
        all_records,
        key=lambda x: (
            float(x.get("tool_call_accuracy") or 0),
            float(x.get("agentic_accuracy") or 0),
            float(x.get("tokens_per_second") or 0),
        ),
        reverse=True,
    )[:3]

    # Top 3 Fastest: tokens_per_second desc, single_stream_tps desc, tool_call_accuracy desc
    fastest = sorted(
        all_records,
        key=lambda x: (
            float(x.get("tokens_per_second") or 0),
            float(x.get("single_stream_tps") or 0),
            float(x.get("tool_call_accuracy") or 0),
        ),
        reverse=True,
    )[:3]

    print()
    print("=" * 74)
    print("🏆 Top 3 Smartest Models (Tool Calling & Agentic Accuracy)")
    print("=" * 74)
    print(f" {'#':<3} {'Model':<26} {'Device':<14} {'Tool Acc':<11} {'Agentic Acc':<13} {'Throughput'}")
    print(f" {'-':<3} {'-'*26:<26} {'-'*14:<14} {'-'*11:<11} {'-'*13:<13} {'-'*10}")
    for i, m in enumerate(smartest, 1):
        model_str = str(m.get("model", "unknown"))[:26]
        dev_str = str(m.get("device", "unknown"))[:14]
        tool_acc = f"{float(m['tool_call_accuracy']) * 100:.1f}%" if m.get("tool_call_accuracy") is not None else "N/A"
        ag_acc = f"{float(m['agentic_accuracy']) * 100:.1f}%" if m.get("agentic_accuracy") is not None else "N/A"
        tps_val = f"{float(m['tokens_per_second']):.1f} tok/s" if m.get("tokens_per_second") is not None else "N/A"
        print(f" {i:<3} {model_str:<26} {dev_str:<14} {tool_acc:<11} {ag_acc:<13} {tps_val}")

    print()
    print("=" * 74)
    print("⚡ Top 3 Fastest Models (Generation Throughput)")
    print("=" * 74)
    print(f" {'#':<3} {'Model':<26} {'Device':<14} {'4-Conc t/s':<13} {'Single t/s':<13} {'Tool Acc'}")
    print(f" {'-':<3} {'-'*26:<26} {'-'*14:<14} {'-'*13:<13} {'-'*13:<13} {'-'*10}")
    for i, m in enumerate(fastest, 1):
        model_str = str(m.get("model", "unknown"))[:26]
        dev_str = str(m.get("device", "unknown"))[:14]
        c_tps = f"{float(m['tokens_per_second']):.1f} tok/s" if m.get("tokens_per_second") is not None else "N/A"
        s_tps = f"{float(m['single_stream_tps']):.1f} tok/s" if m.get("single_stream_tps") is not None else "N/A"
        tool_acc = f"{float(m['tool_call_accuracy']) * 100:.1f}%" if m.get("tool_call_accuracy") is not None else "N/A"
        print(f" {i:<3} {model_str:<26} {dev_str:<14} {c_tps:<13} {s_tps:<13} {tool_acc}")
    print("=" * 74)

def _report(cfg, single, conc_reps, sc_results, sweep_data=None) -> int:
    def tok_sum(rs):
        return sum(r.completion_tokens for r in rs if not r.error)

    def reas_sum(rs):
        return sum(r.reasoning_tokens for r in rs if not r.error)

    s_tok = tok_sum(single)
    s_el = sum(r.elapsed_s for r in single if not r.error)
    single_tps = s_tok / s_el if s_el > 0 else 0.0
    rep_stats = []
    for results, wall in conc_reps:
        tok = tok_sum(results)
        rep_stats.append((tok / wall if wall > 0 else 0.0, tok, wall))
    rep_stats.sort(key=lambda x: x[0])
    conc_tps, c_tok, conc_wall = rep_stats[len(rep_stats) // 2]
    lo = min(x[0] for x in rep_stats)
    hi = max(x[0] for x in rep_stats)
    all_conc = [r for results, _ in conc_reps for r in results]
    ttfts = [r.ttft_s for r in all_conc if r.ttft_s is not None]
    ttft_ms = (sum(ttfts) / len(ttfts) * 1000.0) if ttfts else 0.0

    total = len(sc_results)
    correct = sum(1 for r in sc_results if r.ok)
    acc = correct / total if total else None
    sp = [r for r in sc_results if r.category in ("simple", "parallel")]
    ag = [r for r in sc_results if r.category == "multi_turn"]
    sp_acc = (sum(1 for r in sp if r.ok) / len(sp)) if sp else None
    ag_acc = (sum(1 for r in ag if r.ok) / len(ag)) if ag else None

    c_tok_total = sum(tok_sum(results) for results, _ in conc_reps)
    c_reas_total = sum(reas_sum(results) for results, _ in conc_reps)
    all_tok = s_tok + c_tok_total + sum(r.completion_tokens for r in sc_results)
    all_reas = reas_sum(single) + c_reas_total + sum(r.reasoning_tokens for r in sc_results)
    ratio = min(1.0, all_reas / all_tok) if all_tok else 0.0

    print()
    print("=== Agentic Benchmark Summary ===")
    print(f"endpoint : {cfg.base_url}")
    print(f"device   : {cfg.device}")
    print(f"model    : {cfg.model}")
    print(f"thinking : {'on' if cfg.enable_thinking else 'off'}")
    print(f"throughput single stream : {single_tps:.2f} tok/s ({s_tok} tok)")
    print(f"throughput {cfg.concurrency}-concurrent : {conc_tps:.2f} tok/s (median of {len(conc_reps)} reps; spread {lo:.1f}-{hi:.1f})")
    print(f"mean TTFT (concurrent)   : {ttft_ms:.1f} ms")
    if acc is not None:
        print(f"tool-call accuracy ({total} scenarios) : {correct}/{total} = {acc:.4f}")
    if sp_acc is not None:
        print(f"  simple+parallel        : {sp_acc:.4f}")
    if ag_acc is not None:
        print(f"  agentic multi-turn     : {ag_acc:.4f}")
    print(f"reasoning ratio          : {ratio:.3f}")
    failed = [r.id for r in sc_results if not r.ok]
    if failed:
        print(f"failed scenarios         : {', '.join(failed)}")

    if sweep_data:
        print()
        print("=== Concurrency scaling (t/s) ===")
        for level, tps in sweep_data:
            print(f"  {level:>2} concurrent : {tps:7.1f} tok/s")


    saved_path = _save_report_md(
        cfg,
        single_tps,
        s_tok,
        conc_tps,
        conc_reps,
        lo,
        hi,
        ttft_ms,
        total,
        correct,
        acc,
        sp_acc,
        ag_acc,
        ratio,
        failed,
        sweep_data,
    )
    print()
    print(f"💾 Results saved to {saved_path}")

    _print_leaderboard(cfg.results_dir)
    print()
    print(f"METRIC tokens_per_second={conc_tps:.3f}")
    print(f"METRIC single_stream_tps={single_tps:.3f}")
    print(f"METRIC time_to_first_token_ms={ttft_ms:.3f}")
    if acc is not None:
        print(f"METRIC tool_call_accuracy={acc:.4f}")
    if ag_acc is not None:
        print(f"METRIC agentic_accuracy={ag_acc:.4f}")
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
                                     chat_template_kwargs=cfg.thinking_kwargs))
        t0 = time.monotonic()
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - t0
        total = sum(r.completion_tokens for r in results if not r.error)
        tps = total / wall if wall > 0 else 0.0
        out.append((level, tps))
        for st in states:
            live.note_done(st)
    return out

async def _main(cfg: Config) -> int:
    random.seed(cfg.seed)
    client = ChatClient(cfg.base_url, cfg.model or "", api_key=cfg.api_key)
    try:
        models = await client.check()
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
        header=f"Agentic benchmark — {cfg.model} on {cfg.device} @ {cfg.base_url} (thinking={'on' if cfg.enable_thinking else 'off'}, temp={cfg.temperature}, seed={cfg.seed})",
        enabled=sys.stderr.isatty(),
    )
    live.start()
    try:
        print(
            f"benchmark: {cfg.model} on {cfg.device} @ {cfg.base_url} concurrency={cfg.concurrency} "
            f"max_tokens={cfg.max_tokens} thinking={'on' if cfg.enable_thinking else 'off'} seed={cfg.seed}",
            file=sys.stderr,
        )
        if not sys.stderr.isatty():
            print("hint: run in a terminal for the live trace panel", file=sys.stderr)
        single, _ = await _throughput(client, tracker, live, cfg, concurrent=False)
        conc_reps = []
        for _ in range(max(1, cfg.repeats)):
            conc_reps.append(await _throughput(client, tracker, live, cfg, concurrent=True))
        sc_results = await _toolcalls(client, tracker, live, cfg)
        sweep_data = await _sweep(client, tracker, live, cfg) if cfg.sweep else []
    finally:
        live.stop()
        await client.aclose()
    return _report(cfg, single, conc_reps, sc_results, sweep_data)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tool-eval-bench",
        description="tokens/sec + tool-calling benchmark for OpenAI-compatible (SGLang) endpoints",
    )
    parser.add_argument("--base-url", default=None, help="endpoint URL (default: BENCH_BASE_URL, else http://192.168.1.5:8888)")
    parser.add_argument("--api-key", default=None, help="bearer token for endpoints that require auth (default: BENCH_API_KEY)")
    parser.add_argument("--device", "--gpu", dest="device", default=None, help="device/GPU name for result filenames (default: BENCH_DEVICE or DGX-Spark)")
    parser.add_argument("--results-dir", default=None, help="directory to store result markdown files (default: results)")
    parser.add_argument("--model", default=None, help="model id (default: auto-detect from /v1/models)")
    parser.add_argument("--concurrency", type=int, default=None, help="concurrent streams (default 4)")
    parser.add_argument("--max-tokens", type=int, default=None, help="output-token cap for throughput prompts (default 2048)")
    parser.add_argument("--tool-max-tokens", type=int, default=None, help="output-token cap per tool turn (default 1536)")
    parser.add_argument("--scenarios", type=int, default=None, help="tool-call scenarios to run, 0=all (default 0)")
    parser.add_argument("--repeats", type=int, default=None, help="concurrent rounds; median is reported (default 3)")
    parser.add_argument("--seed", type=int, default=None, help="harness RNG seed (default 42; workload is fixed at temperature 0)")
    parser.add_argument("--thinking", dest="thinking", action="store_true", default=None, help="enable model reasoning (default)")
    parser.add_argument("--no-thinking", dest="thinking", action="store_false", help="disable reasoning (fast mode)")
    parser.add_argument("--system-prompt", default=None, help="prepend this system message to every request")
    parser.add_argument("--sweep", action="store_true", default=None, help="report 1/2/4/8/16 concurrency scaling curve")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (default 0.0)")
    args = parser.parse_args()

    cfg = Config()
    for attr, value in (
        ("base_url", args.base_url),
        ("model", args.model),
        ("device", args.device),
        ("results_dir", args.results_dir),
        ("api_key", args.api_key),
        ("concurrency", args.concurrency),
        ("max_tokens", args.max_tokens),
        ("tool_max_tokens", args.tool_max_tokens),
        ("scenario_limit", args.scenarios),
        ("repeats", args.repeats),
        ("seed", args.seed),
        ("enable_thinking", args.thinking),
        ("system_prompt", args.system_prompt),
        ("sweep", args.sweep),
        ("temperature", args.temperature),
    ):
        if value is not None:
            setattr(cfg, attr, value)

    try:
        return asyncio.run(_main(cfg))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
