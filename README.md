# Agentic Benchmark — t/s + Tool Calling for OpenAI-compatible (SGLang) Endpoints

Benchmark harness for self-hosted models behind an OpenAI-compatible API (SGLang). Measures
**generation throughput (tokens/sec)** at single-stream and N-way concurrency, plus **agentic
tool-calling accuracy** (BFCL-style simple/parallel + τ-bench-style multi-turn), with **live
streamed traces** of the model's reasoning, answers, and tool calls.

Python + `uv`. Deterministic, offline workload (no runtime downloads; only the model endpoint
is contacted).

---

## Quickstart

```bash
# 1. Install dependencies (creates .venv, resolves uv.lock)
uv sync

# 2. Run the full benchmark (defaults: single-stream + 4-concurrency, thinking on,
#    model auto-detected from the endpoint)
./tool-eval-bench --seed 42 --base-url http://192.168.1.5:8888
#    equivalent: uv run --frozen python -m benchmark.main

# 3. Live traces — run in a real terminal (TTY). You get a live panel:
#    per-request table (TTFT / reasoning tokens / answer tokens / t/s) plus a
#    scrolling trace of the model's actual streamed thinking and [TOOL] calls.
```

Default target: `http://192.168.1.5:8888` (SGLang). The model is **auto-detected** from
`GET /v1/models`; override with `--model` or `BENCH_MODEL`.

Typical run: ~3–4 minutes (4× single-stream prompts, 3× 4-concurrent rounds, 20 tool-call
scenarios).

### Useful variants

```bash
# Fast mode: disable reasoning + tool-use system prompt → ~9× lower latency, same accuracy
BENCH_ENABLE_THINKING=false \
BENCH_SYSTEM_PROMPT="You are a helpful assistant with access to tools. When the user asks for data you cannot know from training (weather, prices, flights, products), you MUST call the matching tool. Never say you cannot provide real-time data — call the tool instead." \
uv run --frozen python -m benchmark.main

# Concurrency scaling curve (1/2/4/8/16 streams) — shows GPU headroom beyond 4
BENCH_SWEEP=1 uv run --frozen python -m benchmark.main

# Point at a different endpoint / model
BENCH_BASE_URL=http://192.168.1.5:8000 BENCH_MODEL=my-model uv run --frozen python -m benchmark.main

# Quick sanity run (2 scenarios, small token budget)
BENCH_SCENARIOS=2 BENCH_MAX_TOKENS=256 BENCH_TOOL_MAX_TOKENS=512 uv run --frozen python -m benchmark.main

# The same via CLI flags (flags override env vars)
./tool-eval-bench --no-thinking --sweep
./tool-eval-bench --base-url http://192.168.1.5:8000 --model my-model
./tool-eval-bench --scenarios 2 --max-tokens 256 --tool-max-tokens 512
```

### CLI options (`tool-eval-bench`)

```bash
./tool-eval-bench --seed 42 --base-url http://192.168.1.5:8888
```

| Flag | Default | Meaning |
|---|---|---|
| `--base-url` | `BENCH_BASE_URL` → `http://192.168.1.5:8888` | endpoint URL |
| `--api-key` | `BENCH_API_KEY` → none | bearer token for endpoints that require auth |
| `--model` | auto-detect from `/v1/models` | model id |
| `--concurrency` | 4 | concurrent streams |
| `--max-tokens` | 2048 | throughput output cap |
| `--tool-max-tokens` | 1536 | per-tool-turn cap |
| `--scenarios` | 0 (all) | scenario limit |
| `--repeats` | 3 | concurrent rounds; median reported |
| `--seed` | 42 | harness RNG seed (workload is fixed at temperature 0) |
| `--thinking` / `--no-thinking` | on | reasoning toggle |
| `--system-prompt` | none | prepended system message |
| `--sweep` | off | report 1/2/4/8/16 scaling curve |
| `--temperature` | 0.0 | sampling temperature |

Precedence: CLI flags → `BENCH_*` env vars → defaults.

---

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_BASE_URL` | `http://192.168.1.5:8888` | OpenAI-compatible endpoint |
| `BENCH_API_KEY` | (none) | bearer token; set if the endpoint requires auth |
| `BENCH_MODEL` | (auto-detect) | model id; unset → first model from `GET /v1/models` |
| `BENCH_CONCURRENCY` | `4` | concurrent streams for the headline throughput metric |
| `BENCH_MAX_TOKENS` | `2048` | output-token cap for throughput prompts |
| `BENCH_TOOL_MAX_TOKENS` | `1536` | output-token cap per tool-call turn |
| `BENCH_SCENARIOS` | `0` (= all) | limit number of tool-call scenarios |
| `BENCH_REPEATS` | `3` | concurrent-throughput rounds; the median is reported (stability) |
| `BENCH_ENABLE_THINKING` | `true` | pass `chat_template_kwargs={"enable_thinking": …}` (Nemotron toggle) |
| `BENCH_SYSTEM_PROMPT` | (none) | prepended system message to every request |
| `BENCH_SWEEP` | `off` | also report the 1/2/4/8/16 concurrency scaling curve |
| `BENCH_SEED` | `42` | harness RNG seed (workload is fixed at temperature 0) |
| temperature | `0.0` | fixed (deterministic workload) |

---

## Reported metrics

Printed as `METRIC name=value` lines on stdout (consumed by automation); a
human-readable summary is printed above them.

| Metric | Meaning |
|---|---|
| `tokens_per_second` | **primary** — 4-concurrent aggregate throughput (median of `BENCH_REPEATS`), total output tokens incl. reasoning |
| `single_stream_tps` | single-stream throughput |
| `time_to_first_token_ms` | mean TTFT across concurrent rounds |
| `tool_call_accuracy` | correct scenarios / total (20): simple+parallel graded on tool name + argument subset; multi-turn graded on final answer |
| `agentic_accuracy` | multi-turn agentic scenarios only |
| `reasoning_ratio` | reasoning tokens / total output tokens |

## Live output

On a TTY, `rich` renders:

- header panel — model, endpoint, thinking mode, temperature;
- a table — per request: phase, status, TTFT, reasoning tokens (live char count), answer
  tokens, tokens/sec, tool calls;
- a live trace panel — the currently streaming request's reasoning, then its answer and
  `[TOOL] name({"arg": …})` calls as they are emitted.

Non-TTY (captured) runs emit one concise `[done] …` line per request on stderr; stdout stays
clean for `METRIC` parsing.

---

## Benchmark design & provenance

- **Tool calling** — modeled on the **Berkeley Function Calling Leaderboard (BFCL)**: the model
  must emit a function call (name + JSON arguments) from a user turn; graded by tool-name match +
  argument subset match. Categories:
  - `simple` (8): single call, argument extraction;
  - `parallel` (6): multiple calls in one turn;
  - `multi_turn` (6, **τ-bench / GAIA style**): the harness executes tool calls against a
    deterministic sandbox, feeds results back, and grades the final answer.
- **Throughput** — modeled on **LLMPerf** synthetic-generation: fixed prompts, bounded output,
  tokens/sec measured at 1× and N× concurrency.
- **Determinism** — temperature 0, vendored fixtures, deterministic canned tool executor
  (weather/stock/flights/products/email/calculator), no wall-clock or network dependencies
  beyond the endpoint itself.
- The model's exact token counts (`completion_tokens`, `reasoning_tokens`) come from the
  streamed usage chunk SGLang appends at the end of each response.

### File layout

```
tool-eval-bench          # CLI entrypoint (execs `uv run --frozen python -m benchmark.main "$@"`)
pyproject.toml, uv.lock  # uv project (httpx, rich)
benchmark/
  main.py                # orchestrator, metrics, METRIC output
  sglang_client.py       # async SSE client (reasoning_content + tool-call fragments + usage)
  scenarios.py           # vendored tool scenarios, tool schemas, deterministic executor, throughput prompts
  grade.py               # BFCL-style call match + τ-bench-style outcome grading
  live.py                # rich live panel / concise non-TTY logs
```

---

## Measured results (Nemo-3.5-Lightning, SGLang @ 192.168.1.5:8888)

| Measurement | Value |
|---|---|
| single-stream t/s | ~56–63 |
| 4-concurrent t/s (median-of-3) | ~131–148 |
| 16-concurrent t/s (sweep) | ~280–317 |
| TTFT (4-concurrent) | ~230 ms |
| reasoning ratio | ~0.64–0.78 |
| tool-call accuracy | 0.70 (14/20) — simple 8/8, parallel 0/6, multi-turn 6/6 |
| agentic (multi-turn) accuracy | 1.00 |

### Key findings

1. **The GPU is not saturated at 4 concurrency.** Throughput scales ~linearly with streams
   (1→62, 2→93, 4→142, 8→196, 16→280 t/s) — memory-bandwidth-bound decode. The 4-concurrency
   spec is a workload choice, not a hardware ceiling: ~2× more t/s is available at 16 streams.
2. **Parallel tool calls are a hard limitation of this model.** It reasons *"call twice"* but
   emits exactly one tool call per turn — confirmed with cross-tool requests, explicit
   "call twice" instructions, and `tool_choice="required"`. Fixing it requires a server-side
   change (e.g. SGLang `--tool-call-parser`), not a prompt change. This alone caps accuracy at 0.70.
3. **Reasoning length is highly stochastic** (2× spread even for identical prompts).
   Open-ended "explain/summarize" prompts stochastically hit the 2048-token cap; the bundled
   code-generation prompts keep it bounded (~700–1200 tokens, `finish=stop`).
4. **Fast mode**: `enable_thinking=false` is ~9× faster to answer but tool calling becomes
   unreliable *without* a system prompt; with the tool-use prompt above it is reliable at the
   same 0.70 accuracy. Raw t/s drops (~78 at 4-conc, short bursts under-saturate the GPU).
5. **Measurement stability**: single-sample t/s swings ±7% (127–143). Median-of-3
   (`BENCH_REPEATS=3`) narrows it to ~±3%.

## Requirements

- Python ≥ 3.11, `uv`
- Network access to the model endpoint (default `192.168.1.5:8888`)

## License

MIT — see [LICENSE](LICENSE).
