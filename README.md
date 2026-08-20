# Agentic & Intelligence Benchmark — Throughput + Tool Calling + IFEval + GSM8K + HumanEval

A benchmark harness for **any OpenAI-compatible API** (local, self-hosted, or remote). If your server exposes standard `/v1/chat/completions` and `/v1/models`, it works out of the box — including:
- **vLLM** & **SGLang** (Linux / GPU servers)
- **MLX (`mlx-lm.server`)** (Apple Silicon native)
- **llama.cpp / GGUF (`llama-server`)**
- **Ollama** & **LM Studio**
- **TGI**, **Aphrodite Engine**, **TensorRT-LLM**
- **LiteLLM** proxies or authenticated cloud gateways

Measures **generation throughput (tokens/sec)** at 1× and N-way concurrency, plus a comprehensive **Composite Intelligence Score** spanning:
- **Tool Calling & Agentic Evaluation** (BFCL simple, parallel, complex schemas, no-tool restraint + $\tau$-bench multi-turn & error recovery)
- **Instruction Following** (Google IFEval verifiable constraints)
- **Multi-Step Math Reasoning** (GSM8K)
- **Code Intelligence & Execution** (HumanEval with sandboxed Python test execution)

Python + `uv`. Deterministic, offline workload with **live streamed thinking traces**.

---

## Quickstart

```bash
# 1. Install dependencies (creates .venv, resolves uv.lock)
uv sync

# 2. Run the full benchmark (defaults: single-stream + 8-concurrency, thinking on,
#    model auto-detected from the endpoint)
./tool-eval-bench --seed 42 --base-url http://192.168.1.5:8888
#    equivalent: uv run --frozen python -m benchmark.main

# 3. Live traces — run in a real terminal (TTY). You get a live panel:
#    per-request table (TTFT / reasoning tokens / answer tokens / t/s) plus a
#    scrolling trace of the model's actual streamed thinking and [TOOL] calls.
```

Default target: `http://192.168.1.5:8888`. Both the **model** and the **serving engine** (vLLM, SGLang, MLX, llama.cpp / GGUF, Ollama, LM Studio, TGI, etc.) are **auto-detected** from the endpoint; override with `--model` / `--engine` (or `BENCH_MODEL` / `BENCH_ENGINE`). Hardware device label defaults to `DGX-Spark` on Linux and auto-detects your Apple chip (e.g. `M4-Max`, `M3-Pro`) on macOS.

Typical run: ~3–4 minutes (Single 1×, 4-concurrent, and 8-concurrent throughput rounds, plus intelligence suites).

### Useful variants

```bash
# Target any local or remote server:
./tool-eval-bench --base-url http://localhost:8000                          # vLLM / SGLang default
./tool-eval-bench --base-url http://localhost:8080                          # MLX (mlx_lm.server) / llama.cpp
./tool-eval-bench --base-url http://localhost:11434                         # Ollama
./tool-eval-bench --base-url http://localhost:1234                          # LM Studio
./tool-eval-bench --base-url https://api.myserver.com/v1 --api-key sk-...  # Remote / Auth Gateway

# Fast mode: disable reasoning + tool-use system prompt → ~9× lower latency, same accuracy
./tool-eval-bench --no-thinking --system-prompt "You are a helpful assistant with access to tools. When the user asks for data you cannot know from training (weather, prices, flights, products), you MUST call the matching tool. Never say you cannot provide real-time data — call the tool instead."

# Run specific intelligence suites (e.g. only tools + math, or coding only)
./tool-eval-bench --eval tool,gsm8k
./tool-eval-bench --eval humaneval

# Concurrency scaling curve (1/2/4/8/16 streams) — shows GPU headroom beyond 4
./tool-eval-bench --sweep

# Quick sanity run (2 scenarios, small token budget, skip saving report to disk)
./tool-eval-bench --scenarios 2 --max-tokens 256 --tool-max-tokens 512 --no-record
```

### CLI options (`tool-eval-bench`)
```bash
./tool-eval-bench --seed 42 --base-url http://192.168.1.5:8888
```

| Flag | Default | Meaning |
|---|---|---|
| `--base-url` | `BENCH_BASE_URL` → `http://192.168.1.5:8888` | endpoint URL |
| `--api-key` | `BENCH_API_KEY` → none | bearer token for endpoints that require auth |
| `--device`, `--gpu` | `BENCH_DEVICE` → `DGX-Spark` | device/GPU name for result filenames |
| `--results-dir` | `BENCH_RESULTS_DIR` → `results` | directory where report `.md` files are saved |
| `--engine` | auto-detect (vLLM, SGLang, llama.cpp, Ollama, etc.) | serving engine name/override |
| `--model` | auto-detect from `/v1/models` | model id |
| `--eval` | `all` | evaluation suites: `all` or comma-separated (`tool,ifeval,gsm8k,humaneval`) |
| `--concurrency` | 8 | concurrent streams |
| `--tool-max-tokens` | 1536 | per-tool-turn cap |
| `--scenarios` | 0 (all) | scenario limit |
| `--repeats` | 3 | concurrent rounds; median reported |
| `--seed` | 42 | harness RNG seed (workload is fixed at temperature 0) |
| `--thinking` / `--no-thinking` | on | reasoning toggle |
| `--system-prompt` | none | prepended system message |
| `--sweep` | off | report 1/2/4/8/16 scaling curve |
| `--temperature` | 0.0 | sampling temperature |
| `--no-record` | off | skip saving result report to `results/` markdown file |

Precedence: CLI flags → `BENCH_*` env vars → defaults.

---

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_BASE_URL` | `http://192.168.1.5:8888` | OpenAI-compatible endpoint |
| `BENCH_API_KEY` | (none) | bearer token; set if the endpoint requires auth |
| `BENCH_DEVICE` | `DGX-Spark` | device/GPU label for result filenames |
| `BENCH_RESULTS_DIR` | `results` | directory to save benchmark `.md` reports |
| `BENCH_MODEL` | (auto-detect) | model id; unset → first model from `GET /v1/models` |
| `BENCH_EVAL` | `all` | active eval suites: `all` or comma-separated (`tool,ifeval,gsm8k,humaneval`) |
| `BENCH_CONCURRENCY` | `8` | concurrent streams for the headline throughput metric |
| `BENCH_TOOL_MAX_TOKENS` | `1536` | output-token cap per tool-call turn |
| `BENCH_SCENARIOS` | `0` (= all) | limit number of tool-call scenarios |
| `BENCH_REPEATS` | `3` | concurrent-throughput rounds; the median is reported (stability) |
| `BENCH_ENABLE_THINKING` | `true` | pass `chat_template_kwargs={"enable_thinking": …}` (Nemotron toggle) |
| `BENCH_SYSTEM_PROMPT` | (none) | prepended system message to every request |
| `BENCH_SWEEP` | `off` | also report the 1/2/4/8/16 concurrency scaling curve |
| `BENCH_SEED` | `42` | harness RNG seed (workload is fixed at temperature 0) |
| `BENCH_NO_RECORD` | `off` | do not save result report to `results/` markdown file |
| temperature | `0.0` | fixed (deterministic workload) |

---

## Reported metrics

Printed as `METRIC name=value` lines on stdout (consumed by automation); a
human-readable summary is printed above them.

| Metric | Meaning |
|---|---|
| `tokens_per_second` | **primary** — 8-concurrent aggregate throughput (median of `BENCH_REPEATS`), total output tokens incl. reasoning |
| `conc8_tps` | 8-concurrent throughput |
| `conc4_tps` | 4-concurrent throughput |
| `single_stream_tps` | single-stream (1×) throughput |
| `time_to_first_token_ms` | mean TTFT across concurrent rounds |
| `smart_composite_score` | **composite intelligence score** — average accuracy across all evaluated intelligence suites |
| `tool_call_accuracy` | tool calling accuracy (BFCL exact-match, restraint & tau-bench multi-turn) |
| `ifeval_accuracy` | instruction following accuracy (Google IFEval verifiable constraints) |
| `gsm8k_accuracy` | grade school math multi-step reasoning accuracy (GSM8K) |
| `humaneval_accuracy` | Python functional code generation accuracy with unit test execution (HumanEval) |
| `reasoning_ratio` | reasoning tokens / total output tokens |
## Results & Leaderboard

After each benchmark run:
1. **Result Markdown File**: Automatically saved to `results/<device>-<model>.md` (e.g. `results/DGX-Spark-Nemo-3.5-Lightning.md`). Each file contains structured YAML frontmatter, detailed performance tables, tool accuracy breakdown, and CI metrics.
2. **Terminal Leaderboard**: Scans all result files in `results/` and prints two live rankings comparing models and engines side-by-side:
   - **🏆 Top 3 Smartest Models** (ranked by Composite Intelligence Score)
   - **⚡ Top 3 Fastest Models** (ranked by generation throughput: 8-Conc / 4-Conc / Single)

```
===================================================================================================================
🏆 Top 3 Smartest Models (Composite Intelligence Score)
===================================================================================================================
 #   Model                    Engine       Device       Composite   Tool Acc   IFEval    GSM8K     HumanEval 
 -   ------------------------ ------------ ------------ ----------- ---------- --------- --------- ----------
 1   ornith-1.5-35b-a3b-nvfp4 vLLM         DGX-Spark    75.0%       100.0%     100.0%    100.0%    0.0%      
 2   Nemo-3.5-Lightning       SGLang       DGX-Spark    70.0%       70.0%      N/A       N/A       N/A       

===================================================================================================================
⚡ Top 3 Fastest Models (Generation Throughput: 8-Conc / 4-Conc / Single)
===================================================================================================================
 #   Model                    Engine       Device       8-Conc t/s    4-Conc t/s    Single t/s    Composite   Tool Acc
 -   ------------------------ ------------ ------------ ------------- ------------- ------------- ----------- ----------
 1   Nemo-3.5-Lightning       SGLang       DGX-Spark    196.1 tok/s   137.6 tok/s   61.5 tok/s    70.0%       70.0%
 2   ornith-1.5-35b-a3b-nvfp4 vLLM         DGX-Spark    146.8 tok/s   78.7 tok/s    34.4 tok/s    75.0%       100.0%
===================================================================================================================
```


*(The leaderboard is displayed in terminal only and is not written into the individual model report files.)*
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

- **Tool Calling & Agentic Evaluation** — modeled on the **Berkeley Function Calling Leaderboard (BFCL)** and **$\tau$-bench / GAIA**:
  - `simple` (8): single-call tool selection and argument extraction.
  - `parallel` (6): multi-tool calls in a single response turn.
  - `multi_turn` (6): multi-step execution loop against a deterministic sandbox.
  - `no_tool` (4): general knowledge / creative questions with tools present; tests tool hallucination restraint.
  - `error_recovery` (2): tool returns errors/not-found; tests if model self-corrects and adapts parameters.
  - `complex_args` (1): nested JSON arrays and objects with type validation.
- **Instruction Following (Google IFEval)** — deterministic verification of hard constraints:
  - Strict JSON schema adherence with nested type & key constraints.
  - Negative constraints (e.g. zero commas, forbidden words).
  - Keyword frequency requirements and exact paragraph counts without bullets.
  - Highlighting & tag wrappers (`<response>...</response>`, bold headers).
- **Math Reasoning (GSM8K)** — canonical grade-school multi-step arithmetic word problems with exact integer answer verification (`#### X` / `\boxed{X}`).
- **Code Intelligence (HumanEval)** — functional Python code generation executed inside a sandboxed Python subprocess against strict unit test assertions.
- **Throughput** — **LLMPerf**-style synthetic generation measuring single-stream and N-way concurrent tokens/sec with code generation prompts.
- **Determinism** — temperature 0, vendored fixtures, deterministic canned tool executor, sandboxed code execution, no live internet dependencies.

### File layout

```
tool-eval-bench          # CLI entrypoint (execs `uv run --frozen python -m benchmark.main "$@"`)
pyproject.toml, uv.lock  # uv project (httpx, rich)
results/                 # benchmark result reports (<device>-<model>.md)
benchmark/
  main.py                # orchestrator, metrics, leaderboard, METRIC output
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
