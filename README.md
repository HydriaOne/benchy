# benchy: Frontier Agentic & LLM Intelligence Benchmark

> **High-Throughput (1x, 4x, 8x) • BFCL & $\\tau$-bench • IFEval Hard • AIME Math • GPQA Diamond • HumanEval+**

A high-performance benchmark harness for **any OpenAI-compatible API** (local, self-hosted, or remote). If your server exposes standard `/v1/chat/completions` and `/v1/models`, it works out of the box — including:
- **vLLM** & **SGLang** (Linux / GPU servers)
- **MLX (`mlx-lm.server`)** (Apple Silicon native)
- **llama.cpp / GGUF (`llama-server`)**
- **Ollama** & **LM Studio**
- **TGI**, **Aphrodite Engine**, **TensorRT-LLM**
- **LiteLLM** proxies or authenticated cloud gateways

Measures **3-Tier Generation Throughput (tokens/sec)** (`Single (1x)`, `4-Concurrent (4x)`, and `8-Concurrent (8x)`), plus a comprehensive **Composite Intelligence Score** spanning 5 frontier evaluation suites:
1. **Tool Calling & Agentic Evaluation** (BFCL simple, parallel, complex schemas, distractor tools, no-tool restraint + $\tau$-bench / GAIA Level 2 multi-turn 4-step dependency chains & stateful rollback)
2. **Instruction Following** (Google IFEval Hard with 3–4 simultaneous negative/structural/format orthogonal constraints)
3. **Competition Math Reasoning** (AIME 2024/2025 & Olympiad math with 10+ step deductive reasoning and exact integer answers)
4. **PhD-Level Science Reasoning** (GPQA Diamond Google-proof multiple-choice questions in Physics, Organic Chemistry, Molecular Biology, and Thermodynamics)
5. **Code Intelligence & Execution** (HumanEval+ and LeetCode Stateful Data Structures — `LRUCache`, `MinStack`, `Trie`, `IntervalMerger` with sandboxed Python test execution)

Python + `uv`. Deterministic, offline workload with **live streamed thinking traces**.

---

## Quickstart

```bash
# 1. Install dependencies (creates .venv, resolves uv.lock)
uv sync

# 2. Run the full benchmark (defaults: single + 4x + 8x throughput, thinking auto,
#    model & engine auto-detected from the endpoint)
./tool-eval-bench --seed 42 --base-url http://192.168.1.5:8888
#    equivalent: uv run --frozen python -m benchmark.main

# 3. Live traces — run in a real terminal (TTY). You get a live panel:
#    per-request table (TTFT / reasoning tokens / answer tokens / t/s) plus a
#    scrolling trace of the model's actual streamed thinking and [TOOL] calls.
```

Default target: `http://192.168.1.5:8888`. Both the **model** and the **serving engine** (vLLM, SGLang, MLX, llama.cpp / GGUF, Ollama, LM Studio, TGI, etc.) are **auto-detected** from the endpoint; override with `--model` / `--engine` (or `BENCH_MODEL` / `BENCH_ENGINE`). Hardware device label defaults to `DGX-Spark` on Linux and auto-detects your Apple chip (e.g. `M4-Max`, `M3-Pro`) on macOS.

> **💡 Hardware Tip (DGX Spark / GB10):** If your benchmark numbers on a GB10 machine are ~2–2.5× lower than reported here, your GPU is likely in a known PMIC throttling state ("powercreep"). See [Hardware Health & Known DGX Spark Bug](#️-hardware-health-dgx-spark-normal-vs-powercreep-known-dgx-spark-bug) below.
Typical run: ~3–4 minutes for full throughput (Single, 4x, 8x) and all 5 intelligence suites.
### Useful variants

```bash
# Target any local or remote server:
./tool-eval-bench --base-url http://localhost:8000                          # vLLM / SGLang default
./tool-eval-bench --base-url http://localhost:8080                          # MLX (mlx_lm.server) / llama.cpp
./tool-eval-bench --base-url http://localhost:11434                         # Ollama
./tool-eval-bench --base-url http://localhost:1234                          # LM Studio
./tool-eval-bench --base-url https://api.myserver.com/v1 --api-key sk-...  # Remote / Auth Gateway

# Fast mode: disable reasoning (off) + tool-use system prompt → ~9× lower latency, same accuracy
./tool-eval-bench --no-thinking --system-prompt "You are a helpful assistant with access to tools. When the user asks for data you cannot know from training (weather, prices, flights, products), you MUST call the matching tool. Never say you cannot provide real-time data — call the tool instead."

# Set specific thinking / reasoning effort level (off, low, medium, high, xhigh, auto):
./tool-eval-bench --thinking medium
./tool-eval-bench --thinking high
./tool-eval-bench --reasoning-effort low
# Run specific intelligence suites (e.g. only tools + science, or coding only)
./tool-eval-bench --eval tool,gpqa
./tool-eval-bench --eval gsm8k,humaneval
./tool-eval-bench --eval ifeval

# Concurrency scaling curve (1/2/4/8/16 streams) — shows GPU headroom
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
| `--eval` | `all` | evaluation suites: `all` or comma-separated (`tool,ifeval,gsm8k,gpqa,humaneval`) |
| `--concurrency` | 8 | concurrent streams for primary tier |
| `--max-tokens` | 2048 | throughput generation token cap |
| `--tool-max-tokens` | 1536 | per-turn intelligence suite token cap |
| `--scenarios` | 0 (all) | scenario limit per suite |
| `--repeats` | 3 | concurrent rounds; median reported |
| `--seed` | 42 | harness RNG seed (workload is fixed at temperature 0) |
| `--thinking` | `auto` | thinking / reasoning mode: `off`, `low`, `medium`, `high`, `xhigh`, `auto` |
| `--no-thinking` | (off) | disable reasoning (alias for `--thinking off`) |
| `--reasoning-effort` | none | reasoning effort level (alias for `--thinking`) |
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
| `BENCH_ENGINE` | (auto-detect) | engine name (vLLM, SGLang, MLX, llama.cpp, etc.) |
| `BENCH_EVAL` | `all` | active eval suites: `all` or comma-separated (`tool,ifeval,gsm8k,gpqa,humaneval`) |
| `BENCH_CONCURRENCY` | `8` | concurrent streams for the headline throughput metric |
| `BENCH_MAX_TOKENS` | `2048` | throughput generation token cap |
| `BENCH_TOOL_MAX_TOKENS` | `1536` | output-token cap per intelligence test problem/turn |
| `BENCH_SCENARIOS` | `0` (= all) | limit number of scenarios per suite |
| `BENCH_REPEATS` | `3` | concurrent-throughput rounds; the median is reported (stability) |
| `BENCH_SYSTEM_PROMPT` | (none) | prepended system message to every request |
| `BENCH_THINKING` | `auto` | thinking level: `off`, `low`, `medium`, `high`, `xhigh`, `auto` (`BENCH_ENABLE_THINKING` also supported) |
| `BENCH_SEED` | `42` | harness RNG seed (workload is fixed at temperature 0) |
| `BENCH_NO_RECORD` | `off` | do not save result report to `results/` markdown file |
| `temperature` | `0.0` | fixed (deterministic workload) |

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
| `total_duration_seconds` | total benchmark execution time |
| `smart_composite_score` | **composite intelligence score** — unweighted average accuracy across all evaluated suites |
| `tool_call_accuracy` | tool calling accuracy (BFCL exact-match, distractor selection & tau-bench / GAIA multi-turn) |
| `ifeval_accuracy` | instruction following accuracy (Google IFEval Hard multi-constraint conjunctions) |
| `gsm8k_accuracy` | competition math reasoning accuracy (AIME 2024/2025 & Olympiad math) |
| `gpqa_accuracy` | PhD-level science reasoning accuracy (GPQA Diamond Physics/Chem/Bio) |
| `humaneval_accuracy` | Python functional code & data structures accuracy with unit test execution (HumanEval+) |
| `reasoning_ratio` | reasoning tokens / total output tokens |

---

## Results & Leaderboard

After each benchmark run:
1. **Result Markdown File**: Automatically saved to `results/<device>-<model>.md` (e.g. `results/DGX-Spark-Nemo-3.5-Lightning.md`). Each file contains structured YAML frontmatter, detailed performance tables, tool accuracy breakdown, and CI metrics.
2. **Terminal Leaderboard**: Scans all result files in `results/` and prints two live rankings comparing models and engines side-by-side:
- **🏆 Top 3 Smartest Models** (ranked by Composite Intelligence Score across all 5 suites)
- **⚡ Top 3 Fastest Models** (ranked by generation throughput: 8-Conc / 4-Conc / Single)

```
=============================================================================================================================
🏆 Top 3 Smartest Models (Composite Intelligence Score: Tool, IFEval, AIME Math, GPQA, HumanEval+)
=============================================================================================================================
 #   Model                  Engine     Device       Composite   Tool Acc   IFEval    AIME     GPQA     HumanEval+  Thinking
 -   ---------------------- ---------- ------------ ----------- ---------- --------- -------- -------- ----------- --------
 1   deepseek-v4-flash-0731 vLLM       DGX-Spark    79.3%       96.8%      83.3%     66.7%    83.3%    66.7%       auto
 2   ornith-1.5-35b-a3b-nvf vLLM       DGX-Spark    74.7%       90.3%      66.7%     66.7%    50.0%    100.0%      auto
 3   Nemo-3.5-Lightning     SGLang     DGX-Spark    46.9%       67.7%      16.7%     33.3%    66.7%    50.0%       auto
=============================================================================================================================
⚡ Top 3 Fastest Models (Generation Throughput: 8-Conc / 4-Conc / Single)
=============================================================================================================================
 #   Model                  Engine     Device       8-Conc t/s    4-Conc t/s    Single t/s    Composite   Tool Acc   Thinking
 -   ---------------------- ---------- ------------ ------------- ------------- ------------- ----------- ---------- --------
 1   Nemo-3.5-Lightning     SGLang     DGX-Spark    321.7 tok/s   235.4 tok/s   125.3 tok/s   46.9%       67.7%      auto
 2   ornith-1.5-35b-a3b-nvf vLLM       DGX-Spark    264.5 tok/s   171.9 tok/s   94.4 tok/s    74.7%       90.3%      auto
 3   deepseek-v4-flash-0731 vLLM       DGX-Spark    34.1 tok/s    34.6 tok/s    34.1 tok/s    79.3%       96.8%      auto
```

*(The leaderboard is displayed in terminal only and is not written into the individual model report files.)*

### ⚠️ Hardware Health: DGX Spark Normal vs. "Powercreep" (Known DGX Spark Bug)

> **⚠️ Warning / Tip for GB10 Users:**  
> If your benchmark numbers on a DGX Spark (NVIDIA GB10) machine do not match the numbers reported here (e.g. you observe ~112 tok/s instead of ~264 tok/s, or ~14.7s TTFT instead of ~6.4s), **this hardware anomaly is the most probable reason**.

Under certain operating conditions or power transients, the **DGX Spark (NVIDIA GB10)** can enter a silent hardware throttling state known in the community as **"powercreep"** (or clamped state). This is a known DGX Spark hardware bug / PMIC rail anomaly (you can search online for more details regarding NVIDIA DGX Spark GB10 power and SM clock clamping issues).

#### The Silent Failure Mode
When powercrept:
- Standard monitoring tools (`nvidia-smi`) misleadingly report normal GPU behavior (e.g. **96% GPU utilization**, **P0 power state**, and **zero thermal/power throttling flags** in NVML).
- Under real matrix compute or LLM serving load, the hardware silently clamps down:
  - **Streaming Multiprocessor (SM) Clocks:** Collapse from **~2,300–2,500 MHz** down to **~500–800 MHz** (average SM clock `< 1500 MHz`).
  - **Power Draw:** Collapses from **~80–95 W peak** down to **~10–20 W** (peak power `< 40 W`).
  - **Raw Tensor Compute:** Drops from **~95–100 TFLOP/s** down to **~23–36 TFLOP/s** (~**70% compute loss**).

#### Performance Comparison: Normal vs. Powercreep

The table below compares full benchmark runs of **`ornith-1.5-35b-a3b-nvfp4`** served on **vLLM** on the DGX Spark under healthy normal operation vs. the throttled powercreep state:

| Benchmark Metric | DGX Spark (Normal / Healthy) | DGX Spark (Powercreep / Clamped) | Delta / Impact |
|---|---|---|---|
| **8-Concurrent Throughput** | **`264.46 tok/s`** | **`112.97 tok/s`** | 🔻 **-57.3%** (2.34× slower) |
| **4-Concurrent Throughput** | **`171.90 tok/s`** | **`69.09 tok/s`** | 🔻 **-59.8%** (2.49× slower) |
| **Single-Stream Throughput** | **`94.39 tok/s`** | **`35.69 tok/s`** | 🔻 **-62.2%** (2.64× slower) |
| **Mean TTFT (8-Concurrent)** | **`6,419.5 ms`** (6.4s) | **`14,655.0 ms`** (14.7s) | 🔺 **+128.3%** (2.28× higher latency) |
| **Total Benchmark Wall-Clock** | **`5m 6.9s`** (306.9s) | **`12m 16.3s`** (736.3s) | 🔻 **+139.9%** (2.40× longer run) |
| **Raw FP16 Tensor Compute** | **`~95–100 TFLOP/s`** | **`~23–36 TFLOP/s`** | 🔻 **~70% raw compute collapse** |
| **SM Clock Speeds** | **`~2,300–2,500 MHz`** | **`~500–800 MHz`** | 🔻 Clamped below 1.5 GHz |
| **Peak GPU Power Draw** | **`~80–95 W`** | **`~10–20 W`** | 🔻 VRM / PMIC rail voltage drop |
| **Composite Intelligence Score** | **`74.7%`** | **`74.7%`** | Same model reasoning fidelity |
| **Tool Call Accuracy** | **`90.3%`** (28 / 31) | **`90.3%`** (28 / 31) | Unchanged (deterministic) |
| **Google IFEval Hard** | **`66.7%`** (4 / 6) | **`50.0%`** (3 / 6) | 🔻 Constraint truncation from latency |
| **AIME Math Reasoning** | **`66.7%`** (4 / 6) | **`66.7%`** (4 / 6) | Unchanged |
| **GPQA Diamond Science** | **`50.0%`** (3 / 6) | **`66.7%`** (4 / 6) | Multiple choice deduction |
| **HumanEval+ Code Exec** | **`100.0%`** (6 / 6) | **`100.0%`** (6 / 6) | Sandboxed unit tests pass |

#### How to Reset the State

> **⚠️ Crucial Note on Recovery:**  
> A standard soft reboot (`sudo reboot`) does **NOT** reset the PMIC / VRM voltage regulator state on the GB10 board because the power rails remain energized.  
> 
> **To restart the state:** Simply **power off** the machine completely and disconnect/unplug the power supply for **~10 minutes** to allow the power rails and onboard capacitors to fully discharge. Powering back on will return the DGX Spark to its normal state with full compute (~95–100 TFLOP/s) and full throughput (~264 tok/s) restored.
---

## Live output

On a TTY, `rich` renders:
- **header panel** — model, serving engine, endpoint, thinking mode, temperature, seed;
- **live request table** — per request: phase, status, TTFT, reasoning tokens (live character count), answer tokens, tokens/sec, tool calls; windowed with scrolling summary to fit any terminal size;
- **live trace panel** — the currently streaming request's reasoning, then its answer and `[TOOL] name({"arg": …})` calls as they are emitted in real time.

Non-TTY (captured / CI) runs emit one concise `[done] …` line per request on stderr; stdout stays clean for `METRIC` parsing.

---

## Benchmark design & provenance

1. **Tool Calling & Agentic Evaluation** — modeled on the **Berkeley Function Calling Leaderboard (BFCL)** and **$\tau$-bench / GAIA**:
   - `simple` (8): single-call tool selection and argument extraction.
   - `parallel` (6): multi-tool calls in a single response turn.
   - `multi_turn` (6): multi-step execution loop against a deterministic sandbox.
   - `distractor_tools` (2): supply 12+ distractor tools to test tool ambiguity and precision.
   - `no_tool` (4): general knowledge / creative questions with tools present; tests tool hallucination restraint.
   - `error_recovery` (2): tool returns errors/not-found; tests if model self-corrects, rolls back, and adapts parameters.
   - `complex_args` (1): nested JSON arrays and objects with type validation.
2. **Instruction Following (Google IFEval Hard)** — deterministic verification of hard constraints:
   - Strict JSON schema adherence with nested type & key constraints.
   - Negative constraints (zero commas, forbidden words, case constraints).
   - Keyword frequency requirements and exact paragraph bounds without bullets.
   - Markdown structure formatting and tag wrappers.
3. **Competition Math Reasoning (AIME & Olympiad)** — 6 competition-level math problems with exact integer answers (`#### X` / `\boxed{X}`): modular arithmetic ($7^{2026} \pmod{100}$), Diophantine equations, digit sum combinatorics, and geometry.
4. **PhD-Level Science Reasoning (GPQA Diamond)** — 6 Google-proof multiple-choice questions spanning Quantum Mechanics, Organic Chemistry, Molecular Biology (lac operon), and Statistical Mechanics.
5. **Code Intelligence (HumanEval+ & LeetCode Data Structures)** — Python class & data structure generation (`LRUCache`, `MinStack`, `Trie`, `IntervalMerger`, palindrome partitions) executed in a sandboxed Python subprocess against comprehensive test assertions.
6. **3-Tier Throughput (1x, 4x, 8x)** — **LLMPerf**-style synthetic generation measuring single-stream and concurrent tokens/sec with code generation prompts.
7. **Determinism** — temperature 0, vendored fixtures, deterministic tool executor, sandboxed code execution, zero live internet dependencies.

### File layout

```
tool-eval-bench          # CLI entrypoint (execs `uv run --frozen python -m benchmark.main "$@"`)
pyproject.toml, uv.lock  # uv project (httpx, rich)
results/                 # benchmark result reports (<device>-<model>.md)
benchmark/
  main.py                # orchestrator, metrics, leaderboard, METRIC output
  sglang_client.py       # async SSE client (reasoning_content + tool-call fragments + usage)
  scenarios.py           # vendored tool scenarios, tool schemas, deterministic executor, throughput prompts
  grade.py               # BFCL / IFEval / AIME / GPQA / HumanEval grading routines
  live.py                # rich live panel / windowed table / concise non-TTY logs
```

---

## Requirements

- Python ≥ 3.11, `uv`
- Network access to the model endpoint (default `192.168.1.5:8888`)

## License

MIT — see [LICENSE](LICENSE).
