# benchy: Frontier Agentic & Artificial Analysis Intelligence Index Benchmark

> **High-Throughput (1x, 4x, 8x) • BFCL & $\tau$-bench • IFEval Hard • AIME Math • GPQA Diamond • HumanEval+ • Artificial Analysis Index (CritPt, HLE, T3-Banking, GDPval-AA v2, AA-Omniscience, SciCode, Terminal-Bench, AA-LCR)**

A high-performance benchmark harness for **any OpenAI-compatible API** (local, self-hosted, or remote). If your server exposes standard `/v1/chat/completions` and `/v1/models`, it works out of the box — including:
- **vLLM** & **SGLang** (Linux / GPU servers)
- **MLX (`mlx-lm.server`)** (Apple Silicon native)
- **llama.cpp / GGUF (`llama-server`)**
- **Ollama** & **LM Studio**
- **TGI**, **Aphrodite Engine**, **TensorRT-LLM**
- **LiteLLM** proxies or authenticated cloud gateways

Measures **3-Tier Generation Throughput (tokens/sec)** (`Single (1x)`, `4-Concurrent (4x)`, and `8-Concurrent (8x)`), **Composite Intelligence Score**, and the full **Artificial Analysis Intelligence Index (AA-Index)** across 9 frontier evaluations:

1. **Tool Calling & Agentic Evaluation** (BFCL simple, parallel, complex schemas, distractor tools, no-tool restraint + $\tau$-bench / GAIA Level 2 multi-turn 4-step dependency chains & stateful rollback)
2. **Instruction Following** (Google IFEval Hard with 3–4 simultaneous negative/structural/format orthogonal constraints)
3. **Competition Math Reasoning** (AIME 2024/2025 & Olympiad math with 10+ step deductive reasoning and exact integer answers)
4. **PhD-Level Science Reasoning** (GPQA Diamond 12-question set spanning Quantum Mechanics, Organic Chemistry, Molecular Biology, and Thermodynamics)
5. **Code Intelligence & Execution** (HumanEval+ and LeetCode Stateful Data Structures — `LRUCache`, `MinStack`, `Trie`, `IntervalMerger` with sandboxed Python test execution)
6. **Advanced Theoretical Physics & Math** (**CritPt** — Relativistic Doppler, Landau phase transition exponents, LC resonance, Carnot entropy, Cooper pair flux quantization)
7. **Frontier Multidisciplinary PhD Exam** (**Humanity's Last Exam (HLE)** — Game theory Shapley values, algebraic topology Euler characteristics, Gödel-Löb provability logic, black hole ISCO, and population genetics)
8. **Stateful Banking Agent** (**T3-Banking / $\tau$-bench** — Multi-turn banking DB mutations, fee waiver disputes, card freeze/unfreeze, and insufficient fund guardrails)
9. **White-Collar Economic Audits** (**GDPval-AA v2** — Balance sheet reconciliation, cloud SLA breach penalty audits, SaaS cohort NRR/GRR retention matrices, payroll tax withholding)
10. **Hallucination Restraint & Traps** (**AA-Omniscience** — Adversarial counterfactuals, false premise traps, fictional entities, and precise SI physical constants)
11. **Scientific Python Programming** (**SciCode** — Quantum state purity, Lennard-Jones forces, RK4 integrators, and diffusion numerical algorithms with sandboxed test assertions)
12. **Interactive CLI & Terminal Agent** (**Terminal-Bench v4.0** — Simulated sandbox VFS with access log triage, nginx syntax repair, git merge conflict resolution, and JSON migration validation)
13. **Long-Context Reasoning & Retrieval** (**AA-LCR** — Multi-document needle retrieval, microservice root cause analysis, and procurement liability synthesis over distributed contexts)

Python + `uv`. Deterministic, offline workload with **live streamed thinking traces** and **full token accounting (input / output / reasoning)**.

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

> **Hardware Tip (DGX Spark / GB10):** If your benchmark numbers on a GB10 machine are ~2–2.5× lower than reported here, your GPU is likely in a known PMIC throttling state ("powercreep"). See [Hardware Health & Known DGX Spark Bug](#hardware-health-dgx-spark-normal-vs-powercreep-known-dgx-spark-bug) below.

---

### Useful variants & presets

```bash
# Run the full 9-Suite Artificial Analysis Intelligence Index:
./tool-eval-bench --eval aa-index

# Run the 5 Core Evaluation Suites (fast ~3 min):
./tool-eval-bench --eval core

# Run ALL 13 Evaluation Suites:
./tool-eval-bench --eval all

# Run specific suites (e.g. only agentic banking & terminal, or physics & science):
./tool-eval-bench --eval banking,terminal,scicode
./tool-eval-bench --eval gpqa,critpt,hle

# Target any local or remote server:
./tool-eval-bench --base-url http://localhost:8000                          # vLLM / SGLang default
./tool-eval-bench --base-url http://localhost:8080                          # MLX (mlx_lm.server) / llama.cpp
./tool-eval-bench --base-url http://localhost:11434                         # Ollama
./tool-eval-bench --base-url http://localhost:1234                          # LM Studio
./tool-eval-bench --base-url https://api.myserver.com/v1 --api-key sk-...  # Remote / Auth Gateway

# Fast mode: disable reasoning (off) + tool-use system prompt → ~9× lower latency, same accuracy:
./tool-eval-bench --no-thinking --system-prompt "You are a helpful assistant with access to tools. When the user asks for data you cannot know from training (weather, prices, flights, products), you MUST call the matching tool. Never say you cannot provide real-time data — call the tool instead."

# Set specific thinking / reasoning effort level (off, low, medium, high, xhigh, auto):
./tool-eval-bench --thinking medium
./tool-eval-bench --thinking high
./tool-eval-bench --reasoning-effort low

# Concurrency scaling curve (1/2/4/8/16 streams) — shows GPU headroom:
./tool-eval-bench --sweep

# Quick sanity run (2 scenarios, small token budget, skip saving report to disk):
./tool-eval-bench --scenarios 2 --max-tokens 256 --tool-max-tokens 512 --no-record
```

---

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
| `--quant`, `--quantization` | auto-detect (NVFP4, EXL3, FP8, AWQ, BF16, etc.) | model quantization label/override |
| `--model` | auto-detect from `/v1/models` | model id |
| `--eval` | `all` | evaluation suites: `all`, `aa-index`, `core`, or comma-separated (`tool,ifeval,gsm8k,gpqa,humaneval,critpt,hle,banking,gdpval,omniscience,scicode,terminal,lcr`) |
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
| `BENCH_QUANT` | (auto-detect) | model quantization label (NVFP4, EXL3, FP8, AWQ, BF16, etc.) |
| `BENCH_EVAL` | `all` | active eval suites: `all`, `aa-index`, `core`, or comma-separated (`tool,ifeval,gsm8k,gpqa,humaneval,critpt,hle,banking,gdpval,omniscience,scicode,terminal,lcr`) |
| `BENCH_CONCURRENCY` | `8` | concurrent streams for the headline throughput metric |
| `BENCH_MAX_TOKENS` | `2048` | throughput generation token cap |
| `BENCH_TOOL_MAX_TOKENS` | `1536` | output-token cap per intelligence test problem/turn |
| `BENCH_SCENARIOS` | `0` (= all) | limit number of scenarios per suite |
| `BENCH_REPEATS` | `3` | concurrent-throughput rounds; the median is reported (stability) |
| `BENCH_SYSTEM_PROMPT` | (none) | prepended system message to every request |
| `BENCH_THINKING` | `auto` | thinking level: `off`, `low`, `medium`, `high`, `xhigh`, `auto` (`BENCH_ENABLE_THINKING` also supported) |
| `BENCH_SEED` | `42` | harness RNG seed (workload is fixed at temperature 0) |
| `BENCH_NO_RECORD` | `off` | do not save result report to `results/` markdown file |

---

## Reported metrics

Printed as `METRIC name=value` lines on stdout (consumed by automation); a human-readable summary is printed above them.

| Metric | Meaning |
|---|---|
| `tokens_per_second` | **primary** — 8-concurrent aggregate throughput (median of `BENCH_REPEATS`), total output tokens incl. reasoning |
| `conc8_tps` | 8-concurrent throughput |
| `conc4_tps` | 4-concurrent throughput |
| `single_stream_tps` | single-stream (1×) throughput |
| `time_to_first_token_ms` | mean TTFT across concurrent rounds |
| `total_duration_seconds` | total benchmark execution time |
| `smart_composite_score` | **composite intelligence score** — unweighted average accuracy across all evaluated suites |
| `aa_intelligence_index` | **Artificial Analysis Intelligence Index** — average accuracy across the 9 AA-Index suites |
| `tool_call_accuracy` | tool calling accuracy (BFCL exact-match, distractor selection & tau-bench multi-turn) |
| `ifeval_accuracy` | instruction following accuracy (Google IFEval Hard multi-constraint conjunctions) |
| `gsm8k_accuracy` | competition math reasoning accuracy (AIME 2024/2025 & Olympiad math) |
| `gpqa_accuracy` | PhD-level science reasoning accuracy (GPQA Diamond Physics/Chem/Bio) |
| `humaneval_accuracy` | Python functional code & data structures accuracy with unit test execution (HumanEval+) |
| `critpt_accuracy` | competition physics & math reasoning accuracy (CritPt) |
| `hle_accuracy` | frontier multidisciplinary PhD reasoning accuracy (Humanity's Last Exam) |
| `banking_accuracy` | stateful banking agent accuracy (T3-Banking / tau-bench) |
| `gdpval_accuracy` | white-collar financial workflow audit accuracy (GDPval-AA v2) |
| `omniscience_accuracy` | hallucination restraint & false premise trap accuracy (AA-Omniscience) |
| `scicode_accuracy` | scientific Python numerical computing accuracy (SciCode) |
| `terminal_accuracy` | interactive CLI & bash agent accuracy (Terminal-Bench v4.0) |
| `lcr_accuracy` | long-context retrieval & reasoning accuracy (AA-LCR) |
| `reasoning_ratio` | reasoning tokens / total output tokens |
| `total_tokens` | total tokens consumed during the whole run (input + output, incl. reasoning); the report also carries `input_tokens` / `output_tokens` and a per-phase breakdown |
| `quality_per_time` | **quality/time efficiency** — overall intelligence achieved balanced by wall-clock duration and token conciseness: $\text{Composite (\%)} \times \left(\frac{600}{T}\right)^{0.5} \times \left(\frac{170\text{k}}{N}\right)^{0.3}$ (0–100 point scale) |

---

## Results & Leaderboard

After each benchmark run:
1. **Result Markdown File**: Automatically saved to `results/<device>-<model>.md` (e.g. `results/DGX-Spark-Nemo-3.5-Lightning.md`). Each file contains structured YAML frontmatter (incl. `total_tokens`, `input_tokens`, `output_tokens`), detailed performance tables, a per-phase **Token Consumption** breakdown, individual suite breakdowns, and CI metrics.
2. **Terminal Leaderboard**: Scans all result files in `results/` and prints two live rankings plus **Domain Excellence Champions & Badges** comparing models side-by-side:
   - **Top 3 Smartest Models** (ranked by Composite Intelligence Score & Artificial Analysis Index)
   - **Top 3 Fastest Models** (ranked by generation throughput: 8-Conc / 4-Conc / Single)
   - **Domain Excellence Champions & Badges** (Agentic, Science/Physics, Frontier Reasoning, Coding, Speed, and Quality/Time)

```
===============================================================================================================================================
Top 3 Smartest Models (Composite Intelligence & Artificial Analysis Index)
===============================================================================================================================================
 #   Model                  Engine     Device       Quant    Composite   AA-Index   Q/Time   Tool Acc   GPQA     HLE      Thinking Tokens
 -   ---------------------- ---------- ------------ -------- ----------- ---------- -------- ---------- -------- -------- -------- ------
 1   deepseek-v4-flash-0731 vLLM       DGX-Spark    EXL3     88.6%       89.8%      49.7 pts 93.1%      91.7%    90.0%    auto     152.0k
 2   Ling-3.0-flash-int4    SGLang     DGX-Spark    INT4     81.0%       88.1%      75.3 pts 93.1%      100.0%   90.0%    auto     172.0k
 3   ornith-1.5-35b-a3b-nvf vLLM       DGX-Spark    NVFP4    80.8%       80.8%      86.8 pts 89.7%      83.3%    70.0%    auto     180.1k

===============================================================================================================================================
Top 3 Fastest Models (Generation Throughput: 8-Conc / 4-Conc / Single)
===============================================================================================================================================
 #   Model                  Engine     Device       Quant    8-Conc t/s    4-Conc t/s    Single t/s    Composite   AA-Index   Q/Time   Thinking Tokens
 -   ---------------------- ---------- ------------ -------- ------------- ------------- ------------- ----------- ---------- -------- -------- ------
 1   Nemo-3.5-Lightning     SGLang     DGX-Spark    NVFP4    311.0 tok/s   214.9 tok/s   120.4 tok/s   71.3%       80.6%      74.1 pts auto     208.4k
 2   ornith-1.5-35b-a3b-nvf vLLM       DGX-Spark    NVFP4    260.8 tok/s   176.5 tok/s   93.9 tok/s    80.8%       80.8%      86.8 pts auto     180.1k
 3   Ling-3.0-flash-int4    SGLang     DGX-Spark    INT4     159.5 tok/s   141.0 tok/s   72.0 tok/s    81.0%       88.1%      75.3 pts auto     172.0k
===============================================================================================================================================

===============================================================================================================================================
Domain Excellence Champions & Badges
===============================================================================================================================================
 • [Agentic & Banking Master] : Ling-3.0-flash-int4    [SGLang   INT4  ] — 93.1% Tool Acc • 100.0% Banking • 66.7% Terminal
 • [Science & Physics Leader] : deepseek-v4-flash-0731 [vLLM     EXL3  ] — 100.0% CritPt • 91.7% GPQA • 83.3% AIME
 • [Frontier PhD Reasoning]   : deepseek-v4-flash-0731 [vLLM     EXL3  ] — 100.0% IFEval • 100.0% GDPval • 90.0% HLE
 • [Code Intelligence Leader] : ornith-1.5-35b-a3b-nvf [vLLM     NVFP4 ] — 100.0% HumanEval+ • 83.3% SciCode
 • [Raw Throughput Speed King]: Nemo-3.5-Lightning     [SGLang   NVFP4 ] — 311.0 8-Conc tok/s • 120.4 Single tok/s
 • [Quality/Time Efficiency]  : ornith-1.5-35b-a3b-nvf [vLLM     NVFP4 ] — 86.8 pts Quality/Time
===============================================================================================================================================
```

*(The leaderboard is displayed in terminal only and is not written into the individual model report files. The **Tokens** column reads `total_tokens` from each report's frontmatter; reports produced before this metric existed show `N/A`.)*

---

<a id="hardware-health-dgx-spark-normal-vs-powercreep-known-dgx-spark-bug"></a>
### Hardware Health: DGX Spark Normal vs. "Powercreep" (Known DGX Spark Bug)

> **Warning / Tip for GB10 Users:**  
> If your benchmark numbers on a DGX Spark (NVIDIA GB10) machine do not match the numbers reported here (e.g. you observe ~112 tok/s instead of ~264 tok/s, or ~14.7s TTFT instead of ~6.4s), **this hardware anomaly is the most probable reason**.

Under certain operating conditions or power transients, the **DGX Spark (NVIDIA GB10)** can enter a silent hardware throttling state known in the community as **"powercreep"** (or clamped state). This is a known DGX Spark hardware bug / PMIC rail anomaly.

#### The Silent Failure Mode
When powercrept:
- Standard monitoring tools (`nvidia-smi`) misleadingly report normal GPU behavior (e.g. **96% GPU utilization**, **P0 power state**, and **zero thermal/power throttling flags** in NVML).
- Under real matrix compute or LLM serving load, the hardware silently clamps down:
  - **Streaming Multiprocessor (SM) Clocks:** Collapse from **~2,300–2,500 MHz** down to **~500–800 MHz** (average SM clock `< 1500 MHz`).
  - **Power Draw:** Collapses from **~80–95 W peak** down to **~10–20 W** (peak power `< 40 W`).
  - **Raw Tensor Compute:** Drops from **~95–100 TFLOP/s** down to **~23–36 TFLOP/s** (~**70% compute loss**).

#### How to Reset the State
A standard soft reboot (`sudo reboot`) does **NOT** reset the PMIC / VRM voltage regulator state on the GB10 board because the power rails remain energized.  
> **Crucial Note on Recovery:**  

---

## Live output

On a TTY, `rich` renders:
- **header panel** — model, serving engine, endpoint, thinking mode, temperature, seed;
- **live request table** — per request: phase, status, TTFT, reasoning tokens (live character count), answer tokens, tokens/sec, tool calls; windowed with scrolling summary to fit any terminal size;
- **live trace panel** — the currently streaming request's reasoning, then its answer and `[TOOL] name({"arg": …})` calls as they are emitted in real time.

Non-TTY (captured / CI) runs emit one concise `[done] …` line per request on stderr; stdout stays clean for `METRIC` parsing.

---

## Benchmark design & provenance


> **Core Design Philosophy: Tokens/sec != Speed to Solution (Wall-Clock Latency & Token Economy)**  
> Raw output generation rate (tok/s) alone can be deeply misleading. If Model A generates at 120 tok/s but uses 3,000 tokens across 4 verbose turns, it takes 60 seconds. If Model B generates at 50 tok/s but is concise, precise, calls the tool in 1 turn, and uses 1,000 tokens, it finishes in 30 seconds.  
> **Model B generates less, calls tools less, and wins the clock.**  
> `benchy` captures this via the **`quality_per_time`** metric and awards Domain Excellence badges based on accuracy achieved per unit of wall-clock execution time and token footprint.

1. **Tool Calling & Agentic Evaluation** — modeled on **BFCL** and **$\tau$-bench / GAIA**:
   - `simple` (8): single-call tool selection and argument extraction.
   - `parallel` (6): multi-tool calls in a single response turn.
   - `multi_turn` (6): multi-step execution loop against a deterministic sandbox.
   - `distractor_tools` (2): supply 12+ distractor tools to test tool ambiguity and precision.
   - `no_tool` (4): general knowledge questions with tools present; tests tool hallucination restraint.
   - `error_recovery` (2): tool returns errors/not-found; tests self-correction and rollback.
   - `complex_args` (1): nested JSON arrays and objects with type validation.
2. **Instruction Following (Google IFEval Hard)** — deterministic verification of multi-constraint conjunctions (word counts, tag wrappers, forbidden words, table structures).
3. **Competition Math Reasoning (AIME & Olympiad)** — multi-step deductive math with exact integer answers (`#### X` / `\boxed{X}`).
4. **PhD-Level Science Reasoning (GPQA Diamond)** — 12 Google-proof multiple-choice questions in Physics, Organic Chemistry, Molecular Biology, and Thermodynamics.
5. **Code Intelligence (HumanEval+ & LeetCode)** — Python data structures (`LRUCache`, `MinStack`, `Trie`, `IntervalMerger`) executed in a sandboxed subprocess against comprehensive unit tests.
6. **Advanced Theoretical Physics (CritPt)** — Olympiad-level relativistic Doppler shifts, Landau mean-field critical exponents, Carnot entropy, and LC resonance.
7. **Frontier Multidisciplinary Reasoning (Humanity's Last Exam)** — Curated PhD questions in game theory, algebraic topology, provability logic, and general relativity.
8. **Stateful Banking Agent (T3-Banking)** — Multi-turn banking sandbox with account transfers, fee waivers, card freeze/unfreeze, and fraud disputes.
9. **White-Collar Economic Audits (GDPval-AA v2)** — Balance sheet reconciliation, cloud SLA audit penalties, SaaS cohort retention matrices, and payroll tax compliance.
10. **Hallucination Restraint (AA-Omniscience)** — Adversarial counterfactual traps, false premise recognition, and precise scientific constants.
11. **Scientific Python Programming (SciCode)** — Numerical quantum density purity, Lennard-Jones potential, RK4 integrators, and diffusion algorithms.
12. **Interactive CLI & Terminal Agent (Terminal-Bench v4.0)** — Virtual Filesystem (VFS) log triage, nginx config syntax repair, git merge conflict resolution, and JSON migration validation.
13. **Long-Context Reasoning (AA-LCR)** — Multi-document timeline synthesis, microservice root cause analysis, and procurement liability extraction.
14. **3-Tier Throughput (1x, 4x, 8x)** — LLMPerf-style generation throughput with code generation prompts.
15. **Determinism** — Temperature 0, vendored fixtures, deterministic sandboxes, sandboxed Python subprocess execution, zero live internet dependencies.

### File layout

```
tool-eval-bench          # CLI entrypoint (execs `uv run --frozen python -m benchmark.main "$@"`)
pyproject.toml, uv.lock  # uv project (httpx, rich)
results/                 # benchmark result reports (<device>-<model>.md)
benchmark/
  main.py                # orchestrator, metrics, leaderboard, METRIC output
  sglang_client.py       # async SSE client (reasoning_content + tool-call fragments + usage)
  scenarios.py           # all 13 evaluation suites, tool schemas, deterministic sandboxes (Banking DB, Terminal VFS)
  grade.py               # deterministic grading routines for all evaluations
  live.py                # rich live panel / windowed table / concise non-TTY logs
```

---

## Requirements

- Python ≥ 3.11, `uv`
- Network access to the model endpoint (default `192.168.1.5:8888`)

## License

MIT — see [LICENSE](LICENSE).
