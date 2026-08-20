---
model: "ornith-1.5-35b-a3b-nvfp4"
device: "DGX-Spark"
engine: "vLLM"
endpoint: "http://192.168.1.5:8888"
date: "2026-08-20T10:12:48.271261+00:00"
tokens_per_second: 114.029
conc8_tps: 114.029
conc4_tps: 71.607
single_stream_tps: 36.233
time_to_first_token_ms: 14738.554
total_duration_seconds: 827.692
smart_composite_score: 0.9398
tool_call_accuracy: 0.9259
ifeval_accuracy: 1.0000
gsm8k_accuracy: 1.0000
humaneval_accuracy: 0.8333
reasoning_ratio: 0.0000
---

# Benchmark Report: ornith-1.5-35b-a3b-nvfp4 on DGX-Spark

- **Date:** 2026-08-20 10:12:48 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `ornith-1.5-35b-a3b-nvfp4`
- **Thinking Mode:** `on`
- **Total Execution Time:** **`13m 47.7s`** (827.7s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **🧠 Composite Intelligence Score:** **`94.0%`**

## ⚡ Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`114.03 tok/s`** | median of 3 reps (spread: 100.5–115.9 tok/s) |
| **4-Concurrent Throughput** | **`71.61 tok/s`** | median of 3 reps (spread: 70.9–71.7 tok/s) |
| **Single-Stream Throughput** | **`36.23 tok/s`** | 2817 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`14738.6 ms`** | time to first token |
| **Total Execution Time** | **`13m 47.7s`** | total benchmark wall-clock time (827.7s) |
| **Reasoning Ratio** | **`0.000`** | 0.0% of generated tokens spent reasoning |

## 🛠️ Tool-Calling & Agentic Evaluation (BFCL & tau-bench)

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool Accuracy** | **`92.6%`** | 25 / 27 | BFCL exact-match & tau-bench multi-turn |
| **Single-Turn (Simple / Parallel / Restraint / Complex)** | **`100.0%`** | 19 / 19 | Tool selection, args, restraint & schemas |
| **Agentic Multi-Turn (Execution & Error Recovery)** | **`75.0%`** | 6 / 8 | Multi-step execution & error recovery |

**Failed Scenarios:** `er01, er02`

## 📋 Instruction Following (Google IFEval)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **IFEval Verifiable Constraints** | **`100.0%`** | 6 / 6 | JSON schemas, negative constraints, word counts, formatting |

**Failed Constraints:** `none`

## 🔢 Math Reasoning (GSM8K)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GSM8K Grade School Math** | **`100.0%`** | 6 / 6 | Multi-step arithmetic word problem reasoning |

**Failed Problems:** `none`

## 💻 Code Intelligence (HumanEval)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **HumanEval Python Coding** | **`83.3%`** | 5 / 6 | Functional Python generation with sandboxed test execution |

**Failed Unit Tests:** `he_02 (assertion failed: AssertionError)`

## 📊 Machine-Readable Metrics

```
METRIC tokens_per_second=114.029
METRIC conc8_tps=114.029
METRIC conc4_tps=71.607
METRIC single_stream_tps=36.233
METRIC time_to_first_token_ms=14738.554
METRIC total_duration_seconds=827.692
METRIC smart_composite_score=0.9398
METRIC tool_call_accuracy=0.9259
METRIC ifeval_accuracy=1.0000
METRIC gsm8k_accuracy=1.0000
METRIC humaneval_accuracy=0.8333
METRIC reasoning_ratio=0.0000
```
