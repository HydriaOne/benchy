---
model: "deepseek-v4-flash-0731"
device: "DGX-Spark"
engine: "vLLM"
endpoint: "http://192.168.1.5:8888"
thinking: "auto"
date: "2026-08-21T13:22:57.942563+00:00"
tokens_per_second: 34.092
conc8_tps: 34.092
conc4_tps: 34.648
single_stream_tps: 34.085
time_to_first_token_ms: 54173.389
total_duration_seconds: 1374.258
smart_composite_score: 0.7935
tool_call_accuracy: 0.9677
ifeval_accuracy: 0.8333
gsm8k_accuracy: 0.6667
gpqa_accuracy: 0.8333
humaneval_accuracy: 0.6667
reasoning_ratio: 0.0000
---

# Benchmark Report: deepseek-v4-flash-0731 on DGX-Spark

- **Date:** 2026-08-21 13:22:57 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `deepseek-v4-flash-0731`
- **Thinking Mode:** `auto`
- **Total Execution Time:** **`22m 54.3s`** (1374.3s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **🧠 Composite Intelligence Score:** **`79.4%`**

## ⚡ Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`34.09 tok/s`** | median of 3 reps (spread: 33.7–35.1 tok/s) |
| **4-Concurrent Throughput** | **`34.65 tok/s`** | median of 3 reps (spread: 33.8–35.0 tok/s) |
| **Single-Stream Throughput** | **`34.09 tok/s`** | 1858 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`54173.4 ms`** | time to first token |
| **Total Execution Time** | **`22m 54.3s`** | total benchmark wall-clock time (1374.3s) |
| **Reasoning Ratio** | **`0.000`** | 0.0% of generated tokens spent reasoning |

## 🛠️ Tool-Calling & Agentic Evaluation (BFCL & tau-bench)

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool Accuracy** | **`96.8%`** | 30 / 31 | BFCL exact-match, distractor selection & multi-turn |
| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`95.2%`** | 20 / 21 | Tool selection, args, restraint, distractors & schemas |
| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`100.0%`** | 10 / 10 | Multi-step dependency chains & stateful rollback |

**Failed Scenarios:** `s05`

## 📋 Instruction Following (Google IFEval Hard)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **IFEval Hard Constraints** | **`83.3%`** | 5 / 6 | Multi-constraint conjunctions, JSON ranges, negative constraints |

**Failed Constraints:** `ifeval_h06 (too short (0 < 50 words))`

## 🔢 Math Reasoning (AIME & Competition Math)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AIME / Competition Math** | **`66.7%`** | 4 / 6 | Modular arithmetic, combinatorics, algebra & geometry proofs |

**Failed Problems:** `aime_03 (no integer answer found in response), aime_05 (no integer answer found in response)`

## 🔬 PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`83.3%`** | 5 / 6 | Google-proof PhD-level deduction & domain reasoning |

**Failed Questions:** `gpqa_01 (no choice (A/B/C/D) extracted)`

## 💻 Code Intelligence (HumanEval+ Data Structures)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **HumanEval+ Code & Data Structures** | **`66.7%`** | 4 / 6 | LRUCache, MinStack, Trie, interval merging with test execution |

**Failed Unit Tests:** `he_05 (assertion failed: AssertionError), he_06 (assertion failed: AssertionError)`

## 📊 Machine-Readable Metrics

```
METRIC tokens_per_second=34.092
METRIC conc8_tps=34.092
METRIC conc4_tps=34.648
METRIC single_stream_tps=34.085
METRIC time_to_first_token_ms=54173.389
METRIC total_duration_seconds=1374.258
METRIC smart_composite_score=0.7935
METRIC tool_call_accuracy=0.9677
METRIC ifeval_accuracy=0.8333
METRIC gsm8k_accuracy=0.6667
METRIC gpqa_accuracy=0.8333
METRIC humaneval_accuracy=0.6667
METRIC reasoning_ratio=0.0000
```
