---
model: "ornith-1.5-35b-a3b-nvfp4"
device: "DGX-Spark"
engine: "vLLM"
endpoint: "http://192.168.1.5:8888"
thinking: "auto"
date: "2026-08-21T11:18:06.873565+00:00"
tokens_per_second: 264.459
conc8_tps: 264.459
conc4_tps: 171.901
single_stream_tps: 94.393
time_to_first_token_ms: 6419.533
total_duration_seconds: 306.929
smart_composite_score: 0.7473
tool_call_accuracy: 0.9032
ifeval_accuracy: 0.6667
gsm8k_accuracy: 0.6667
gpqa_accuracy: 0.5000
humaneval_accuracy: 1.0000
reasoning_ratio: 0.0000
---

# Benchmark Report: ornith-1.5-35b-a3b-nvfp4 on DGX-Spark

- **Date:** 2026-08-21 11:18:06 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `ornith-1.5-35b-a3b-nvfp4`
- **Thinking Mode:** `auto`
- **Total Execution Time:** **`5m 6.9s`** (306.9s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **🧠 Composite Intelligence Score:** **`74.7%`**

## ⚡ Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`264.46 tok/s`** | median of 3 reps (spread: 259.8–273.1 tok/s) |
| **4-Concurrent Throughput** | **`171.90 tok/s`** | median of 3 reps (spread: 169.9–173.1 tok/s) |
| **Single-Stream Throughput** | **`94.39 tok/s`** | 2699 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`6419.5 ms`** | time to first token |
| **Total Execution Time** | **`5m 6.9s`** | total benchmark wall-clock time (306.9s) |
| **Reasoning Ratio** | **`0.000`** | 0.0% of generated tokens spent reasoning |

## 🛠️ Tool-Calling & Agentic Evaluation (BFCL & tau-bench)

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool Accuracy** | **`90.3%`** | 28 / 31 | BFCL exact-match, distractor selection & multi-turn |
| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`95.2%`** | 20 / 21 | Tool selection, args, restraint, distractors & schemas |
| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`80.0%`** | 8 / 10 | Multi-step dependency chains & stateful rollback |

**Failed Scenarios:** `s05, er01, er02`

## 📋 Instruction Following (Google IFEval Hard)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **IFEval Hard Constraints** | **`66.7%`** | 4 / 6 | Multi-constraint conjunctions, JSON ranges, negative constraints |

**Failed Constraints:** `ifeval_h04 (no markdown table found), ifeval_h05 (section headers following tags are not in ALL CAPS)`

## 🔢 Math Reasoning (AIME & Competition Math)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AIME / Competition Math** | **`66.7%`** | 4 / 6 | Modular arithmetic, combinatorics, algebra & geometry proofs |

**Failed Problems:** `aime_01 (got 7, expected 49), aime_05 (no integer answer found in response)`

## 🔬 PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`50.0%`** | 3 / 6 | Google-proof PhD-level deduction & domain reasoning |

**Failed Questions:** `gpqa_01 (no choice (A/B/C/D) extracted), gpqa_02 (no choice (A/B/C/D) extracted), gpqa_04 (no choice (A/B/C/D) extracted)`

## 💻 Code Intelligence (HumanEval+ Data Structures)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **HumanEval+ Code & Data Structures** | **`100.0%`** | 6 / 6 | LRUCache, MinStack, Trie, interval merging with test execution |

**Failed Unit Tests:** `none`

## 📊 Machine-Readable Metrics

```
METRIC tokens_per_second=264.459
METRIC conc8_tps=264.459
METRIC conc4_tps=171.901
METRIC single_stream_tps=94.393
METRIC time_to_first_token_ms=6419.533
METRIC total_duration_seconds=306.929
METRIC smart_composite_score=0.7473
METRIC tool_call_accuracy=0.9032
METRIC ifeval_accuracy=0.6667
METRIC gsm8k_accuracy=0.6667
METRIC gpqa_accuracy=0.5000
METRIC humaneval_accuracy=1.0000
METRIC reasoning_ratio=0.0000
```
