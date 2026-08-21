---
model: "Nemo-3.5-Lightning"
device: "DGX-Spark"
engine: "SGLang"
endpoint: "http://192.168.1.5:8888"
thinking: "auto"
date: "2026-08-21T11:04:03.381740+00:00"
tokens_per_second: 321.737
conc8_tps: 321.737
conc4_tps: 235.443
single_stream_tps: 125.259
time_to_first_token_ms: 228.143
total_duration_seconds: 288.971
smart_composite_score: 0.4688
tool_call_accuracy: 0.6774
ifeval_accuracy: 0.1667
gsm8k_accuracy: 0.3333
gpqa_accuracy: 0.6667
humaneval_accuracy: 0.5000
reasoning_ratio: 0.7845
---

# Benchmark Report: Nemo-3.5-Lightning on DGX-Spark

- **Date:** 2026-08-21 11:04:03 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `SGLang`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `Nemo-3.5-Lightning`
- **Thinking Mode:** `auto`
- **Total Execution Time:** **`4m 49.0s`** (289.0s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **🧠 Composite Intelligence Score:** **`46.9%`**

## ⚡ Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`321.74 tok/s`** | median of 3 reps (spread: 311.5–334.3 tok/s) |
| **4-Concurrent Throughput** | **`235.44 tok/s`** | median of 3 reps (spread: 222.8–243.0 tok/s) |
| **Single-Stream Throughput** | **`125.26 tok/s`** | 3474 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`228.1 ms`** | time to first token |
| **Total Execution Time** | **`4m 49.0s`** | total benchmark wall-clock time (289.0s) |
| **Reasoning Ratio** | **`0.784`** | 78.4% of generated tokens spent reasoning |

## 🛠️ Tool-Calling & Agentic Evaluation (BFCL & tau-bench)

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool Accuracy** | **`67.7%`** | 21 / 31 | BFCL exact-match, distractor selection & multi-turn |
| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`66.7%`** | 14 / 21 | Tool selection, args, restraint, distractors & schemas |
| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`70.0%`** | 7 / 10 | Multi-step dependency chains & stateful rollback |

**Failed Scenarios:** `s05, p01, p02, p03, p04, p05, p06, er01, er02, chain_01`

## 📋 Instruction Following (Google IFEval Hard)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **IFEval Hard Constraints** | **`16.7%`** | 1 / 6 | Multi-constraint conjunctions, JSON ranges, negative constraints |

**Failed Constraints:** `ifeval_h01 (got 0 paragraphs (expected exactly 3)), ifeval_h03 (word count 0 outside [60, 90]), ifeval_h04 (no markdown table found), ifeval_h05 (missing <audit> or </audit> tags), ifeval_h06 (too short (0 < 50 words))`

## 🔢 Math Reasoning (AIME & Competition Math)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AIME / Competition Math** | **`33.3%`** | 2 / 6 | Modular arithmetic, combinatorics, algebra & geometry proofs |

**Failed Problems:** `aime_01 (got 43, expected 49), aime_02 (no integer answer found in response), aime_03 (no integer answer found in response), aime_05 (no integer answer found in response)`

## 🔬 PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`66.7%`** | 4 / 6 | Google-proof PhD-level deduction & domain reasoning |

**Failed Questions:** `gpqa_01 (no choice (A/B/C/D) extracted), gpqa_04 (no choice (A/B/C/D) extracted)`

## 💻 Code Intelligence (HumanEval+ Data Structures)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **HumanEval+ Code & Data Structures** | **`50.0%`** | 3 / 6 | LRUCache, MinStack, Trie, interval merging with test execution |

**Failed Unit Tests:** `he_01 (assertion failed: AssertionError), he_05 (assertion failed: SyntaxError: unterminated string literal (det), he_06 (assertion failed: AssertionError)`

## 📊 Machine-Readable Metrics

```
METRIC tokens_per_second=321.737
METRIC conc8_tps=321.737
METRIC conc4_tps=235.443
METRIC single_stream_tps=125.259
METRIC time_to_first_token_ms=228.143
METRIC total_duration_seconds=288.971
METRIC smart_composite_score=0.4688
METRIC tool_call_accuracy=0.6774
METRIC ifeval_accuracy=0.1667
METRIC gsm8k_accuracy=0.3333
METRIC gpqa_accuracy=0.6667
METRIC humaneval_accuracy=0.5000
METRIC reasoning_ratio=0.7845
```
