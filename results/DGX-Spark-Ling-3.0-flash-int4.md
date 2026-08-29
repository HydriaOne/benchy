---
model: "Ling-3.0-flash-int4"
device: "DGX-Spark"
engine: "SGLang"
quant: "INT4"
endpoint: "http://192.168.1.5:8888"
thinking: "auto"
date: "2026-08-29T12:34:11.863523+00:00"
tokens_per_second: 159.462
conc8_tps: 159.462
conc4_tps: 140.984
single_stream_tps: 71.961
time_to_first_token_ms: 4452.505
total_duration_seconds: 689.347
smart_composite_score: 0.8101
aa_intelligence_index: 0.8815
tool_call_accuracy: 0.9310
ifeval_accuracy: 0.3333
gsm8k_accuracy: 0.5000
gpqa_accuracy: 1.0000
humaneval_accuracy: 0.8333
critpt_accuracy: 1.0000
hle_accuracy: 0.9000
banking_accuracy: 1.0000
gdpval_accuracy: 0.8333
omniscience_accuracy: 0.7000
scicode_accuracy: 0.8333
terminal_accuracy: 0.6667
lcr_accuracy: 1.0000
reasoning_ratio: 0.5800
quality_per_time: 75.3175
total_tokens: 171968
input_tokens: 70628
output_tokens: 101340
---

# Benchmark Report: Ling-3.0-flash-int4 on DGX-Spark

- **Date:** 2026-08-29 12:34:11 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `SGLang`
- **Quantization:** `INT4`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `Ling-3.0-flash-int4`
- **Thinking Mode:** `auto`
- **Total Execution Time:** **`11m 29.3s`** (689.3s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **Composite Intelligence Score:** **`81.0%`**
- **Artificial Analysis Intelligence Index:** **`88.1%`**

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`159.46 tok/s`** | median of 3 reps (spread: 158.5–160.6 tok/s) |
| **4-Concurrent Throughput** | **`140.98 tok/s`** | median of 3 reps (spread: 138.3–141.9 tok/s) |
| **Single-Stream Throughput** | **`71.96 tok/s`** | 2940 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`4452.5 ms`** | time to first token |
| **Total Execution Time** | **`11m 29.3s`** | total benchmark wall-clock time (689.3s) |
| **Reasoning Ratio** | **`0.580`** | 58.0% of generated tokens spent reasoning |
| **Quality / Time Efficiency** | **`68.3 pts`** | intelligence / (tokens × seconds) × 10¹⁰ (0-100 scale) |

## Token Consumption

| Phase | Input (prompt) | Output (completion) | Reasoning | Total |
|---|---|---|---|---|
| Throughput single (1x) | 156 | 2,940 | 570 | 3,096 |
| Throughput 4x (3 reps) | 468 | 8,588 | 2,018 | 9,056 |
| Throughput 8x (3 reps) | 936 | 16,412 | 3,496 | 17,348 |
| Intelligence suites | 69,068 | 73,400 | 52,691 | 142,468 |
| **Total** | **70,628** | **101,340** | **58,775** | **171,968** |

## Tool-Calling & Agentic Evaluation (BFCL & tau-bench)

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool Accuracy** | **`93.1%`** | 27 / 29 | BFCL exact-match, distractor selection & multi-turn |
| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`95.2%`** | 20 / 21 | Tool selection, args, restraint, distractors & schemas |
| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`87.5%`** | 7 / 8 | Multi-step dependency chains & stateful rollback |

**Failed Scenarios:** `s05, m04`

## Instruction Following (Google IFEval Hard)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **IFEval Hard Constraints** | **`33.3%`** | 2 / 6 | Multi-constraint conjunctions, JSON ranges, negative constraints |

**Failed Scenarios:** `ifeval_h01 (got 1 paragraphs (expected exactly 3)), ifeval_h03 (word count 673 outside [60, 90]), ifeval_h04 (conclusion contains forbidden letter 'e' in: ['sentence', 'least', 'ZERO']), ifeval_h06 (used forbidden word(s): ['encryption', 'cipher', 'keys', 'protection', 'secure', 'key', 'encrypt', 'ciphers', 'protect'])`

## Math Reasoning (AIME & Competition Math)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AIME / Competition Math** | **`50.0%`** | 3 / 6 | Modular arithmetic, combinatorics, algebra & geometry proofs |

**Failed Scenarios:** `aime_01 (got 4, expected 49), aime_03 (got 6, expected 10), aime_05 (got 11, expected 154)`

## PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`100.0%`** | 12 / 12 | Google-proof PhD-level deduction & domain reasoning |

**Failed Scenarios:** `none`

## Code Intelligence (HumanEval+ Data Structures)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **HumanEval+ Code & Data Structures** | **`83.3%`** | 5 / 6 | LRUCache, MinStack, Trie, interval merging with test execution |

**Failed Scenarios:** `he_06 (assertion failed: SyntaxError: invalid decimal literal)`

## Advanced Physics & Math Reasoning (CritPt)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **CritPt Competition Physics** | **`100.0%`** | 8 / 8 | Phase transitions, relativistic Doppler, thermodynamics, harmonic oscillator |

**Failed Scenarios:** `none`

## Frontier Multidisciplinary PhD Exam (Humanity's Last Exam)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Humanity's Last Exam (HLE)** | **`90.0%`** | 9 / 10 | Game theory, algebraic topology, provability logic, black holes, genetics |

**Failed Scenarios:** `hle_01 (got 30, expected 35)`

## Stateful Banking Agent (T3-Banking / tau-bench)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **T3-Banking Agent** | **`100.0%`** | 8 / 8 | Multi-turn bank DB mutations, fee waivers, card freeze & dispute workflows |

**Failed Scenarios:** `none`

## White-Collar Economic Audits (GDPval-AA v2)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GDPval-AA v2 Workflows** | **`83.3%`** | 5 / 6 | Balance sheet reconciliation, vendor SLA audit, SaaS metrics, payroll tax |

**Failed Scenarios:** `gdpval_03 (invalid JSON: Expecting value: line 1 column 1 (c)`

## Hallucination Restraint & Adversarial Traps (AA-Omniscience)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-Omniscience Traps** | **`70.0%`** | 7 / 10 | Counterfactual false premises, fictional entities, precise scientific recall |

**Failed Scenarios:** `omni_03 (missing expected facts: ['70', '173.05']), omni_06 (missing expected facts: ['1.380649', '10^-23', 'e-23']), omni_09 (missing expected facts: ['0', 'zero', 'none'])`

## Scientific Python Computing (SciCode)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **SciCode Scientific Programming** | **`83.3%`** | 5 / 6 | Quantum purity, Lennard-Jones, RK4 integrator, diffusion & matrix math |

**Failed Scenarios:** `scicode_05 (assertion failed: NameError: name 'row_sum潜水' is not defined)`

## Interactive CLI & Terminal Agent (Terminal-Bench v4.0)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Terminal-Bench CLI Agent** | **`66.7%`** | 4 / 6 | VFS log triage, nginx syntax repair, git merge conflict resolution, JSON migration |

**Failed Scenarios:** `term_01 (expected file '/tmp/failed_ips.txt' was not created in VFS), term_05 (file '/app/services/auth.env' missing expected string: 'API_KEY=${NEW_API_KEY}')`

## Long-Context Reasoning & Retrieval (AA-LCR)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-LCR Long Context** | **`100.0%`** | 4 / 4 | Multi-document timeline synthesis, incident root cause, procurement liability |

**Failed Scenarios:** `none`

## Machine-Readable Metrics

```
METRIC tokens_per_second=159.462
METRIC conc8_tps=159.462
METRIC conc4_tps=140.984
METRIC single_stream_tps=71.961
METRIC time_to_first_token_ms=4452.505
METRIC total_duration_seconds=689.347
METRIC smart_composite_score=0.8101
METRIC aa_intelligence_index=0.8815
METRIC tool_call_accuracy=0.9310
METRIC ifeval_accuracy=0.3333
METRIC gsm8k_accuracy=0.5000
METRIC gpqa_accuracy=1.0000
METRIC humaneval_accuracy=0.8333
METRIC critpt_accuracy=1.0000
METRIC hle_accuracy=0.9000
METRIC banking_accuracy=1.0000
METRIC gdpval_accuracy=0.8333
METRIC omniscience_accuracy=0.7000
METRIC scicode_accuracy=0.8333
METRIC terminal_accuracy=0.6667
METRIC lcr_accuracy=1.0000
METRIC quality_per_time=75.3175
METRIC reasoning_ratio=0.5800
METRIC total_tokens=171968
```
