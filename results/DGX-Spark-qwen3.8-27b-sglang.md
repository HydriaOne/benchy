---
model: "qwen3.8-27b-sglang"
device: "DGX-Spark"
engine: "SGLang"
quant: "NVFP4"
endpoint: "http://192.168.1.5:8888"
thinking: "auto"
date: "2026-08-29T10:24:00.190896+00:00"
tokens_per_second: 136.228
conc8_tps: 136.228
conc4_tps: 110.546
single_stream_tps: 38.570
time_to_first_token_ms: 735.465
total_duration_seconds: 793.240
smart_composite_score: 0.8101
aa_intelligence_index: 0.7889
tool_call_accuracy: 0.9310
ifeval_accuracy: 1.0000
gsm8k_accuracy: 0.8333
gpqa_accuracy: 0.8333
humaneval_accuracy: 0.6667
critpt_accuracy: 1.0000
hle_accuracy: 0.7000
banking_accuracy: 1.0000
gdpval_accuracy: 1.0000
omniscience_accuracy: 0.4000
scicode_accuracy: 0.5000
terminal_accuracy: 0.6667
lcr_accuracy: 1.0000
reasoning_ratio: 0.7431
quality_per_time: 71.0231
total_tokens: 165510
input_tokens: 84586
output_tokens: 80924
---

# Benchmark Report: qwen3.8-27b-sglang on DGX-Spark

- **Date:** 2026-08-29 10:24:00 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `SGLang`
- **Quantization:** `NVFP4`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `qwen3.8-27b-sglang`
- **Thinking Mode:** `auto`
- **Total Execution Time:** **`13m 13.2s`** (793.2s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **Composite Intelligence Score:** **`81.0%`**
- **Artificial Analysis Intelligence Index:** **`78.9%`**

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`136.23 tok/s`** | median of 3 reps (spread: 134.5–142.7 tok/s) |
| **4-Concurrent Throughput** | **`110.55 tok/s`** | median of 3 reps (spread: 69.2–114.9 tok/s) |
| **Single-Stream Throughput** | **`38.57 tok/s`** | 931 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`735.5 ms`** | time to first token |
| **Total Execution Time** | **`13m 13.2s`** | total benchmark wall-clock time (793.2s) |
| **Reasoning Ratio** | **`0.743`** | 74.3% of generated tokens spent reasoning |
| **Quality / Time Efficiency** | **`61.7 pts`** | intelligence / (tokens × seconds) × 10¹⁰ (0-100 scale) |

## Token Consumption

| Phase | Input (prompt) | Output (completion) | Reasoning | Total |
|---|---|---|---|---|
| Throughput single (1x) | 282 | 931 | 300 | 1,213 |
| Throughput 4x (3 reps) | 846 | 2,958 | 1,130 | 3,804 |
| Throughput 8x (3 reps) | 1,692 | 5,809 | 1,962 | 7,501 |
| Intelligence suites | 81,766 | 71,226 | 56,746 | 152,992 |
| **Total** | **84,586** | **80,924** | **60,138** | **165,510** |

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
| **IFEval Hard Constraints** | **`100.0%`** | 6 / 6 | Multi-constraint conjunctions, JSON ranges, negative constraints |

**Failed Scenarios:** `none`

## Math Reasoning (AIME & Competition Math)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AIME / Competition Math** | **`83.3%`** | 5 / 6 | Modular arithmetic, combinatorics, algebra & geometry proofs |

**Failed Scenarios:** `aime_05 (no integer answer found in response)`

## PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`83.3%`** | 10 / 12 | Google-proof PhD-level deduction & domain reasoning |

**Failed Scenarios:** `gpqa_01 (no choice (A/B/C/D) extracted), gpqa_07 (no choice (A/B/C/D) extracted)`

## Code Intelligence (HumanEval+ Data Structures)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **HumanEval+ Code & Data Structures** | **`66.7%`** | 4 / 6 | LRUCache, MinStack, Trie, interval merging with test execution |

**Failed Scenarios:** `he_01 (assertion failed: AssertionError), he_06 (assertion failed: AssertionError)`

## Advanced Physics & Math Reasoning (CritPt)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **CritPt Competition Physics** | **`100.0%`** | 8 / 8 | Phase transitions, relativistic Doppler, thermodynamics, harmonic oscillator |

**Failed Scenarios:** `none`

## Frontier Multidisciplinary PhD Exam (Humanity's Last Exam)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Humanity's Last Exam (HLE)** | **`70.0%`** | 7 / 10 | Game theory, algebraic topology, provability logic, black holes, genetics |

**Failed Scenarios:** `hle_01 (got 30, expected 35), hle_03 (no choice (A/B/C/D) extracted), hle_08 (no choice (A/B/C/D) extracted)`

## Stateful Banking Agent (T3-Banking / tau-bench)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **T3-Banking Agent** | **`100.0%`** | 8 / 8 | Multi-turn bank DB mutations, fee waivers, card freeze & dispute workflows |

**Failed Scenarios:** `none`

## White-Collar Economic Audits (GDPval-AA v2)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GDPval-AA v2 Workflows** | **`100.0%`** | 6 / 6 | Balance sheet reconciliation, vendor SLA audit, SaaS metrics, payroll tax |

**Failed Scenarios:** `none`

## Hallucination Restraint & Adversarial Traps (AA-Omniscience)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-Omniscience Traps** | **`40.0%`** | 4 / 10 | Counterfactual false premises, fictional entities, precise scientific recall |

**Failed Scenarios:** `omni_02 (hallucinated false premise instead of restraining), omni_04 (hallucinated false premise instead of restraining), omni_06 (missing expected facts: ['1.380649', '10^-23', 'e-23']), omni_07 (hallucinated false premise instead of restraining), omni_09 (missing expected facts: ['0', 'zero', 'none']), omni_10 (hallucinated false premise instead of restraining)`

## Scientific Python Computing (SciCode)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **SciCode Scientific Programming** | **`50.0%`** | 3 / 6 | Quantum purity, Lennard-Jones, RK4 integrator, diffusion & matrix math |

**Failed Scenarios:** `scicode_01 (assertion failed: TypeError: must be real number, not NoneType), scicode_02 (assertion failed: TypeError: must be real number, not NoneType), scicode_06 (assertion failed: TypeError: 'NoneType' object is not subscript)`

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
METRIC tokens_per_second=136.228
METRIC conc8_tps=136.228
METRIC conc4_tps=110.546
METRIC single_stream_tps=38.570
METRIC time_to_first_token_ms=735.465
METRIC total_duration_seconds=793.240
METRIC smart_composite_score=0.8101
METRIC aa_intelligence_index=0.7889
METRIC tool_call_accuracy=0.9310
METRIC ifeval_accuracy=1.0000
METRIC gsm8k_accuracy=0.8333
METRIC gpqa_accuracy=0.8333
METRIC humaneval_accuracy=0.6667
METRIC critpt_accuracy=1.0000
METRIC hle_accuracy=0.7000
METRIC banking_accuracy=1.0000
METRIC gdpval_accuracy=1.0000
METRIC omniscience_accuracy=0.4000
METRIC scicode_accuracy=0.5000
METRIC terminal_accuracy=0.6667
METRIC lcr_accuracy=1.0000
METRIC reasoning_ratio=0.7431
METRIC total_tokens=165510
METRIC quality_per_time=71.0231
```
