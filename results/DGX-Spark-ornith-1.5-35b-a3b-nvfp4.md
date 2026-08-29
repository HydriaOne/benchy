---
model: "ornith-1.5-35b-a3b-nvfp4"
device: "DGX-Spark"
engine: "vLLM"
quant: "NVFP4"
endpoint: "http://192.168.1.5:8888"
thinking: "auto"
date: "2026-08-29T13:38:42.194032+00:00"
tokens_per_second: 260.760
conc8_tps: 260.760
conc4_tps: 176.459
single_stream_tps: 93.924
time_to_first_token_ms: 338.580
total_duration_seconds: 502.949
smart_composite_score: 0.8081
aa_intelligence_index: 0.8083
tool_call_accuracy: 0.8966
ifeval_accuracy: 0.6667
gsm8k_accuracy: 0.6667
gpqa_accuracy: 0.8333
humaneval_accuracy: 1.0000
critpt_accuracy: 1.0000
hle_accuracy: 0.7000
banking_accuracy: 0.8750
gdpval_accuracy: 1.0000
omniscience_accuracy: 0.7000
scicode_accuracy: 0.8333
terminal_accuracy: 0.3333
lcr_accuracy: 1.0000
reasoning_ratio: 0.0000
quality_per_time: 86.7524
total_tokens: 180069
input_tokens: 73683
output_tokens: 106386
---

# Benchmark Report: ornith-1.5-35b-a3b-nvfp4 on DGX-Spark

- **Date:** 2026-08-29 13:38:42 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM`
- **Quantization:** `NVFP4`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `ornith-1.5-35b-a3b-nvfp4`
- **Thinking Mode:** `auto`
- **Total Execution Time:** **`8m 22.9s`** (502.9s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **Composite Intelligence Score:** **`80.8%`**
- **Artificial Analysis Intelligence Index:** **`80.8%`**

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`260.76 tok/s`** | median of 3 reps (spread: 260.1–268.5 tok/s) |
| **4-Concurrent Throughput** | **`176.46 tok/s`** | median of 3 reps (spread: 170.3–180.5 tok/s) |
| **Single-Stream Throughput** | **`93.92 tok/s`** | 2943 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`338.6 ms`** | time to first token |
| **Total Execution Time** | **`8m 22.9s`** | total benchmark wall-clock time (502.9s) |
| **Reasoning Ratio** | **`0.000`** | 0.0% of generated tokens spent reasoning |
| **Quality / Time Efficiency** | **`89.2 pts`** | intelligence / (tokens × seconds) × 10¹⁰ (0-100 scale) |

## Token Consumption

| Phase | Input (prompt) | Output (completion) | Reasoning | Total |
|---|---|---|---|---|
| Throughput single (1x) | 114 | 2,943 | 0 | 3,057 |
| Throughput 4x (3 reps) | 342 | 8,269 | 0 | 8,611 |
| Throughput 8x (3 reps) | 684 | 17,109 | 0 | 17,793 |
| Intelligence suites | 72,543 | 78,065 | 0 | 150,608 |
| **Total** | **73,683** | **106,386** | **0** | **180,069** |

## Tool-Calling & Agentic Evaluation (BFCL & tau-bench)

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool Accuracy** | **`89.7%`** | 26 / 29 | BFCL exact-match, distractor selection & multi-turn |
| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`90.5%`** | 19 / 21 | Tool selection, args, restraint, distractors & schemas |
| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`87.5%`** | 7 / 8 | Multi-step dependency chains & stateful rollback |

**Failed Scenarios:** `s05, m04, nt03`

## Instruction Following (Google IFEval Hard)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **IFEval Hard Constraints** | **`66.7%`** | 4 / 6 | Multi-constraint conjunctions, JSON ranges, negative constraints |

**Failed Scenarios:** `ifeval_h04 (no markdown table found), ifeval_h05 (section headers following tags are not in ALL CAPS)`

## Math Reasoning (AIME & Competition Math)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AIME / Competition Math** | **`66.7%`** | 4 / 6 | Modular arithmetic, combinatorics, algebra & geometry proofs |

**Failed Scenarios:** `aime_02 (got 144, expected 15), aime_05 (no integer answer found in response)`

## PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`83.3%`** | 10 / 12 | Google-proof PhD-level deduction & domain reasoning |

**Failed Scenarios:** `gpqa_01 (no choice (A/B/C/D) extracted), gpqa_04 (no choice (A/B/C/D) extracted)`

## Code Intelligence (HumanEval+ Data Structures)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **HumanEval+ Code & Data Structures** | **`100.0%`** | 6 / 6 | LRUCache, MinStack, Trie, interval merging with test execution |

**Failed Scenarios:** `none`

## Advanced Physics & Math Reasoning (CritPt)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **CritPt Competition Physics** | **`100.0%`** | 8 / 8 | Phase transitions, relativistic Doppler, thermodynamics, harmonic oscillator |

**Failed Scenarios:** `none`

## Frontier Multidisciplinary PhD Exam (Humanity's Last Exam)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Humanity's Last Exam (HLE)** | **`70.0%`** | 7 / 10 | Game theory, algebraic topology, provability logic, black holes, genetics |

**Failed Scenarios:** `hle_01 (got 60, expected 35), hle_06 (got -88, expected -89), hle_10 (got 1, expected 9)`

## Stateful Banking Agent (T3-Banking / tau-bench)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **T3-Banking Agent** | **`87.5%`** | 7 / 8 | Multi-turn bank DB mutations, fee waivers, card freeze & dispute workflows |

**Failed Scenarios:** `bank_04 (final answer missing expected info: ['insufficient', '450', 'cannot'])`

## White-Collar Economic Audits (GDPval-AA v2)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GDPval-AA v2 Workflows** | **`100.0%`** | 6 / 6 | Balance sheet reconciliation, vendor SLA audit, SaaS metrics, payroll tax |

**Failed Scenarios:** `none`

## Hallucination Restraint & Adversarial Traps (AA-Omniscience)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-Omniscience Traps** | **`70.0%`** | 7 / 10 | Counterfactual false premises, fictional entities, precise scientific recall |

**Failed Scenarios:** `omni_03 (missing expected facts: ['70', '173.05']), omni_06 (missing expected facts: ['1.380649', '10^-23', 'e-23']), omni_09 (missing expected facts: ['0', 'zero', 'none'])`

## Scientific Python Computing (SciCode)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **SciCode Scientific Programming** | **`83.3%`** | 5 / 6 | Quantum purity, Lennard-Jones, RK4 integrator, diffusion & matrix math |

**Failed Scenarios:** `scicode_04 (assertion failed: ModuleNotFoundError: No module named 'numpy')`

## Interactive CLI & Terminal Agent (Terminal-Bench v4.0)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Terminal-Bench CLI Agent** | **`33.3%`** | 2 / 6 | VFS log triage, nginx syntax repair, git merge conflict resolution, JSON migration |

**Failed Scenarios:** `term_01 (expected file '/tmp/failed_ips.txt' was not created in VFS), term_02 (terminal summary missing expected details: ['syntax is ok', 'successful', '8000']), term_05 (file '/app/services/auth.env' missing expected string: 'API_KEY=${NEW_API_KEY}'), term_06 (terminal summary missing expected details: ['valid', 'json', 'trailing comma'])`

## Long-Context Reasoning & Retrieval (AA-LCR)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-LCR Long Context** | **`100.0%`** | 4 / 4 | Multi-document timeline synthesis, incident root cause, procurement liability |

**Failed Scenarios:** `none`

## Machine-Readable Metrics

```
METRIC tokens_per_second=260.760
METRIC conc8_tps=260.760
METRIC conc4_tps=176.459
METRIC single_stream_tps=93.924
METRIC time_to_first_token_ms=338.580
METRIC total_duration_seconds=502.949
METRIC smart_composite_score=0.8081
METRIC aa_intelligence_index=0.8083
METRIC tool_call_accuracy=0.8966
METRIC ifeval_accuracy=0.6667
METRIC gsm8k_accuracy=0.6667
METRIC gpqa_accuracy=0.8333
METRIC humaneval_accuracy=1.0000
METRIC critpt_accuracy=1.0000
METRIC hle_accuracy=0.7000
METRIC banking_accuracy=0.8750
METRIC gdpval_accuracy=1.0000
METRIC omniscience_accuracy=0.7000
METRIC scicode_accuracy=0.8333
METRIC terminal_accuracy=0.3333
METRIC lcr_accuracy=1.0000
METRIC quality_per_time=86.7524
METRIC reasoning_ratio=0.0000
METRIC total_tokens=180069
```
