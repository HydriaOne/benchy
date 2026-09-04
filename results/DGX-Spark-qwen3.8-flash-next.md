---
model: "qwen3.8-flash-next"
device: "DGX-Spark"
engine: "vLLM"
quant: "NVFP4"
endpoint: "http://192.168.1.5:8888"
thinking: "medium"
date: "2026-09-04T18:51:02.504757+00:00"
tokens_per_second: 95.952
conc8_tps: 95.952
conc4_tps: 91.797
single_stream_tps: 39.080
time_to_first_token_ms: 11428.474
total_duration_seconds: 1333.310
smart_composite_score: 0.9036
aa_intelligence_index: 0.8907
tool_call_accuracy: 0.8966
ifeval_accuracy: 1.0000
gsm8k_accuracy: 0.8333
gpqa_accuracy: 0.9167
humaneval_accuracy: 1.0000
critpt_accuracy: 1.0000
hle_accuracy: 0.9000
banking_accuracy: 1.0000
gdpval_accuracy: 1.0000
omniscience_accuracy: 0.7000
scicode_accuracy: 1.0000
terminal_accuracy: 0.5000
lcr_accuracy: 1.0000
reasoning_ratio: 0.5487
quality_per_time: 58.7765
total_tokens: 188377
input_tokens: 72259
output_tokens: 116118
---

# Benchmark Report: qwen3.8-flash-next on DGX-Spark

- **Date:** 2026-09-04 18:51:02 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM`
- **Quantization:** `NVFP4`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `qwen3.8-flash-next`
- **Thinking Mode:** `medium`
- **Total Execution Time:** **`22m 13.3s`** (1333.3s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **Composite Intelligence Score:** **`90.4%`**
- **Artificial Analysis Intelligence Index:** **`89.1%`**

> ⚠️ **Warning: Reasoning Token Starvation Detected**  
> **6 scenario(s)** (`nt03, aime_05, gpqa_03, omni_04, omni_08, term_02`) burned their token budget inside `<think>` (`finish_reason: length` with near-zero answer tokens).  
> The model was truncated before producing a final answer. Increase `--tool-max-tokens` / `--max-tokens` or evaluate with `--no-thinking` / `--thinking low` to prevent answer truncation.

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`95.95 tok/s`** | median of 3 reps (spread: 95.8–96.2 tok/s) |
| **4-Concurrent Throughput** | **`91.80 tok/s`** | median of 3 reps (spread: 87.1–95.2 tok/s) |
| **Single-Stream Throughput** | **`39.08 tok/s`** | 2472 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`11428.5 ms`** | time to first token |
| **Total Execution Time** | **`22m 13.3s`** | total benchmark wall-clock time (1333.3s) |
| **Reasoning Ratio** | **`0.549`** | 54.9% of generated tokens spent reasoning |
| **Quality / Time Efficiency** | **`58.8 pts`** | intelligence × time efficiency × token economy (0-100 scale) |

## Token Consumption

| Phase | Input (prompt) | Output (completion) | Reasoning | Total |
|---|---|---|---|---|
| Throughput single (1x) | 114 | 2,472 | 447 | 2,586 |
| Throughput 4x (3 reps) | 342 | 7,608 | 1,488 | 7,950 |
| Throughput 8x (3 reps) | 684 | 15,001 | 2,752 | 15,685 |
| Intelligence suites | 71,119 | 91,037 | 59,024 | 162,156 |
| **Total** | **72,259** | **116,118** | **63,711** | **188,377** |

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
| **GPQA Diamond (Physics / Chem / Bio)** | **`91.7%`** | 11 / 12 | Google-proof PhD-level deduction & domain reasoning |

**Failed Scenarios:** `gpqa_03 (no choice (A/B/C/D) extracted)`

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
| **GDPval-AA v2 Workflows** | **`100.0%`** | 6 / 6 | Balance sheet reconciliation, vendor SLA audit, SaaS metrics, payroll tax |

**Failed Scenarios:** `none`

## Hallucination Restraint & Adversarial Traps (AA-Omniscience)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-Omniscience Traps** | **`70.0%`** | 7 / 10 | Counterfactual false premises, fictional entities, precise scientific recall |

**Failed Scenarios:** `omni_04 (hallucinated false premise instead of restraining), omni_06 (missing expected facts: ['1.380649', '10^-23', 'e-23']), omni_08 (missing expected facts: ['2023', 'jupiter', 'ganymede'])`

## Scientific Python Computing (SciCode)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **SciCode Scientific Programming** | **`100.0%`** | 6 / 6 | Quantum purity, Lennard-Jones, RK4 integrator, diffusion & matrix math |

**Failed Scenarios:** `none`

## Interactive CLI & Terminal Agent (Terminal-Bench v4.0)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Terminal-Bench CLI Agent** | **`50.0%`** | 3 / 6 | VFS log triage, nginx syntax repair, git merge conflict resolution, JSON migration |

**Failed Scenarios:** `term_01 (expected file '/tmp/failed_ips.txt' was not created in VFS), term_02 (terminal summary missing expected details: ['syntax is ok', 'successful', '8000']), term_05 (file '/app/services/auth.env' missing expected string: 'API_KEY=${NEW_API_KEY}')`

## Long-Context Reasoning & Retrieval (AA-LCR)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-LCR Long Context** | **`100.0%`** | 4 / 4 | Multi-document timeline synthesis, incident root cause, procurement liability |

**Failed Scenarios:** `none`

## Machine-Readable Metrics

```
METRIC tokens_per_second=95.952
METRIC conc8_tps=95.952
METRIC conc4_tps=91.797
METRIC single_stream_tps=39.080
METRIC time_to_first_token_ms=11428.474
METRIC total_duration_seconds=1333.310
METRIC smart_composite_score=0.9036
METRIC aa_intelligence_index=0.8907
METRIC tool_call_accuracy=0.8966
METRIC ifeval_accuracy=1.0000
METRIC gsm8k_accuracy=0.8333
METRIC gpqa_accuracy=0.9167
METRIC humaneval_accuracy=1.0000
METRIC critpt_accuracy=1.0000
METRIC hle_accuracy=0.9000
METRIC banking_accuracy=1.0000
METRIC gdpval_accuracy=1.0000
METRIC omniscience_accuracy=0.7000
METRIC scicode_accuracy=1.0000
METRIC terminal_accuracy=0.5000
METRIC lcr_accuracy=1.0000
METRIC quality_per_time=58.7765
METRIC reasoning_ratio=0.5487
METRIC total_tokens=188377
```
