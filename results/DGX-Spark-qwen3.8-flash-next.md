---
model: "qwen3.8-flash-next"
device: "DGX-Spark"
engine: "vLLM"
quant: "NVFP4"
endpoint: "http://192.168.1.5:8888"
thinking: "medium"
date: "2026-09-05T13:33:13.232245+00:00"
tokens_per_second: 97.594
conc8_tps: 97.594
conc4_tps: 94.289
single_stream_tps: 41.970
time_to_first_token_ms: 12054.730
total_duration_seconds: 1321.938
smart_composite_score: 0.9074
aa_intelligence_index: 0.8963
tool_call_accuracy: 0.8966
ifeval_accuracy: 1.0000
gsm8k_accuracy: 0.8333
gpqa_accuracy: 1.0000
humaneval_accuracy: 1.0000
critpt_accuracy: 1.0000
hle_accuracy: 0.8000
banking_accuracy: 1.0000
gdpval_accuracy: 1.0000
omniscience_accuracy: 0.6000
scicode_accuracy: 1.0000
terminal_accuracy: 0.6667
lcr_accuracy: 1.0000
reasoning_ratio: 0.5125
quality_per_time: 60.1116
total_tokens: 179829
input_tokens: 69807
output_tokens: 110022
---

# Benchmark Report: qwen3.8-flash-next on DGX-Spark

- **Date:** 2026-09-05 13:33:13 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM`
- **Quantization:** `NVFP4`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `qwen3.8-flash-next`
- **Thinking Mode:** `medium`
- **Total Execution Time:** **`22m 1.9s`** (1321.9s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **Composite Intelligence Score:** **`90.7%`**
- **Artificial Analysis Intelligence Index:** **`89.6%`**

> ⚠️ **Warning: Reasoning Token Starvation Detected**  
> **4 scenario(s)** (`nt03, aime_05, omni_02, omni_03`) burned their token budget inside `<think>` (`finish_reason: length` with near-zero answer tokens).  
> The model was truncated before producing a final answer. Increase `--tool-max-tokens` / `--max-tokens` or evaluate with `--no-thinking` / `--thinking low` to prevent answer truncation.

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`97.59 tok/s`** | median of 3 reps (spread: 92.0–99.7 tok/s) |
| **4-Concurrent Throughput** | **`94.29 tok/s`** | median of 3 reps (spread: 93.0–98.1 tok/s) |
| **Single-Stream Throughput** | **`41.97 tok/s`** | 2403 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`12054.7 ms`** | time to first token |
| **Total Execution Time** | **`22m 1.9s`** | total benchmark wall-clock time (1321.9s) |
| **Reasoning Ratio** | **`0.512`** | 51.2% of generated tokens spent reasoning |
| **Quality / Time Efficiency** | **`60.1 pts`** | intelligence × time efficiency × token economy (0-100 scale) |

## Token Consumption

| Phase | Input (prompt) | Output (completion) | Reasoning | Total |
|---|---|---|---|---|
| Throughput single (1x) | 114 | 2,403 | 445 | 2,517 |
| Throughput 4x (3 reps) | 342 | 7,220 | 1,451 | 7,562 |
| Throughput 8x (3 reps) | 684 | 15,246 | 3,063 | 15,930 |
| Intelligence suites | 68,667 | 85,153 | 51,423 | 153,820 |
| **Total** | **69,807** | **110,022** | **56,382** | **179,829** |

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
| **GPQA Diamond (Physics / Chem / Bio)** | **`100.0%`** | 12 / 12 | Google-proof PhD-level deduction & domain reasoning |

**Failed Scenarios:** `none`

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
| **Humanity's Last Exam (HLE)** | **`80.0%`** | 8 / 10 | Game theory, algebraic topology, provability logic, black holes, genetics |

**Failed Scenarios:** `hle_01 (got 30, expected 35), hle_03 (got (D) [last choice], expected (B))`

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
| **AA-Omniscience Traps** | **`60.0%`** | 6 / 10 | Counterfactual false premises, fictional entities, precise scientific recall |

**Failed Scenarios:** `omni_02 (hallucinated false premise instead of restraining), omni_03 (missing expected facts: ['70', '173.05']), omni_06 (missing expected facts: ['1.380649', '10^-23', 'e-23']), omni_09 (missing expected facts: ['0', 'zero', 'none'])`

## Scientific Python Computing (SciCode)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **SciCode Scientific Programming** | **`100.0%`** | 6 / 6 | Quantum purity, Lennard-Jones, RK4 integrator, diffusion & matrix math |

**Failed Scenarios:** `none`

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
METRIC tokens_per_second=97.594
METRIC conc8_tps=97.594
METRIC conc4_tps=94.289
METRIC single_stream_tps=41.970
METRIC time_to_first_token_ms=12054.730
METRIC total_duration_seconds=1321.938
METRIC smart_composite_score=0.9074
METRIC aa_intelligence_index=0.8963
METRIC tool_call_accuracy=0.8966
METRIC ifeval_accuracy=1.0000
METRIC gsm8k_accuracy=0.8333
METRIC gpqa_accuracy=1.0000
METRIC humaneval_accuracy=1.0000
METRIC critpt_accuracy=1.0000
METRIC hle_accuracy=0.8000
METRIC banking_accuracy=1.0000
METRIC gdpval_accuracy=1.0000
METRIC omniscience_accuracy=0.6000
METRIC scicode_accuracy=1.0000
METRIC terminal_accuracy=0.6667
METRIC lcr_accuracy=1.0000
METRIC quality_per_time=60.1116
METRIC reasoning_ratio=0.5125
METRIC total_tokens=179829
```
