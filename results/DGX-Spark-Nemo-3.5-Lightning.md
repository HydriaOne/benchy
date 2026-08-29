---
model: "Nemo-3.5-Lightning"
device: "DGX-Spark"
engine: "SGLang"
quant: "NVFP4"
endpoint: "http://192.168.1.5:8888"
thinking: "auto"
date: "2026-08-29T14:03:05.066905+00:00"
tokens_per_second: 311.004
conc8_tps: 311.004
conc4_tps: 214.933
single_stream_tps: 120.413
time_to_first_token_ms: 350.261
total_duration_seconds: 492.162
smart_composite_score: 0.7133
aa_intelligence_index: 0.8056
tool_call_accuracy: 0.6897
ifeval_accuracy: 0.1667
gsm8k_accuracy: 0.5000
gpqa_accuracy: 0.8333
humaneval_accuracy: 0.6667
critpt_accuracy: 1.0000
hle_accuracy: 0.8000
banking_accuracy: 1.0000
gdpval_accuracy: 0.8333
omniscience_accuracy: 0.7000
scicode_accuracy: 0.6667
terminal_accuracy: 0.6667
lcr_accuracy: 0.7500
reasoning_ratio: 0.7893
quality_per_time: 74.0931
total_tokens: 208370
input_tokens: 81812
output_tokens: 126558
---

# Benchmark Report: Nemo-3.5-Lightning on DGX-Spark

- **Date:** 2026-08-29 14:03:05 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `SGLang`
- **Quantization:** `NVFP4`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `Nemo-3.5-Lightning`
- **Thinking Mode:** `auto`
- **Total Execution Time:** **`8m 12.2s`** (492.2s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `3`)
- **Seed:** `42`
- **Composite Intelligence Score:** **`71.3%`**
- **Artificial Analysis Intelligence Index:** **`80.6%`**

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`311.00 tok/s`** | median of 3 reps (spread: 293.8–346.0 tok/s) |
| **4-Concurrent Throughput** | **`214.93 tok/s`** | median of 3 reps (spread: 202.4–232.9 tok/s) |
| **Single-Stream Throughput** | **`120.41 tok/s`** | 3855 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`350.3 ms`** | time to first token |
| **Total Execution Time** | **`8m 12.2s`** | total benchmark wall-clock time (492.2s) |
| **Reasoning Ratio** | **`0.789`** | 78.9% of generated tokens spent reasoning |
| **Quality / Time Efficiency** | **`69.6 pts`** | intelligence / (tokens × seconds) × 10¹⁰ (0-100 scale) |

## Token Consumption

| Phase | Input (prompt) | Output (completion) | Reasoning | Total |
|---|---|---|---|---|
| Throughput single (1x) | 142 | 3,855 | 2,776 | 3,997 |
| Throughput 4x (3 reps) | 426 | 11,654 | 8,499 | 12,080 |
| Throughput 8x (3 reps) | 852 | 21,867 | 15,278 | 22,719 |
| Intelligence suites | 80,392 | 89,182 | 73,333 | 169,574 |
| **Total** | **81,812** | **126,558** | **99,886** | **208,370** |

## Tool-Calling & Agentic Evaluation (BFCL & tau-bench)

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool Accuracy** | **`69.0%`** | 20 / 29 | BFCL exact-match, distractor selection & multi-turn |
| **Single-Turn (Simple / Parallel / Restraint / Complex / Distractors)** | **`61.9%`** | 13 / 21 | Tool selection, args, restraint, distractors & schemas |
| **Agentic Multi-Turn (Execution, Chains & Error Recovery)** | **`87.5%`** | 7 / 8 | Multi-step dependency chains & stateful rollback |

**Failed Scenarios:** `s05, p01, p02, p03, p04, p05, p06, m04, nt03`

## Instruction Following (Google IFEval Hard)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **IFEval Hard Constraints** | **`16.7%`** | 1 / 6 | Multi-constraint conjunctions, JSON ranges, negative constraints |

**Failed Scenarios:** `ifeval_h01 (got 0 paragraphs (expected exactly 3)), ifeval_h03 (word count 0 outside [60, 90]), ifeval_h04 (no markdown table found), ifeval_h05 (missing <audit> or </audit> tags), ifeval_h06 (too short (0 < 50 words))`

## Math Reasoning (AIME & Competition Math)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AIME / Competition Math** | **`50.0%`** | 3 / 6 | Modular arithmetic, combinatorics, algebra & geometry proofs |

**Failed Scenarios:** `aime_02 (got 144, expected 15), aime_03 (no integer answer found in response), aime_05 (no integer answer found in response)`

## PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`83.3%`** | 10 / 12 | Google-proof PhD-level deduction & domain reasoning |

**Failed Scenarios:** `gpqa_01 (no choice (A/B/C/D) extracted), gpqa_04 (no choice (A/B/C/D) extracted)`

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
| **Humanity's Last Exam (HLE)** | **`80.0%`** | 8 / 10 | Game theory, algebraic topology, provability logic, black holes, genetics |

**Failed Scenarios:** `hle_01 (got 30 (last integer), expected 35), hle_03 (no choice (A/B/C/D) extracted)`

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
| **SciCode Scientific Programming** | **`66.7%`** | 4 / 6 | Quantum purity, Lennard-Jones, RK4 integrator, diffusion & matrix math |

**Failed Scenarios:** `scicode_01 (assertion failed: SyntaxError: unterminated string literal (det), scicode_05 (assertion failed: SyntaxError: '(' was never closed)`

## Interactive CLI & Terminal Agent (Terminal-Bench v4.0)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Terminal-Bench CLI Agent** | **`66.7%`** | 4 / 6 | VFS log triage, nginx syntax repair, git merge conflict resolution, JSON migration |

**Failed Scenarios:** `term_02 (file '/etc/nginx/conf.d/api.conf' missing expected string: '8000;'), term_05 (file '/app/services/auth.env' missing expected string: 'API_KEY=${NEW_API_KEY}')`

## Long-Context Reasoning & Retrieval (AA-LCR)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **AA-LCR Long Context** | **`75.0%`** | 3 / 4 | Multi-document timeline synthesis, incident root cause, procurement liability |

**Failed Scenarios:** `lcr_02 (failed to extract needle 'auth-token-revocation-worker-node-4')`

## Machine-Readable Metrics

```
METRIC tokens_per_second=311.004
METRIC conc8_tps=311.004
METRIC conc4_tps=214.933
METRIC single_stream_tps=120.413
METRIC time_to_first_token_ms=350.261
METRIC total_duration_seconds=492.162
METRIC smart_composite_score=0.7133
METRIC aa_intelligence_index=0.8056
METRIC tool_call_accuracy=0.6897
METRIC ifeval_accuracy=0.1667
METRIC gsm8k_accuracy=0.5000
METRIC gpqa_accuracy=0.8333
METRIC humaneval_accuracy=0.6667
METRIC critpt_accuracy=1.0000
METRIC hle_accuracy=0.8000
METRIC banking_accuracy=1.0000
METRIC gdpval_accuracy=0.8333
METRIC omniscience_accuracy=0.7000
METRIC scicode_accuracy=0.6667
METRIC terminal_accuracy=0.6667
METRIC lcr_accuracy=0.7500
METRIC quality_per_time=74.0931
METRIC reasoning_ratio=0.7893
METRIC total_tokens=208370
```
