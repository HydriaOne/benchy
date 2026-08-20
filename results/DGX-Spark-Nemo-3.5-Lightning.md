---
model: "Nemo-3.5-Lightning"
device: "DGX-Spark"
engine: "SGLang"
endpoint: "http://192.168.1.5:8888"
date: "2026-08-20T07:15:00.000000+00:00"
tokens_per_second: 196.100
conc8_tps: 196.100
conc4_tps: 137.590
single_stream_tps: 61.528
time_to_first_token_ms: 260.422
smart_composite_score: 0.7000
tool_call_accuracy: 0.7000
agentic_accuracy: 1.0000
reasoning_ratio: 0.6481
---

# Benchmark Report: Nemo-3.5-Lightning on DGX-Spark

- **Date:** 2026-08-20 07:15:00 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `SGLang`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `Nemo-3.5-Lightning`
- **Thinking Mode:** `on`
- **Concurrency:** `4` (repeats: `3`)
- **Seed:** `42`

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`196.10 tok/s`** | concurrency sweep (8 streams) |
| **4-Concurrent Throughput** | **`137.59 tok/s`** | median of 3 reps (spread: 123.8–148.3 tok/s) |
| **Single-Stream Throughput** | **`61.53 tok/s`** | 3882 tokens generated |
| **Mean TTFT (Concurrent)** | **`260.4 ms`** | time to first token |
| **Reasoning Ratio** | **`0.648`** | 64.8% of generated tokens spent reasoning |

## Tool-Calling & Agentic Accuracy

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool-Call Accuracy** | **`70.0%`** | 14 / 20 | BFCL-style exact match & tau-bench evaluation |
| **Simple + Parallel** | **`57.1%`** | - | Single-call arg extraction (8/8) + parallel multi-call (0/6) |
| **Agentic Multi-Turn** | **`100.0%`** | 6 / 6 | Multi-step tool execution & final answer grading |

**Failed Scenarios:** `p01, p02, p03, p04, p05, p06`

## Concurrency Scaling

| Concurrency | Throughput (tok/s) |
|---|---|
| 1 streams | 61.9 tok/s |
| 2 streams | 93.0 tok/s |
| 4 streams | 141.5 tok/s |
| 8 streams | 196.1 tok/s |
| 16 streams | 279.8 tok/s |

## Machine-Readable Metrics

```
METRIC tokens_per_second=137.590
METRIC single_stream_tps=61.528
METRIC time_to_first_token_ms=260.422
METRIC tool_call_accuracy=0.7000
METRIC agentic_accuracy=1.0000
METRIC reasoning_ratio=0.6481
```
