---
model: "ornith-1.5-35b-a3b-nvfp4"
device: "DGX-Spark"
engine: "vLLM (v0.1.dev20003+gad848fc41.d20260815)"
endpoint: "http://192.168.1.5:8888"
date: "2026-08-20T08:22:46.502792+00:00"
tokens_per_second: 78.017
single_stream_tps: 34.623
time_to_first_token_ms: 0.000
tool_call_accuracy: 1.0000
agentic_accuracy: null
reasoning_ratio: 0.0000
---

# Benchmark Report: ornith-1.5-35b-a3b-nvfp4 on DGX-Spark

- **Date:** 2026-08-20 08:22:46 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM (v0.1.dev20003+gad848fc41.d20260815)`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `ornith-1.5-35b-a3b-nvfp4`
- **Thinking Mode:** `on`
- **Concurrency:** `4` (repeats: `1`)
- **Seed:** `42`

## Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **4-Concurrent Throughput** | **`78.02 tok/s`** | median of 1 reps (spread: 78.0–78.0 tok/s) |
| **Single-Stream Throughput** | **`34.62 tok/s`** | 512 tokens generated |
| **Mean TTFT (Concurrent)** | **`0.0 ms`** | time to first token |
| **Reasoning Ratio** | **`0.000`** | 0.0% of generated tokens spent reasoning |

## Tool-Calling & Agentic Accuracy

| Category | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **Overall Tool-Call Accuracy** | **`100.0%`** | 1 / 1 | BFCL-style exact match & tau-bench evaluation |
| **Simple + Parallel** | **`100.0%`** | - | Single-call arg extraction + parallel multi-call |

**Failed Scenarios:** `none`

## Machine-Readable Metrics

```
METRIC tokens_per_second=78.017
METRIC single_stream_tps=34.623
METRIC time_to_first_token_ms=0.000
METRIC tool_call_accuracy=1.0000
METRIC reasoning_ratio=0.0000
```
