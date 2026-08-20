---
model: "ornith-1.5-35b-a3b-nvfp4"
device: "DGX-Spark"
engine: "vLLM"
endpoint: "http://192.168.1.5:8888"
date: "2026-08-20T14:34:18.063788+00:00"
tokens_per_second: 146.616
conc8_tps: 146.616
conc4_tps: 76.676
single_stream_tps: 34.192
time_to_first_token_ms: 0.000
total_duration_seconds: 37.888
smart_composite_score: 0.0000
tool_call_accuracy: null
ifeval_accuracy: null
gsm8k_accuracy: null
gpqa_accuracy: 0.0000
humaneval_accuracy: null
reasoning_ratio: 0.0000
---

# Benchmark Report: ornith-1.5-35b-a3b-nvfp4 on DGX-Spark

- **Date:** 2026-08-20 14:34:18 UTC
- **Device / GPU:** `DGX-Spark`
- **Serving Engine:** `vLLM`
- **Endpoint:** `http://192.168.1.5:8888`
- **Model:** `ornith-1.5-35b-a3b-nvfp4`
- **Thinking Mode:** `on`
- **Total Execution Time:** **`37.9s`** (37.9s)
- **Concurrency Tiers:** `Single (1x)`, `4-Concurrent`, `8-Concurrent` (repeats: `1`)
- **Seed:** `42`
- **🧠 Composite Intelligence Score:** **`0.0%`**

## ⚡ Throughput Performance

| Metric | Value | Details |
|---|---|---|
| **8-Concurrent Throughput** | **`146.62 tok/s`** | median of 1 reps (spread: 146.6–146.6 tok/s) |
| **4-Concurrent Throughput** | **`76.68 tok/s`** | median of 1 reps (spread: 76.7–76.7 tok/s) |
| **Single-Stream Throughput** | **`34.19 tok/s`** | 512 tokens generated |
| **Mean TTFT (8-Concurrent)** | **`0.0 ms`** | time to first token |
| **Total Execution Time** | **`37.9s`** | total benchmark wall-clock time (37.9s) |
| **Reasoning Ratio** | **`0.000`** | 0.0% of generated tokens spent reasoning |

## 🔬 PhD Science Reasoning (GPQA Diamond)

| Benchmark | Accuracy | Correct / Total | Details |
|---|---|---|---|
| **GPQA Diamond (Physics / Chem / Bio)** | **`0.0%`** | 0 / 2 | Google-proof PhD-level deduction & domain reasoning |

**Failed Questions:** `gpqa_01 (no choice (A/B/C/D) extracted), gpqa_02 (no choice (A/B/C/D) extracted)`

## 📊 Machine-Readable Metrics

```
METRIC tokens_per_second=146.616
METRIC conc8_tps=146.616
METRIC conc4_tps=76.676
METRIC single_stream_tps=34.192
METRIC time_to_first_token_ms=0.000
METRIC total_duration_seconds=37.888
METRIC smart_composite_score=0.0000
METRIC gpqa_accuracy=0.0000
METRIC reasoning_ratio=0.0000
```
