# Benchmark & Evaluation Module Completion Report

**Author:** Benchmark & Evaluation Lead
**Date:** July 2025
**Status:** Complete

---

## Objective

The Benchmark & Evaluation Lead was responsible for designing, implementing, and validating a benchmark runner capable of measuring LLM inference performance on llama.cpp. The module needed to:

- Send prompts to a running llama.cpp server
- Measure throughput (tokens/sec), latency, and resource usage
- Produce structured output in JSON and CSV formats
- Support configurable parameters (model, token count, trials, temperature)
- Integrate with the project's shared metrics library (`evaluation/metrics.py`)

---

## Completed Work

The benchmark runner is fully operational and has been validated on an AWS EC2 Graviton4 instance running llama.cpp with KleidiAI optimizations.

### Core Implementation

| Task | Status |
|------|--------|
| Connected benchmark runner to llama.cpp server | Done |
| Replaced OpenAI-compatible `/v1/chat/completions` endpoint with llama.cpp `/completion` endpoint | Done |
| Replaced OpenAI-style payload with llama.cpp-native payload format | Done |
| Fixed response parser to handle llama.cpp completion API response structure | Done |
| Fixed token counting to use `tokens_predicted` from server response | Done |
| Throughput calculation verified and working | Done |
| JSON output with metadata and aggregate statistics | Done |
| CSV output with per-prompt results | Done |
| Benchmark executes all 31 prompts across configurable trials | Done |
| Supports configurable `--max-tokens` via CLI | Done |
| Successfully tested on EC2 with Q4_K_M quantized model | Done |

### Endpoint and Payload Changes

The original benchmark targeted the OpenAI-compatible `/v1/chat/completions` endpoint. The llama.cpp server exposes a different API surface. The following changes were made:

| Component | Original | Modified |
|-----------|----------|----------|
| Endpoint | `/v1/chat/completions` | `/completion` |
| Prompt field | `messages: [{"role":"user","content":"..."}]` | `prompt: "..."` |
| Token limit | `max_tokens` | `n_predict` |
| Response content | `choices[0].message.content` | `content` |
| Token count | `usage.completion_tokens` | `tokens_predicted` |
| Prompt tokens | `usage.prompt_tokens` | `tokens_evaluated` |

### Parser Fix

The initial implementation returned `tokens=0` and `tps=0.0` for every prompt because the response parser expected OpenAI-format fields. The fix involved:

1. Reading `data.get("content", "")` for response text
2. Reading `data.get("tokens_predicted", 0)` for generated token count
3. Reading `data.get("tokens_evaluated", 0)` for prompt token count
4. Fixing indentation errors in the `send_request` method that prevented the response from being parsed correctly

---

## Performance Observations

Benchmark runs were executed on an AWS EC2 `c8g.large` instance (Graviton4, 2 vCPU, 4 GB RAM) with the `Q4_K_M` quantized model.

### Observed Metrics

| Metric | Approximate Value |
|--------|-------------------|
| Tokens per second | 18-20 TPS |
| Tokens generated per prompt | 64 (configured via `--max-tokens`) |
| Duration per prompt | 3-4 seconds |
| Prompts per trial | 31 |
| Trials per run | 3 |
| Total benchmark requests | 93 |
| Total benchmark duration | ~5-6 minutes |

### Per-Prompt Example

```
[sqa_01] tokens=64 ttft=0.0ms tps=19.2 duration=3.33s
```

---

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| TTFT reported as 0.0ms | Time to first token is not measured | The benchmark uses the non-streaming `/completion` endpoint. Streaming support (`stream: true`) would be required for accurate TTFT measurement. |
| CPU and memory metrics depend on `psutil` | Metrics unavailable if `psutil` is not installed or process access is denied | Ensure `psutil` is installed (`pip install psutil`) and the benchmark runs as the same user that owns the server process. |
| Non-streaming endpoint | No inter-token latency (ITL) distribution | With streaming enabled, per-token timestamps could be captured for detailed latency analysis. |
| Single-threaded request processing | Sequential prompt execution | Parallel benchmarking could reduce total wall-clock time. |

---

## Repository Changes

### Files Modified

| File | Changes |
|------|---------|
| `benchmark/benchmark.py` | Replaced OpenAI endpoint with llama.cpp `/completion` endpoint; fixed response parser; fixed indentation errors in `send_request` method; added `n_predict` and `stop` parameters to payload |
| `results/` | Directory containing benchmark output (JSON and CSV files) |
| `docs/SETUP.md` | New — complete setup guide for new team members |
| `docs/PERSON3_COMPLETED_WORK.md` | New — this document |

### Files Unchanged

| File | Reason |
|------|--------|
| `evaluation/metrics.py` | Pure metric computation functions — no changes needed |
| `evaluation/prompts.json` | Prompt dataset — unchanged |
| `evaluation/evaluate.py` | Evaluation logic — unchanged |
| `setup.sh` | Environment setup — unchanged |
| `README.md` | Project documentation — unchanged |

---

## Validation Performed

| Test | Result |
|------|--------|
| Health endpoint (`curl /health`) | `{"status":"ok"}` |
| Completion endpoint (`curl /completion`) | JSON response with `content` and `tokens_predicted` fields |
| Benchmark execution with 1 trial | 31/31 prompts succeeded |
| Benchmark execution with 3 trials | 93/93 prompts succeeded |
| JSON output generation | Valid JSON with metadata, aggregate statistics, and per-prompt results |
| CSV output generation | Valid CSV with correct headers and data rows |
| Multiple prompt execution | All 31 prompts across 5 categories executed successfully |
| Token counting accuracy | `tokens_predicted` matches expected output range |
| Throughput calculation | TPS values consistent with observed performance (~18-20 TPS) |

---

## Deliverables

| Deliverable | Status |
|-------------|--------|
| Benchmark runner (`benchmark/benchmark.py`) | Complete |
| Performance metrics (tokens/sec, latency, duration) | Complete |
| CSV reports (`results/*.csv`) | Complete |
| JSON reports (`results/*.json`) | Complete |
| Setup documentation (`docs/SETUP.md`) | Complete |
| Completion report (`docs/PERSON3_COMPLETED_WORK.md`) | Complete |

---

## Future Improvements

| Improvement | Priority | Description |
|-------------|----------|-------------|
| Streaming TTFT measurement | High | Enable `stream: true` in the benchmark payload and capture per-token timestamps from SSE chunks for accurate TTFT and ITL distribution |
| Parallel benchmarking | Medium | Execute multiple prompts concurrently using `asyncio` or `threading` to reduce total benchmark duration |
| Additional benchmark prompts | Medium | Expand the prompt dataset beyond 31 prompts for more comprehensive evaluation |
| Visualization dashboards | Low | Generate charts from benchmark results using `matplotlib` or `plotly` |
| Automatic chart generation | Low | Add a post-benchmark step to produce latency histograms, throughput bar charts, and comparison plots |
| Prometheus/Grafana integration | Low | Export benchmark metrics to Prometheus for long-term monitoring and dashboarding |
| Speculative decoding benchmarks | Medium | Extend the benchmark to measure speculative decoding acceptance rates using `compute_acceptance_rate` from `evaluation/metrics.py` |

---

## Final Status

**The Benchmark & Evaluation module is complete and ready for integration with the remaining project modules.**

The benchmark runner successfully:

- Connects to a running llama.cpp server
- Executes all 31 evaluation prompts across configurable trials
- Reports accurate token counts, throughput, and duration
- Produces structured JSON and CSV output
- Handles errors gracefully with per-prompt error reporting
- Has been validated on AWS EC2 Graviton4 hardware

All deliverables have been produced and documented. The module is production-ready for hackathon judging and teammate handoff.
