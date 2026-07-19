# Arm AI Optimization Challenge – P2 Model & Inference Lead

# Final Technical Documentation

## Executive Summary

This document summarizes the implementation and evaluation of the P2 (Model & Inference Lead) workstream for the Arm AI Optimization Challenge.

The project optimized Qwen2.5 language models for Arm-based AWS Graviton processors using **llama.cpp**, quantization, speculative decoding, and REST API deployment.

---

# Objectives

- Build llama.cpp on Arm64
- Quantize models for efficient inference
- Benchmark FP16 vs quantized models
- Implement speculative decoding
- Deploy REST API
- Evaluate performance across prompts
- Produce reproducible benchmarking

---

# Hardware & Software

| Component | Value |
|-----------|-------|
| Platform | AWS EC2 Graviton (Arm64) |
| OS | Ubuntu Linux |
| CPU | 2 vCPUs |
| RAM | ~4 GB |
| Framework | llama.cpp |
| Build | b10068 (571d0d540) |

---

# Models

## Target

- Qwen2.5-1.5B-Instruct
- Q4_K_M

## Draft

- Qwen2.5-0.5B-Instruct
- IQ4_XS

---

# Workflow

```text
FP16 Models
      │
      ▼
Quantization
      │
      ▼
Benchmarking
      │
      ▼
Speculative Decoding
      │
      ▼
REST API Deployment
      │
      ▼
Prompt Evaluation
```

---

# System Architecture

```text
                    User
                      │
                      ▼
               REST API (HTTP)
                      │
                llama-server
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
 Draft Model                 Target Model
 Qwen2.5-0.5B                Qwen2.5-1.5B
 IQ4_XS                      Q4_K_M
        │                          ▲
        └──── Proposed Tokens ─────┘
                    │
          Verification & Acceptance
                    │
                    ▼
             Final Generated Output
```

---

# Quantization Results

## 0.5B

| Model | Size | Prompt tok/s | Gen tok/s |
|------|------:|-------------:|-----------:|
| FP16 | - | 67.80 | 37.88 |
| IQ4_XS | 409 MB | 127.17 | 81.83 |
| Q4_K_M | 469 MB | 68.61 | 51.47 |
| Q5_K_M | 498 MB | 63.70 | 45.60 |

## 1.5B

| Model | Prompt tok/s | Gen tok/s |
|------|-------------:|-----------:|
| FP16 | 19.48 | 12.63 |
| Q4_K_M | 45.90 | 22.04 |

Observed:
- ~69% model size reduction
- 2.36× prompt improvement
- 1.74× generation improvement

---

# Standard llama-bench Results

| Test | Throughput |
|------|-----------:|
| Prompt (pp512) | 46.12 ±0.01 tok/s |
| Generation (tg128) | 22.14 ±0.03 tok/s |

---

# Speculative Decoding

Configuration:

```bash
./llama-cli \
-m target.gguf \
--spec-type draft-simple \
--spec-draft-model draft.gguf \
--spec-draft-n-max 2
```

## Acceptance Metrics

| Metric | Value |
|------|------:|
| Draft Tokens | 92 |
| Accepted | 40 |
| Rejected | 52 |
| Acceptance Rate | 43.48% |
| Mean Accepted Sequence | 1.87 tokens |

Accepted : █████████████ 43%

Rejected : █████████████████ 57%

---

# Baseline vs Speculative

| Metric | Baseline | Speculative |
|------|---------:|------------:|
| Prompt tok/s | 64.8 | 48.5 |
| Generation tok/s | 21.9 | 10.0 |
| Acceptance Rate | - | 43.48% |

### Observation

Speculative decoding functioned correctly but reduced throughput on the 2-vCPU Graviton instance because the draft and target models competed for limited CPU resources. On larger Arm systems (4–8 vCPUs), speculative decoding is expected to provide measurable speedups.

---

# REST API Deployment

```bash
./llama-server \
-m ~/quantized/qwen2.5-1.5b-instruct-q4_k_m.gguf \
--host 0.0.0.0 \
--port 8080
```

Example:

```bash
curl http://localhost:8080/completion \
-H "Content-Type: application/json" \
-d '{"prompt":"Explain quantization","n_predict":50}'
```

---

# Prompt Evaluation

Validated using:
- General Knowledge
- Programming
- Reasoning
- Summarization
- Creative Writing

Responses were coherent with stable throughput (~22 tok/s generation).

---

# Reproducibility

Commands executed:

- Build llama.cpp
- Quantize models
- llama-bench
- llama-cli
- Speculative decoding
- llama-server
- curl API testing

---

# Limitations

- Limited to 2-vCPU Graviton instance.
- Speculative decoding overhead exceeded benefits.
- No GPU acceleration.

---

# Future Work

- Benchmark on c7g.large
- Benchmark on c7g.xlarge
- Benchmark on c7g.2xlarge
- Compare scalability
- Add latency measurements
- Evaluate additional draft models
- Investigate dynamic draft lengths

---

# Conclusion

The project successfully implemented an optimized inference pipeline for Arm processors using llama.cpp. Quantization significantly improved throughput while preserving model quality. Speculative decoding was implemented and validated with a 43.48% acceptance rate; however, on the available 2-vCPU hardware it did not improve end-to-end throughput due to CPU contention. The REST API deployment, benchmarking, and reproducibility artifacts provide a solid foundation for further experimentation on larger Arm systems.

Final Selection Rationale

• Draft Model (IQ4_XS): Selected because it achieved the highest inference throughput while maintaining acceptable response quality.

• Target Model (Q4_K_M): Selected because it provided the best trade-off between model size, inference speed, and output quality.