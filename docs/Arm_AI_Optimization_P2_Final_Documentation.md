# Arm AI Optimization Challenge

# Technical Documentation

## Executive Summary

This document presents the implementation, optimization, benchmarking, and evaluation of large language model inference on Arm-based processors using **llama.cpp**. The project focuses on improving inference efficiency through model quantization, benchmarking, speculative decoding, and REST API deployment while maintaining high-quality text generation.

The implementation utilizes Qwen2.5 language models running on an AWS EC2 Graviton (Arm64) instance and evaluates various quantization techniques to identify the optimal balance between inference speed, memory usage, and output quality.

---

# Project Objectives

The primary objectives of this project are:

- Deploy and evaluate Large Language Models on Arm64 architecture.
- Optimize inference using GGUF quantized models.
- Benchmark FP16 and multiple quantization formats.
- Evaluate speculative decoding for faster inference.
- Deploy the optimized model through a REST API.
- Measure throughput and analyze performance.
- Produce reproducible benchmarking results.

---

# Hardware & Software Configuration

| Component | Specification |
|-----------|--------------|
| Platform | AWS EC2 Graviton (Arm64) |
| Operating System | Ubuntu Linux |
| CPU | 2 vCPUs |
| Memory | ~4 GB RAM |
| Inference Framework | llama.cpp |
| Build Version | b10068 (571d0d540) |

---

# Models Used

## Target Model

- Model: **Qwen2.5-1.5B-Instruct**
- Quantization: **Q4_K_M**

## Draft Model

- Model: **Qwen2.5-0.5B-Instruct**
- Quantization: **IQ4_XS**

### Model Selection Rationale

The selected models share the same tokenizer, making them fully compatible for speculative decoding.

- **Draft Model (IQ4_XS):**
  - Small model size
  - Highest inference throughput
  - Suitable for rapid draft token generation

- **Target Model (Q4_K_M):**
  - Good balance between accuracy and performance
  - Reduced memory footprint
  - High-quality text generation

---

# System Workflow

```text
FP16 Models
      │
      ▼
Model Quantization
      │
      ▼
Performance Benchmarking
      │
      ▼
Speculative Decoding
      │
      ▼
REST API Deployment
      │
      ▼
Prompt Evaluation
      │
      ▼
Performance Analysis
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

# Quantization Evaluation

## Qwen2.5-0.5B

| Model | Size | Prompt tok/s | Generation tok/s |
|------|------:|-------------:|-----------------:|
| FP16 | - | 67.80 | 37.88 |
| IQ4_XS | 409 MB | 127.17 | 81.83 |
| Q4_K_M | 469 MB | 68.61 | 51.47 |
| Q5_K_M | 498 MB | 63.70 | 45.60 |

---

## Qwen2.5-1.5B

| Model | Prompt tok/s | Generation tok/s |
|------|-------------:|-----------------:|
| FP16 | 19.48 | 12.63 |
| Q4_K_M | 45.90 | 22.04 |

### Key Observations

- Approximately **69% reduction** in model size after quantization.
- Approximately **2.36× increase** in prompt processing throughput.
- Approximately **1.74× increase** in generation throughput.
- Quantization substantially reduced memory usage while maintaining satisfactory response quality.

---

# Benchmark Results

Standard benchmarking was performed using **llama-bench**.

| Test | Throughput |
|------|-----------:|
| Prompt Processing (pp512) | 46.12 ±0.01 tok/s |
| Token Generation (tg128) | 22.14 ±0.03 tok/s |

The optimized Q4_K_M model demonstrated a significant improvement over FP16 inference, making it the preferred deployment configuration.

---

# Speculative Decoding

Speculative decoding was implemented using a lightweight draft model together with the larger target model.

Configuration:

```bash
./llama-cli \
-m target.gguf \
--spec-type draft-simple \
--spec-draft-model draft.gguf \
--spec-draft-n-max 2
```

## Acceptance Statistics

| Metric | Value |
|------|------:|
| Draft Tokens | 92 |
| Accepted Tokens | 40 |
| Rejected Tokens | 52 |
| Acceptance Rate | 43.48% |
| Mean Accepted Sequence Length | 1.87 Tokens |

Visualization

```
Accepted : █████████████ 43%

Rejected : █████████████████ 57%
```

---

# Baseline vs Speculative Decoding

| Metric | Baseline | Speculative |
|------|---------:|------------:|
| Prompt Throughput | 64.8 tok/s | 48.5 tok/s |
| Generation Throughput | 21.9 tok/s | 10.0 tok/s |
| Acceptance Rate | - | 43.48% |

## Analysis

Speculative decoding operated correctly and successfully verified draft tokens using the target model. However, on the available 2-vCPU AWS Graviton instance, the computational overhead of executing both models simultaneously outweighed the performance benefits.

Larger Arm instances with additional CPU cores are expected to provide significantly better speculative decoding performance.

---

# REST API Deployment

The optimized model was deployed using **llama-server**.

Launch command:

```bash
./llama-server \
-m ~/quantized/qwen2.5-1.5b-instruct-q4_k_m.gguf \
--host 0.0.0.0 \
--port 8080
```

Example API request:

```bash
curl http://localhost:8080/completion \
-H "Content-Type: application/json" \
-d '{
"prompt":"Explain quantization",
"n_predict":50
}'
```

The REST interface enables external applications to interact with the optimized language model through standard HTTP requests.

---

# Prompt Evaluation

Inference quality was evaluated across multiple prompt categories.

- General Knowledge
- Programming
- Logical Reasoning
- Summarization
- Creative Writing

The optimized model consistently generated coherent and contextually relevant responses while maintaining an average generation throughput of approximately **22 tokens/second**.

---

# Reproducibility

The complete workflow can be reproduced using the following steps:

1. Build `llama.cpp`
2. Download Qwen2.5 models
3. Convert to GGUF (if required)
4. Quantize models
5. Run `llama-bench`
6. Evaluate inference using `llama-cli`
7. Execute speculative decoding experiments
8. Deploy `llama-server`
9. Validate REST API using `curl`
10. Compare benchmark results

---

# Project Outcomes

The project successfully demonstrated:

- Efficient Arm64 inference using llama.cpp.
- Significant improvements through model quantization.
- Successful implementation of speculative decoding.
- Benchmark-driven model selection.
- REST API deployment for production-style inference.
- Reproducible benchmarking workflow.

---

# Limitations

- Evaluation performed on a 2-vCPU AWS Graviton instance.
- CPU resources limited speculative decoding performance.
- No GPU acceleration was available.
- Latency measurements were outside the current evaluation scope.

---

# Future Work

Future improvements may include:

- Benchmarking on c7g.large
- Benchmarking on c7g.xlarge
- Benchmarking on c7g.2xlarge
- Measuring latency and memory utilization
- Evaluating additional draft model combinations
- Investigating adaptive speculative decoding strategies
- Comparing Arm performance across larger instances
- Automating benchmarking and report generation

---

# Conclusion

This project demonstrates an optimized large language model inference pipeline for Arm-based processors using **llama.cpp**. Quantization substantially improved throughput and reduced memory consumption while preserving response quality. Speculative decoding was successfully implemented and validated, achieving an acceptance rate of **43.48%**. Although the available 2-vCPU environment limited its overall speedup, the implementation establishes a scalable foundation for future deployment on larger Arm systems.

The combination of quantization, benchmarking, speculative decoding, REST API deployment, and reproducible experimentation provides a comprehensive workflow for efficient LLM inference on Arm architecture.