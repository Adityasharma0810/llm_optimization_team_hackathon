# Setup Guide

Complete setup instructions for the LLM Optimization Hackathon project — KleidiAI-powered llama.cpp on ARM64.

---

## Project Overview

This repository provides high-performance LLM inference on ARM64 hardware using [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Arm KleidiAI](https://github.com/ARM-software/kleidiai) optimized kernels. KleidiAI provides hardware-accelerated int8 matrix multiply (i8mm), SVE/SVE2 vector operations, and dot product instructions for maximum throughput on Arm Neoverse V2 (Graviton4) and compatible processors.

The project includes:
- A one-command environment setup script (`setup.sh`)
- A benchmark runner for measuring inference performance
- Evaluation prompts and metrics for systematic testing
- Hardware capability detection and verification tools

---

## Prerequisites

### Operating System

| Requirement | Version |
|-------------|---------|
| OS | Ubuntu 22.04 LTS or newer (24.04, 26.04) |
| Architecture | ARM64 (aarch64) only |

### Software

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| Python | 3.10+ | Benchmark runner, metrics, evaluation scripts |
| Git | 2.x | Repository cloning, submodule management |
| CMake | 3.21+ | Build system for llama.cpp |
| GCC | 11+ | C/C++ compiler |
| G++ | 11+ | C++ compiler |
| Ninja | Any | Build accelerator (installed by setup.sh) |
| curl | Any | Server health checks, API testing |
| wget | Any | Model downloads |

### Recommended Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 2 GB | 4 GB+ |
| Disk | 5 GB free | 10 GB+ free |
| CPU | ARMv8.2+ with NEON | ARMv9.0+ with SVE2 + i8mm |

### Tested Instances

| Instance | CPU | SVE | I8MM | KleidiAI Status |
|----------|-----|-----|------|-----------------|
| AWS EC2 c8g.large | Graviton4 (Neoverse V2) | Yes | Yes | Full |
| AWS EC2 c8g.xlarge | Graviton4 (Neoverse V2) | Yes | Yes | Full |
| AWS EC2 c7g.large | Graviton3 (Neoverse V1) | Yes | No | Partial |
| AWS EC2 t4g.small | Graviton2 (Neoverse N1) | No | No | Disabled |

---

## Clone Repository

```bash
git clone https://github.com/Adityasharma0810/llm_optimization_team_hackathon.git
cd llm_optimization_team_hackathon
```

---

## Python Environment

### Create a virtual environment

```bash
python3 -m venv venv
```

### Activate the virtual environment

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The key Python packages required are:
- `requests` — HTTP client for benchmark runner
- `psutil` — CPU and memory metrics collection

If a `requirements.txt` is not present, install manually:

```bash
pip install requests psutil
```

---

## Build llama.cpp

The `setup.sh` script automates the full build. However, to build manually:

### Clone llama.cpp

```bash
git clone --recursive https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
```

### Configure with CMake (KleidiAI enabled)

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_KLEIDIAI=ON \
  -DGGML_CPU_KLEIDIAI=ON
```

| Flag | Purpose |
|------|---------|
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build with full optimizations |
| `-DGGML_KLEIDIAI=ON` | Enable KleidiAI library integration |
| `-DGGML_CPU_KLEIDIAI=ON` | Enable KleidiAI CPU backend optimizations |

### Build

```bash
cmake --build build --config Release -j$(nproc)
```

The `-j$(nproc)` flag uses all available CPU cores for parallel compilation.

### Verify build

```bash
ls ~/llama.cpp/build/bin/llama-server
ls ~/llama.cpp/build/bin/llama-cli
```

---

## Download Model

llama.cpp requires GGUF-format models. Download a quantized model and place it in a known directory.

### Recommended location

```
/home/ubuntu/quantized/
```

### Example: download a model

```bash
mkdir -p /home/ubuntu/quantized
cd /home/ubuntu/quantized
```

Using `huggingface-cli`:

```bash
pip install huggingface-hub
huggingface-cli download TheBloke/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir .
```

Using `wget`:

```bash
wget https://huggingface.co/TheBloke/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -O /home/ubuntu/quantized/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### Verify the model file

```bash
ls -lh /home/ubuntu/quantized/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

## Start llama-server

```bash
~/llama.cpp/build/bin/llama-server \
  -m /home/ubuntu/quantized/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080
```

### Flag Reference

| Flag | Purpose |
|------|---------|
| `-m <path>` | Path to the GGUF model file |
| `--host 0.0.0.0` | Listen on all network interfaces (required for remote access) |
| `--port 8080` | TCP port for the HTTP server |

### Optional flags

| Flag | Purpose |
|------|---------|
| `-t 4` | Number of CPU threads (default: auto-detected) |
| `-c 2048` | Context size in tokens (default: 512) |
| `--n-predict 512` | Default max tokens to generate |

The server will print a startup message indicating it is ready to accept requests.

---

## Verify Server

### Health endpoint

```bash
curl http://localhost:8080/health
```

Expected output:

```json
{"status":"ok"}
```

### Completion endpoint

```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","n_predict":20}'
```

Expected output (abbreviated):

```json
{
  "content": "Hello! How can I help you today?",
  "tokens_predicted": 8,
  "tokens_evaluated": 1,
  ...
}
```

The `content` field contains the generated text. The `tokens_predicted` field reports how many tokens were generated. The `tokens_evaluated` field reports how many prompt tokens were processed.

---

## Running Benchmark

### Basic usage

```bash
python3 -m benchmark.benchmark \
  --server-url http://localhost:8080 \
  --model-name Q4_K_M \
  --max-tokens 64
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--server-url` | Yes | — | Base URL of the llama-server (e.g. `http://localhost:8080`) |
| `--model-name` | Yes | — | Model identifier for reporting (e.g. `Q4_K_M`) |
| `--trials` | No | `3` | Number of trials to run per prompt |
| `--temperature` | No | `0.0` | Sampling temperature (0.0 = deterministic) |
| `--max-tokens` | No | `512` | Maximum tokens to generate per response |
| `--timeout` | No | `120.0` | HTTP request timeout in seconds |
| `--output` | No | `results/` | Output directory for result files |
| `--no-json` | No | `false` | Skip JSON output |
| `--no-csv` | No | `false` | Skip CSV output |
| `--log-level` | No | `INFO` | Logging verbosity: DEBUG, INFO, WARNING, ERROR |

### Example: full benchmark run

```bash
python3 -m benchmark.benchmark \
  --server-url http://13.211.208.159:8080 \
  --model-name Q4_K_M \
  --trials 3 \
  --max-tokens 64 \
  --output results/
```

### Example: quick smoke test

```bash
python3 -m benchmark.benchmark \
  --server-url http://localhost:8080 \
  --model-name Q4_K_M \
  --trials 1 \
  --max-tokens 32 \
  --log-level DEBUG
```

---

## Output

Results are saved to the `results/` directory (configurable via `--output`).

### Files created

| File | Format | Description |
|------|--------|-------------|
| `benchmark_results_<timestamp>.json` | JSON | Full results with metadata, aggregate statistics, and per-prompt data |
| `benchmark_results_<timestamp>.csv` | CSV | Tabular results suitable for spreadsheet analysis |

### Metrics explained

| Metric | Description |
|--------|-------------|
| Tokens Generated | Number of tokens produced by the model for this prompt |
| Tokens/sec (TPS) | Throughput — tokens generated divided by total duration |
| Duration | Total wall-clock time for the request in seconds |
| Latency (ITL mean) | Mean inter-token latency — average time between consecutive tokens in milliseconds |
| TTFT | Time to first token in milliseconds (0.0 when using non-streaming endpoint) |
| CPU Usage | Process CPU utilization percentage at time of measurement |
| Memory Usage | Process resident set size (RSS) in megabytes |

### JSON structure

The JSON output contains:

```json
{
  "metadata": {
    "server_url": "http://localhost:8080",
    "model_name": "Q4_K_M",
    "trials": 3,
    "temperature": 0.0,
    "max_tokens": 64,
    "timestamp": "2025-07-20T12:00:00+00:00",
    "total_prompts": 31,
    "total_results": 93,
    "successful": 93,
    "failed": 0
  },
  "aggregate": {
    "tokens_per_second": { "mean": 18.5, "median": 18.2, ... },
    "ttft_ms": { "mean": 0.0, ... },
    "inter_token_latency_ms": { "mean": 54.1, ... },
    "duration_sec": { "mean": 3.5, ... }
  },
  "results": [ ... ]
}
```

---

## Common Errors

### Connection refused

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Cause:** The llama-server is not running or is listening on a different port.

**Solution:** Start the server and verify it is listening on the expected port:

```bash
curl http://localhost:8080/health
```

---

### 404 endpoint not found

```
HTTPError: 404 Client Error: Not Found
```

**Cause:** The benchmark is targeting a non-existent endpoint (e.g. `/v1/chat/completions` on a server that only exposes `/completion`).

**Solution:** Ensure `--server-url` points to the correct base URL. The benchmark uses the `/completion` endpoint.

---

### ModuleNotFoundError: psutil

```
ModuleNotFoundError: No module named 'psutil'
```

**Cause:** The `psutil` package is not installed in the active Python environment.

**Solution:**

```bash
pip install psutil
```

---

### Security Group closed (EC2)

```
curl: (7) Failed to connect to <ip> port 8080: Connection refused
```

**Cause:** AWS Security Group does not allow inbound traffic on port 8080.

**Solution:** Add an inbound rule to the EC2 Security Group:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| Custom TCP | TCP | 8080 | 0.0.0.0/0 |

---

### Server not running

```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8080)
```

**Cause:** The llama-server process has stopped or was never started.

**Solution:** Restart the server:

```bash
~/llama.cpp/build/bin/llama-server \
  -m /home/ubuntu/quantized/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080
```

---

### Wrong endpoint

```
tokens=0 tps=0.0
```

**Cause:** The server responded but the response format does not match what the parser expects. This can happen if the server exposes `/v1/chat/completions` instead of `/completion`.

**Solution:** Verify the server exposes the correct endpoint:

```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","n_predict":10}'
```

---

### IndentationError / return outside function

```
IndentationError: unexpected indent
SyntaxError: 'return' outside function
```

**Cause:** The benchmark source file has a syntax error, typically from a failed merge or manual edit.

**Solution:** Verify syntax:

```bash
python3 -m py_compile benchmark/benchmark.py
```

If errors are reported, restore the file from git or re-apply edits carefully.

---

### Benchmark hangs

**Cause:** The server is processing a request but not responding within the timeout period. This can happen with very long prompts or large `--max-tokens` values.

**Solution:**
1. Reduce `--max-tokens` (e.g. `--max-tokens 32`).
2. Check server logs for errors.
3. Verify the server is not overloaded:

```bash
curl http://localhost:8080/health
```

---

## Troubleshooting Checklist

Before reporting an issue, verify each item:

| Step | Command | Expected Result |
|------|---------|-----------------|
| 1. Health endpoint | `curl http://localhost:8080/health` | `{"status":"ok"}` |
| 2. Completion endpoint | `curl http://localhost:8080/completion -H "Content-Type: application/json" -d '{"prompt":"Hello","n_predict":10}'` | JSON with `"content"` field |
| 3. Python syntax check | `python3 -m py_compile benchmark/benchmark.py` | No output (success) |
| 4. Benchmark command | `python3 -m benchmark.benchmark --server-url http://localhost:8080 --model-name Q4_K_M --max-tokens 32 --trials 1` | Logs showing `tokens=N` where N > 0 |
| 5. Results directory | `ls results/` | `.json` and `.csv` files present |

### Expected success output

```
Loading prompts...
Loaded 31 prompts...
=== Trial 1/1 (31 prompts) ===
  [1/31] sqa_01 — Simple QA
[sqa_01] trial=0 category=short_qa
[sqa_01] tokens=32 ttft=0.0ms tps=18.5 duration=1.73s
...
========================================================================
  BENCHMARK SUMMARY
========================================================================
  Model:              Q4_K_M
  Server:             http://localhost:8080
  Trials:             1
  Prompts:            31
  Successful:         31
  Failed:             0
========================================================================
```
