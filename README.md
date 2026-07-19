# LLM Optimization — KleidiAI-Powered llama.cpp on ARM64

High-performance LLM inference on ARM64 hardware using [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Arm KleidiAI](https://github.com/ARM-software/kleidiai) optimized kernels.

KleidiAI provides hardware-accelerated int8 matrix multiply (i8mm), SVE/SVE2 vector operations, and dot product instructions for maximum throughput on Arm Neoverse V2 (Graviton4) and compatible processors.

---

## Requirements

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Architecture | ARM64 (aarch64) | ARM64 (aarch64) |
| CPU | ARMv8.2+ with NEON | ARMv9.0+ with SVE2 + i8mm |
| RAM | 2 GB | 4 GB+ |
| Disk | 5 GB free | 10 GB+ free |
| Network | Internet access for setup | — |

### Tested Hardware

| Instance / Platform | CPU | SVE | I8MM | KleidiAI | Status |
|---------------------|-----|-----|------|----------|--------|
| AWS EC2 c8g.large | Graviton4 (Neoverse V2) | Yes | Yes | Full | Verified |
| AWS EC2 c8g.xlarge | Graviton4 (Neoverse V2) | Yes | Yes | Full | Verified |
| AWS EC2 c7g.large | Graviton3 (Neoverse V1) | Yes | No | Partial | Verified |
| AWS EC2 t4g.small | Graviton2 (Neoverse N1) | No | No | Disabled | Verified |

### Software

- Ubuntu 22.04 LTS or newer (24.04, 26.04)
- GCC 11+
- CMake 3.21+
- Git
- Internet connection

---

## Quick Start

One command to set up everything:

```bash
git clone https://github.com/Adityasharma0810/llm_optimization_team_hackathon.git
cd llm_optimization_team_hackathon
chmod +x setup.sh
./setup.sh
```

The script will:
1. Detect your Ubuntu version and architecture
2. Install all required build tools
3. Clone llama.cpp with KleidiAI submodules
4. Build with `-DGGML_KLEIDIAI=ON -DGGML_CPU_KLEIDIAI=ON`
5. Verify the build succeeded

### Options

```bash
./setup.sh -j4              # Use 4 parallel build jobs
./setup.sh --clean          # Wipe existing build and rebuild
./setup.sh --skip-update    # Skip apt update (faster on repeat runs)
```

---

## Repository Structure

```
llm_optimization_team_hackathon/
├── setup.sh              # One-command environment setup
├── verify.sh             # Dependency and build verification
├── probe.py              # Hardware capability detector
├── Dockerfile            # Multi-stage production Docker build
├── .dockerignore         # Docker build context exclusions
├── README.md             # This file
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI/CD pipeline
└── docs/
    └── TROUBLESHOOTING.md # Common issues and solutions
```

---

## Usage

### Download a Model

llama.cpp requires GGUF-format models. Download one from [Hugging Face](https://huggingface.co/models?search=gguf):

```bash
# Example: small Q4_K_M quantized model
mkdir -p ~/models
cd ~/models

# Using huggingface-cli (install with: pip install huggingface-hub)
pip install huggingface-hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir .

# Or using wget
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -O ~/models/model.gguf
```

### Run llama-cli (Interactive)

```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/model.gguf \
  -p "What is the capital of France?" \
  -n 256
```

### Run llama-server (API)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/model.gguf \
  --host 0.0.0.0 \
  --port 8080
```

Test the API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

### Run via Docker

```bash
docker build -t llama-kleidai .
docker run -p 8080:8080 -v ~/models:/models llama-kleidai \
  -m /models/model.gguf --host 0.0.0.0 --port 8080
```

---

## Verification

After setup, run these to confirm everything works:

```bash
# Check all dependencies and build status
./verify.sh

# Full hardware capability report
python3 probe.py

# JSON output (for CI/CD)
python3 probe.py --json

# Single-line summary
python3 probe.py --compact
```

### Expected Output (probe.py)

```
──────────────────────────────────────────────────────────
  Hardware
──────────────────────────────────────────────────────────
  CPU Model                     Neoverse-V2 (Graviton4)  YES
  Implementer                   ARM  YES
  Architecture                  8  YES
  Cores                         2
  Memory Total                  3914476 kB

──────────────────────────────────────────────────────────
  ARM ISA Extensions
──────────────────────────────────────────────────────────
  NEON                          supported  YES
  DOTPROD                       supported  YES
  SVE                           supported  YES
  SVE2                          supported  YES
  I8MM                          supported  YES

──────────────────────────────────────────────────────────
  Build Tools
──────────────────────────────────────────────────────────
  GCC                           15.2.0
  CMake                         4.2.3
  Ninja                         1.13.2

──────────────────────────────────────────────────────────
  llama.cpp Build
──────────────────────────────────────────────────────────
  Source                        YES
  KleidiAI (CPU backend)        ON
  Build Type                    Release

──────────────────────────────────────────────────────────
  Readiness Assessment
──────────────────────────────────────────────────────────
  ARM64 architecture            PASS
  SVE/SVE2 available            PASS
  I8MM available                PASS
  KleidiAI enabled in build     PASS
  llama.cpp source present      PASS

  >>> SYSTEM IS FULLY READY FOR llama.cpp WITH KleidiAI <<<
```

### Expected Output (verify.sh)

```
═══════════════════════════════════════════════════════════
  Dependency Checks
═══════════════════════════════════════════════════════════
  [PASS]  gcc installed        (15.2.0)
  [PASS]  g++ installed        (15.2.0)
  [PASS]  cmake installed      (4.2.3)
  [PASS]  git installed        (2.53.0)
  [PASS]  python3 installed    (3.14.3)
  [PASS]  ninja-build installed (1.13.2)

═══════════════════════════════════════════════════════════
  Build Verification
═══════════════════════════════════════════════════════════
  [PASS]  llama-cli binary
  [PASS]  llama-server binary
  [PASS]  llama-quantize binary
  [PASS]  GGML_CPU_KLEIDIAI=ON

═══════════════════════════════════════════════════════════
  ARM ISA Support
═══════════════════════════════════════════════════════════
  [PASS]  NEON support
  [PASS]  DOTPROD support
  [PASS]  SVE support
  [PASS]  I8MM support

RESULT: 10/10 checks passed
```

---

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions. Common issues:

| Problem | Solution |
|---------|----------|
| `GGML_CPU_KLEIDIAI:BOOL=OFF` | Use a Graviton4+ instance (c8g/m8g/r8g). Graviton2/3 lack i8mm. |
| Build killed during compilation | Add swap space: `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| `Permission denied` on .pem | Run: `icacls.exe .\key.pem /inheritance:r /grant:r "${env:USERNAME}:(R)"` (Windows) |
| SSH connection refused | Check EC2 security group has inbound SSH (port 22). |
| cmake version too old | Use Ubuntu 22.04+ or install cmake via snap: `sudo snap install cmake` |
| `git submodule` errors | Run: `cd ~/llama.cpp && git submodule update --init --recursive` |

---

## Project Links

- **Repository**: https://github.com/Adityasharma0810/llm_optimization_team_hackathon
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **KleidiAI**: https://github.com/ARM-software/kleidiai
- **AWS Graviton**: https://github.com/aws/aws-graviton-getting-started

---

## License

This project is provided as-is for hackathon purposes. llama.cpp is MIT-licensed. KleidiAI is Apache-2.0 licensed.
