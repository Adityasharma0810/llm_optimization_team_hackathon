# Troubleshooting Guide

## Table of Contents

1. [Build Failures](#build-failures)
2. [KleidiAI Not Activating](#kleidai-not-activating)
3. [Memory Issues](#memory-issues)
4. [SSH and Connectivity](#ssh-and-connectivity)
5. [Docker Issues](#docker-issues)
6. [Runtime Errors](#runtime-errors)

---

## Build Failures

### Out of Memory (OOM) During Build

**Symptom:**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Cause:** The compiler process was killed by the Linux OOM killer. t4g.small has 2GB RAM; parallel compilation of large C++ files can exceed this.

**Fix:**
```bash
# Option 1: Use fewer parallel jobs
./setup.sh -j1

# Option 2: Add swap space (recommended)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap persistent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### CMake Version Too Old

**Symptom:**
```
CMake Error: CMake was unable to find a build program corresponding to "Ninja"
```
or
```
CMake Error at CMakeLists.txt: Minimum version X.Y required
```

**Fix:**
```bash
# Ubuntu 22.04 ships cmake 3.22 which is sufficient.
# If you have an older version:
sudo snap install cmake --classic
# or
sudo apt install cmake
```

### Git Submodule Errors

**Symptom:**
```
fatal: couldn't find remote ref v1.24.0
```

**Fix:**
```bash
cd ~/llama.cpp
git submodule update --init --recursive
# If that fails, re-clone:
cd ~
rm -rf llama.cpp
git clone --recursive https://github.com/ggerganov/llama.cpp.git
```

### OpenSSL Not Found (Warning, Not Fatal)

**Symptom:**
```
CMake Warning: Could NOT find OpenSSL, HTTPS support disabled
```

**Impact:** HTTPS support in cpp-httplib is disabled. Does not affect core llama.cpp functionality.

**Fix (optional):**
```bash
sudo apt install libssl-dev
# Then re-run cmake configuration
cd ~/llama.cpp && cmake -B build -DGGML_KLEIDIAI=ON -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
```

---

## KleidiAI Not Activating

### GGML_CPU_KLEIDIAI Stays OFF

**Symptom:** `grep GGML_CPU_KLEIDIAI ~/llama.cpp/build/CMakeCache.txt` shows `OFF`.

**Causes (check in order):**

1. **Wrong instance type.** KleidiAI requires i8mm (ARMv8.6+ / ARMv9.0+). Verify:
   ```bash
   grep -o 'i8mm' /proc/cpuinfo
   ```
   If empty, your CPU does not support i8mm. Use Graviton4 (c8g/m8g/r8g).

2. **Wrong cmake flag.** You must pass both flags:
   ```bash
   cmake -B build -DGGML_KLEIDIAI=ON -DGGML_CPU_KLEIDIAI=ON
   ```
   `GGML_KLEIDIAI` alone is not enough; `GGML_CPU_KLEIDIAI` must also be `ON`.

3. **Stale build cache.** Clear and reconfigure:
   ```bash
   cd ~/llama.cpp
   rm -rf build
   cmake -B build -DGGML_KLEIDIAI=ON -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
   ```

### CPU Feature Detection Table

| CPU | Architecture | NEON | DOTPROD | SVE | SVE2 | I8MM | KleidiAI |
|-----|-------------|------|---------|-----|------|------|----------|
| Neoverse N1 (Graviton2/t4g) | ARMv8.2 | Yes | Yes | No | No | No | No |
| Neoverse V1 (Graviton3/c7g) | ARMv8.4 | Yes | Yes | Yes | No | No | Partial |
| Neoverse V2 (Graviton4/c8g) | ARMv9.0 | Yes | Yes | Yes | Yes | Yes | Full |
| Neoverse V3 (Graviton5/m9g) | ARMv9.2 | Yes | Yes | Yes | Yes | Yes | Full |

---

## Memory Issues

### Instance Running Out of Memory at Runtime

**Symptom:** `llama-server` or `llama-cli` crashes with OOM when loading large models.

**Fix:**
- Use a larger instance (more RAM).
- Use a smaller quantized model (Q4_K_M instead of F16).
- Check available memory: `free -h`

### Swap File Creation

```bash
# Create 4GB swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify
swapon --show

# Make persistent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## SSH and Connectivity

### Connection Refused

**Checklist:**
1. EC2 instance state is `Running`
2. Security group has inbound rule: Type `SSH`, Port `22`, Source `0.0.0.0/0`
3. Public IP is correct (changes on stop/start)
4. Key pair matches the `.pem` file

### Permission Denied (Publickey)

**Windows:**
```powershell
icacls.exe ".\key.pem" /inheritance:r /grant:r "${env:USERNAME}:(R)"
```

**Linux/macOS:**
```bash
chmod 400 ./key.pem
```

### Host Key Verification Failed

```bash
# Remove old host key
ssh-keygen -R <IP_ADDRESS>
# Or connect with:
ssh -o StrictHostKeyChecking=no -i key.pem ubuntu@<IP>
```

---

## Docker Issues

### Cannot Build for ARM64 on x86

```bash
# Use buildx with QEMU emulation
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx create --use --name arm64-builder --platform linux/arm64
docker buildx build --platform linux/arm64 -t llama-kleidai .
```

### Container Cannot Find Shared Libraries

```bash
# Verify LD_LIBRARY_PATH is set
docker run --rm llama-kleidai env | grep LD_LIBRARY
# Should show: LD_LIBRARY_PATH=/opt/llama.cpp/bin
```

---

## Runtime Errors

### "model not found" Error

```bash
# Verify model file exists and is readable
ls -la ~/models/model.gguf

# Verify you're passing the correct path
~/llama.cpp/build/bin/llama-cli -m ~/models/model.gguf --help
```

### Slow Inference

- Ensure you're running on a Graviton4+ instance with KleidiAI enabled.
- Check KleidiAI status: `grep GGML_CPU_KLEIDIAI ~/llama.cpp/build/CMakeCache.txt`
- Use optimized quantizations: Q4_K_M, Q5_K_M, Q8_0.
- Ensure adequate memory — no swap thrashing: `vmstat 1`

### Server Not Accessible from Outside

```bash
# Check server is listening
ss -tlnp | grep 8080

# Check EC2 security group allows inbound TCP 8080
# (AWS Console → EC2 → Security Groups → Inbound Rules)
```
