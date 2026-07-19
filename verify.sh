#!/usr/bin/env bash
#
# verify.sh — Comprehensive verification of llama.cpp + KleidiAI environment.
#
# Checks: build tools, ARM ISA support, KleidiAI activation, binaries,
# required files, and runtime readiness.
#
# Exit code: 0 if all checks pass, 1 if any fail.
#
# Usage:
#   chmod +x verify.sh
#   ./verify.sh
#   ./setup.sh && ./verify.sh   # run after setup

set -uo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
LLAMA_CPP_DIR="${HOME}/llama.cpp"
BUILD_DIR="${LLAMA_CPP_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# ──────────────────────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Disable colors when not a TTY
if [[ ! -t 1 ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' DIM='' NC=''
fi

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
pass() {
    printf "  ${GREEN}[PASS]${NC}  %-38s %s\n" "$1" "${2:-}"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    printf "  ${RED}[FAIL]${NC}  %-38s %s\n" "$1" "${2:-}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

warn() {
    printf "  ${YELLOW}[WARN]${NC}  %-38s %s\n" "$1" "${2:-}"
    WARN_COUNT=$((WARN_COUNT + 1))
}

section() {
    echo ""
    printf "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${NC}\n"
    printf "${BOLD}${CYAN}  %s${NC}\n" "$1"
    printf "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${NC}\n"
}

cmd_exists() {
    command -v "$1" &>/dev/null
}

# ──────────────────────────────────────────────────────────────────────────────
# Section 1: Build Tools
# ──────────────────────────────────────────────────────────────────────────────
section "Build Tools"

# GCC
if cmd_exists gcc; then
    GCC_VER=$(gcc --version 2>/dev/null | head -1 | grep -oP '[\d]+\.[\d]+[\.\d]*')
    GCC_MAJOR=$(echo "${GCC_VER}" | cut -d. -f1)
    if [[ "${GCC_MAJOR}" -ge 11 ]]; then
        pass "gcc installed" "(${GCC_VER})"
    else
        fail "gcc installed" "(v${GCC_VER}, need 11+)"
    fi
else
    fail "gcc installed" "(not found)"
fi

# G++
if cmd_exists g++; then
    GPP_VER=$(g++ --version 2>/dev/null | head -1 | grep -oP '[\d]+\.[\d]+[\.\d]*')
    GPP_MAJOR=$(echo "${GPP_VER}" | cut -d. -f1)
    if [[ "${GPP_MAJOR}" -ge 11 ]]; then
        pass "g++ installed" "(${GPP_VER})"
    else
        fail "g++ installed" "(v${GPP_VER}, need 11+)"
    fi
else
    fail "g++ installed" "(not found)"
fi

# CMake
if cmd_exists cmake; then
    CMAKE_VER=$(cmake --version 2>/dev/null | head -1 | grep -oP '[\d]+\.[\d]+[\.\d]*')
    CMAKE_MAJOR=$(echo "${CMAKE_VER}" | cut -d. -f1)
    CMAKE_MINOR=$(echo "${CMAKE_VER}" | cut -d. -f2)
    if [[ "${CMAKE_MAJOR}" -ge 3 && "${CMAKE_MINOR}" -ge 21 ]] || [[ "${CMAKE_MAJOR}" -ge 4 ]]; then
        pass "cmake installed" "(${CMAKE_VER})"
    else
        fail "cmake installed" "(v${CMAKE_VER}, need 3.21+)"
    fi
else
    fail "cmake installed" "(not found)"
fi

# Git
if cmd_exists git; then
    GIT_VER=$(git --version 2>/dev/null | grep -oP '[\d]+\.[\d]+[\.\d]*')
    pass "git installed" "(${GIT_VER})"
else
    fail "git installed" "(not found)"
fi

# Python3
if cmd_exists python3; then
    PY_VER=$(python3 --version 2>/dev/null | grep -oP '[\d]+\.[\d]+[\.\d]*')
    pass "python3 installed" "(${PY_VER})"
else
    fail "python3 installed" "(not found)"
fi

# Ninja
if cmd_exists ninja; then
    NJ_VER=$(ninja --version 2>/dev/null)
    pass "ninja-build installed" "(${NJ_VER})"
elif cmd_exists ninja-build; then
    pass "ninja-build installed" ""
else
    warn "ninja-build not found" "(optional, cmake falls back to make)"
fi

# Make (fallback build system)
if cmd_exists make; then
    MAKE_VER=$(make --version 2>/dev/null | head -1 | grep -oP '[\d]+\.[\d]+[\.\d]*')
    pass "make installed" "(${MAKE_VER})"
else
    warn "make not found" "(needed if ninja unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Section 2: ARM ISA Support
# ──────────────────────────────────────────────────────────────────────────────
section "ARM ISA Support"

CPUINFO=""
if [[ -f /proc/cpuinfo ]]; then
    CPUINFO=$(grep -oP '(?<=Features\s{1,4}).*' /proc/cpuinfo | head -1)
fi

if [[ -z "${CPUINFO}" ]]; then
    warn "Cannot read /proc/cpuinfo" "(not Linux or permission denied)"
else
    check_feature() {
        local name="$1"
        local flag="$2"
        if echo "${CPUINFO}" | grep -qw "${flag}"; then
            pass "${name} support"
        else
            fail "${name} support" "(not detected)"
        fi
    }

    check_feature "NEON"     "asimd"
    check_feature "DOTPROD"  "dotprod"
    check_feature "SVE"      "sve"
    check_feature "SVE2"     "sve2"
    check_feature "I8MM"     "i8mm"
    check_feature "BF16"     "bf16"
    check_feature "AES"      "aes"
    check_feature "CRC32"    "crc32"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Section 3: Architecture
# ──────────────────────────────────────────────────────────────────────────────
section "System"

ARCH=$(uname -m)
if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    pass "Architecture is ARM64" "(${ARCH})"
else
    fail "Architecture is ARM64" "(found: ${ARCH})"
fi

if [[ -f /etc/os-release ]]; then
    OS_NAME=$(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)
    pass "Operating system" "(${OS_NAME})"
else
    warn "Cannot detect OS" "(/etc/os-release not found)"
fi

KERNEL=$(uname -r)
pass "Kernel version" "(${KERNEL})"

CORES=$(nproc 2>/dev/null || echo "N/A")
pass "CPU cores" "(${CORES})"

MEM=$(free -h 2>/dev/null | awk '/^Mem:/{print $2}')
pass "Total memory" "(${MEM})"

# ──────────────────────────────────────────────────────────────────────────────
# Section 4: llama.cpp Source
# ──────────────────────────────────────────────────────────────────────────────
section "llama.cpp Source"

if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    pass "llama.cpp source exists" "(${LLAMA_CPP_DIR})"
    cd "${LLAMA_CPP_DIR}"
    COMMIT=$(git log --oneline -1 2>/dev/null)
    pass "Git commit" "(${COMMIT})"
    SUBMODULE_COUNT=$(git submodule status 2>/dev/null | wc -l)
    if [[ "${SUBMODULE_COUNT}" -gt 0 ]]; then
        pass "Submodules initialized" "(${SUBMODULE_COUNT} submodules)"
    else
        warn "No submodules found" "(may not have been cloned with --recursive)"
    fi
else
    fail "llama.cpp source exists" "(not found at ${LLAMA_CPP_DIR})"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Section 5: Build Status
# ──────────────────────────────────────────────────────────────────────────────
section "Build Verification"

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    pass "CMakeCache exists"

    # Check KleidiAI
    KLEIDIAI_CPU=$(grep "GGML_CPU_KLEIDIAI:BOOL=" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    if [[ "${KLEIDIAI_CPU}" == "ON" ]]; then
        pass "GGML_CPU_KLEIDIAI=ON"
    else
        fail "GGML_CPU_KLEIDIAI" "(found: ${KLEIDIAI_CPU:-not set})"
    fi

    KLEIDIAI_TOP=$(grep "GGML_KLEIDIAI:" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | head -1 | cut -d= -f2)
    if [[ "${KLEIDIAI_TOP}" == "ON" ]]; then
        pass "GGML_KLEIDIAI=ON"
    else
        warn "GGML_KLEIDIAI" "(found: ${KLEIDIAI_TOP:-not set})"
    fi

    BUILD_TYPE=$(grep "CMAKE_BUILD_TYPE:STRING=" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    if [[ "${BUILD_TYPE}" == "Release" ]]; then
        pass "Build type" "(Release)"
    else
        warn "Build type" "(found: ${BUILD_TYPE:-unknown}, expected Release)"
    fi

    # CPU flags
    CPU_FLAGS=$(grep "GGML_CPU_AARCH64_FLAGS:STRING=" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    if [[ -n "${CPU_FLAGS}" ]]; then
        pass "CPU compiler flags" "(${CPU_FLAGS:0:60})"
    else
        warn "CPU compiler flags" "(none set)"
    fi
else
    fail "CMakeCache exists" "(build directory not found)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Section 6: Binaries
# ──────────────────────────────────────────────────────────────────────────────
section "Binaries"

BINARIES=(
    "llama-cli:Main CLI inference"
    "llama-server:HTTP API server"
    "llama-quantize:Model quantization"
    "llama-bench:Benchmarking tool"
    "llama-embedding:Embedding extraction"
    "llama-simple:Simple example"
)

for entry in "${BINARIES[@]}"; do
    BIN_NAME="${entry%%:*}"
    BIN_DESC="${entry##*:}"
    BIN_PATH="${BIN_DIR}/${BIN_NAME}"

    if [[ -x "${BIN_PATH}" ]]; then
        pass "${BIN_NAME}" "(${BIN_DESC})"
    elif [[ -f "${BIN_PATH}" ]]; then
        fail "${BIN_NAME}" "(exists but not executable)"
    else
        fail "${BIN_NAME}" "(not found)"
    fi
done

# Check shared libraries
SHARED_LIBS=(
    "libllama.so"
    "libggml.so"
    "libggml-base.so"
    "libggml-cpu.so"
)

echo ""
for lib in "${SHARED_LIBS[@]}"; do
    FOUND=false
    for dir in "${BIN_DIR}" "${BUILD_DIR}/src" "${BUILD_DIR}/ggml/src"; do
        if [[ -f "${dir}/${lib}" ]]; then
            pass "Shared library ${lib}"
            FOUND=true
            break
        fi
    done
    if [[ "${FOUND}" == false ]]; then
        fail "Shared library ${lib}" "(not found)"
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# Section 7: Runtime Smoke Test
# ──────────────────────────────────────────────────────────────────────────────
section "Runtime Smoke Test"

if [[ -x "${BIN_DIR}/llama-cli" ]]; then
    VERSION_OUTPUT=$("${BIN_DIR}/llama-cli" --version 2>&1 || true)
    if [[ -n "${VERSION_OUTPUT}" ]]; then
        pass "llama-cli --version" "(${VERSION_OUTPUT:0:40})"
    else
        warn "llama-cli --version" "(no output)"
    fi
fi

if [[ -x "${BIN_DIR}/llama-server" ]]; then
    SERVER_HELP=$("${BIN_DIR}/llama-server" --help 2>&1 | head -1 || true)
    if [[ -n "${SERVER_HELP}" ]]; then
        pass "llama-server --help" "(responds correctly)"
    else
        warn "llama-server --help" "(no output)"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Section 8: Environment Scripts
# ──────────────────────────────────────────────────────────────────────────────
section "Repository Scripts"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in setup.sh verify.sh probe.py; do
    SCRIPT_PATH="${SCRIPT_DIR}/${script}"
    if [[ -f "${SCRIPT_PATH}" ]]; then
        if [[ -x "${SCRIPT_PATH}" ]] || [[ "${script}" == *.py ]]; then
            pass "${script}" "(present)"
        else
            warn "${script}" "(present but not executable — run: chmod +x ${script})"
        fi
    else
        fail "${script}" "(not found)"
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# Section 9: Disk Space
# ──────────────────────────────────────────────────────────────────────────────
section "Disk Space"

AVAIL_KB=$(df -k "${HOME}" 2>/dev/null | awk 'NR==2{print $4}')
if [[ -n "${AVAIL_KB}" ]]; then
    AVAIL_GB=$((AVAIL_KB / 1048576))
    if [[ "${AVAIL_GB}" -ge 5 ]]; then
        pass "Free disk space" "(${AVAIL_GB} GB available)"
    elif [[ "${AVAIL_GB}" -ge 2 ]]; then
        warn "Free disk space" "(${AVAIL_GB} GB — tight, models need 1-4 GB each)"
    else
        fail "Free disk space" "(${AVAIL_GB} GB — need 5+ GB free)"
    fi
fi

SWAP_KB=$(free -k 2>/dev/null | awk '/^Swap:/{print $2}')
if [[ -n "${SWAP_KB}" && "${SWAP_KB}" -gt 0 ]]; then
    SWAP_MB=$((SWAP_KB / 1024))
    pass "Swap available" "(${SWAP_MB} MB)"
else
    warn "No swap configured" "(builds may OOM on low-RAM instances)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
TOTAL=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))
printf "${BOLD}══════════════════════════════════════════════════════════════${NC}\n"
printf "${BOLD}  RESULTS: ${GREEN}${PASS_COUNT} passed${NC}, ${RED}${FAIL_COUNT} failed${NC}, ${YELLOW}${WARN_COUNT} warnings${NC} ${BOLD}(of ${TOTAL} checks)${NC}\n"
printf "${BOLD}══════════════════════════════════════════════════════════════${NC}\n"

if [[ ${FAIL_COUNT} -eq 0 ]]; then
    echo ""
    printf "  ${GREEN}${BOLD}>>> ALL CRITICAL CHECKS PASSED <<<${NC}\n"
    echo ""
    exit 0
else
    echo ""
    printf "  ${RED}${BOLD}>>> ${FAIL_COUNT} CHECK(S) FAILED — see above for details <<<${NC}\n"
    echo ""
    exit 1
fi
