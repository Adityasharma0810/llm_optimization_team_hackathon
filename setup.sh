#!/usr/bin/env bash
#
# setup.sh — One-command environment setup for llama.cpp with KleidiAI on ARM64 Ubuntu.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # default: 2 build jobs
#   ./setup.sh -j4          # override parallel jobs
#   ./setup.sh --clean      # wipe existing build and rebuild from scratch
#   ./setup.sh --skip-update  # skip apt update/upgrade (faster on repeated runs)
#
# Idempotent: safe to run multiple times. Skips steps already completed.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_DIR="${HOME}/llama.cpp"
BUILD_DIR="${LLAMA_CPP_DIR}/build"
MIN_CMAKE_VERSION="3.21"
MIN_GCC_VERSION="11"
REQUIRED_PACKAGES=(
    build-essential
    cmake
    git
    python3
    python3-pip
    ninja-build
    curl
    wget
)

# ──────────────────────────────────────────────────────────────────────────────
# Colors and output helpers
# ──────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { printf "${BLUE}[INFO]${NC}    %s\n" "$*"; }
success() { printf "${GREEN}[PASS]${NC}    %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARN]${NC}    %s\n" "$*"; }
error()   { printf "${RED}[FAIL]${NC}    %s\n" "$*" >&2; }
step()    { printf "\n${CYAN}${BOLD}═══ %s ═══${NC}\n" "$*"; }

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
BUILD_JOBS=2
CLEAN_BUILD=false
SKIP_UPDATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j)
            BUILD_JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --skip-update)
            SKIP_UPDATE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-j N] [--clean] [--skip-update]"
            echo ""
            echo "Options:"
            echo "  -j N             Number of parallel build jobs (default: 2)"
            echo "  --clean          Remove existing build directory and rebuild"
            echo "  --skip-update    Skip apt update/upgrade"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────
step "Pre-flight checks"

# Must be Ubuntu
if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
    error "This script requires Ubuntu. Detected: $(cat /etc/os-release | head -1)"
    exit 1
fi

UBUNTU_VERSION=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
UBUNTU_CODENAME=$(grep VERSION_CODENAME /etc/os-release | cut -d'=' -f2)
info "Ubuntu ${UBUNTU_VERSION} (${UBUNTU_CODENAME})"

# Must be ARM64
ARCH=$(uname -m)
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
    error "This script requires an ARM64 (aarch64) system. Detected: ${ARCH}"
    exit 1
fi
success "Architecture: ${ARCH}"

# Check for required CPU features
if [[ -f /proc/cpuinfo ]]; then
    CPU_FEATURES=$(grep -oP '(?<=Features\s{1,4}).*' /proc/cpuinfo | head -1)
    for feat in i8mm sve dotprod; do
        if echo "${CPU_FEATURES}" | grep -qw "${feat}"; then
            success "CPU feature: ${feat}"
        else
            warn "CPU feature: ${feat} — not detected. KleidiAI may not fully activate."
        fi
    done
fi

# ──────────────────────────────────────────────────────────────────────────────
# System update
# ──────────────────────────────────────────────────────────────────────────────
if [[ "${SKIP_UPDATE}" == false ]]; then
    step "System update"
    info "Running apt update..."
    sudo apt-get update -qq
    info "Running apt upgrade..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq
    success "System packages updated"
else
    step "System update"
    info "Skipped (--skip-update flag)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Install dependencies
# ──────────────────────────────────────────────────────────────────────────────
step "Installing dependencies"

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if dpkg -s "${pkg}" &>/dev/null; then
        success "${pkg} (already installed)"
    else
        MISSING_PACKAGES+=("${pkg}")
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    info "Installing: ${MISSING_PACKAGES[*]}"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING_PACKAGES[@]}"
    success "Missing packages installed"
else
    success "All required packages already installed"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Version checks
# ──────────────────────────────────────────────────────────────────────────────
step "Tool version verification"

# GCC version
GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
if [[ "${GCC_VERSION}" -ge "${MIN_GCC_VERSION}" ]]; then
    success "gcc version: $(gcc --version | head -1)"
else
    error "gcc ${MIN_GCC_VERSION}+ required, found ${GCC_VERSION}"
    exit 1
fi

# G++ version
success "g++ version: $(g++ --version | head -1)"

# CMake version
CMAKE_VERSION_RAW=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
CMAKE_MAJOR=$(echo "${CMAKE_VERSION_RAW}" | cut -d. -f1)
CMAKE_MINOR=$(echo "${CMAKE_VERSION_RAW}" | cut -d. -f2)
if [[ "${CMAKE_MAJOR}" -gt "${MIN_CMAKE_VERSION}" ]] || \
   { [[ "${CMAKE_MAJOR}" -eq "${MIN_CMAKE_VERSION}" ]] && [[ "${CMAKE_MINOR}" -ge 0 ]]; }; then
    success "cmake version: $(cmake --version | head -1)"
else
    error "cmake ${MIN_CMAKE_VERSION}+ required, found ${CMAKE_VERSION_RAW}"
    exit 1
fi

# Git
success "git version: $(git --version)"

# Python3
success "python3 version: $(python3 --version)"

# Ninja
success "ninja version: $(ninja --version 2>/dev/null || echo 'not found')"

# ──────────────────────────────────────────────────────────────────────────────
# Clone or update llama.cpp
# ──────────────────────────────────────────────────────────────────────────────
step "llama.cpp source"

if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    info "Existing llama.cpp found at ${LLAMA_CPP_DIR}"
    cd "${LLAMA_CPP_DIR}"
    info "Pulling latest changes..."
    git pull --ff-only 2>/dev/null || warn "Could not fast-forward. Using existing version."
    info "Updating submodules..."
    git submodule update --init --recursive 2>/dev/null || warn "Submodule update had issues."
    success "llama.cpp source is up to date"
else
    info "Cloning llama.cpp..."
    git clone --recursive "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
    success "llama.cpp cloned to ${LLAMA_CPP_DIR}"
fi

cd "${LLAMA_CPP_DIR}"
LLAMA_VERSION=$(git log --oneline -1)
info "llama.cpp version: ${LLAMA_VERSION}"

# ──────────────────────────────────────────────────────────────────────────────
# Clean build (optional)
# ──────────────────────────────────────────────────────────────────────────────
if [[ "${CLEAN_BUILD}" == true ]]; then
    step "Clean build"
    if [[ -d "${BUILD_DIR}" ]]; then
        info "Removing existing build directory..."
        rm -rf "${BUILD_DIR}"
        success "Build directory cleaned"
    else
        info "No existing build directory to clean"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# CMake configure
# ──────────────────────────────────────────────────────────────────────────────
step "CMake configure"

CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_KLEIDIAI=ON
    -DGGML_CPU_KLEIDIAI=ON
)

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    info "Configuring with KleidiAI enabled..."
    cmake -B "${BUILD_DIR}" "${CMAKE_FLAGS[@]}" 2>&1 | tee /tmp/llama-cmake.log
    success "CMake configuration complete"
else
    info "Existing CMake cache found. Reconfiguring..."
    cmake -B "${BUILD_DIR}" "${CMAKE_FLAGS[@]}" 2>&1 | tee /tmp/llama-cmake.log
    success "CMake reconfiguration complete"
fi

# Verify KleidiAI is enabled in cache
if grep -q "GGML_CPU_KLEIDIAI:BOOL=ON" "${BUILD_DIR}/CMakeCache.txt"; then
    success "GGML_CPU_KLEIDIAI is ON"
else
    error "GGML_CPU_KLEIDIAI is OFF — KleidiAI did not activate."
    error "This CPU may not support i8mm/SVE. Check with probe.py."
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────────
step "Building llama.cpp (${BUILD_JOBS} jobs)"

cmake --build "${BUILD_DIR}" --config Release -j"${BUILD_JOBS}" 2>&1 | tee /tmp/llama-build.log
BUILD_EXIT=$?

if [[ ${BUILD_EXIT} -ne 0 ]]; then
    error "Build failed. Full log: /tmp/llama-build.log"
    exit 1
fi

success "Build completed successfully"

# ──────────────────────────────────────────────────────────────────────────────
# Post-build verification
# ──────────────────────────────────────────────────────────────────────────────
step "Post-build verification"

BINARIES=("llama-cli" "llama-server" "llama-quantize" "llama-bench")

for bin in "${BINARIES[@]}"; do
    if [[ -x "${BUILD_DIR}/bin/${bin}" ]]; then
        success "bin/${bin} — executable"
    else
        warn "bin/${bin} — not found"
    fi
done

# Quick smoke test
if [[ -x "${BUILD_DIR}/bin/llama-cli" ]]; then
    LLAMA_VERSION_OUTPUT=$("${BUILD_DIR}/bin/llama-cli" --version 2>&1 || true)
    if [[ -n "${LLAMA_VERSION_OUTPUT}" ]]; then
        success "llama-cli --version: ${LLAMA_VERSION_OUTPUT}"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
printf "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}\n"
printf "${GREEN}${BOLD}║              SETUP COMPLETE — ALL CHECKS PASSED             ║${NC}\n"
printf "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}\n"
echo ""
info "llama.cpp path:   ${LLAMA_CPP_DIR}"
info "Build path:       ${BUILD_DIR}"
info "Binaries:         ${BUILD_DIR}/bin/"
info "CMake flags:      ${CMAKE_FLAGS[*]}"
echo ""
info "Next steps:"
info "  1. Run ./verify.sh to check all dependencies"
info "  2. Run python3 probe.py to see hardware capabilities"
info "  3. Download a GGUF model and run:"
info "     ${BUILD_DIR}/bin/llama-cli -m <model.gguf>"
info "  4. Or start the server:"
info "     ${BUILD_DIR}/bin/llama-server -m <model.gguf> --host 0.0.0.0 --port 8080"
echo ""
