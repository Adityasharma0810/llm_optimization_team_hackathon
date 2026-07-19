#!/usr/bin/env python3
"""
probe.py — Hardware and software capability probe for llama.cpp + KleidiAI.

Detects CPU model, architecture, ARM ISA extensions, memory, build tool
versions, and KleidiAI status. Designed to run on any ARM64 Ubuntu system.

Usage:
    python3 probe.py              # full report
    python3 probe.py --json       # JSON output for CI/CD
    python3 probe.py --compact    # single-line summary
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Color helpers (auto-disabled when not a TTY)
# ──────────────────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()


class C:
    """ANSI color codes."""
    RESET   = "\033[0m"    if USE_COLOR else ""
    BOLD    = "\033[1m"     if USE_COLOR else ""
    GREEN   = "\033[0;32m"  if USE_COLOR else ""
    YELLOW  = "\033[1;33m"  if USE_COLOR else ""
    RED     = "\033[0;31m"  if USE_COLOR else ""
    CYAN    = "\033[0;36m"  if USE_COLOR else ""
    DIM     = "\033[2m"     if USE_COLOR else ""


def _status(label: str, value: str, ok: bool = True) -> None:
    """Print a single probe result line."""
    tag = f"{C.GREEN}YES{C.RESET}" if ok else f"{C.RED}NO{C.RESET}"
    print(f"  {C.CYAN}{label:<28}{C.RESET} {value}  {tag if ok is not None else ''}")


def _section(title: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}{'─' * 50}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─' * 50}{C.RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Run helpers
# ──────────────────────────────────────────────────────────────────────────────
def run(cmd: str, fallback: str = "N/A") -> str:
    """Run a shell command, return stdout or fallback on failure."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        return output if output else fallback
    except (subprocess.TimeoutExpired, Exception):
        return fallback


def run_rc(cmd: str) -> int:
    """Run a shell command, return exit code."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.returncode
    except (subprocess.TimeoutExpired, Exception):
        return -1


# ──────────────────────────────────────────────────────────────────────────────
# CPU information
# ──────────────────────────────────────────────────────────────────────────────
def get_cpuinfo() -> dict[str, Any]:
    """Parse /proc/cpuinfo for ARM CPU details."""
    info: dict[str, Any] = {
        "model_name": "N/A",
        "implementer": "N/A",
        "architecture": "N/A",
        "variant": "N/A",
        "part": "N/A",
        "revision": "N/A",
        "features": [],
        "bogomips": "N/A",
        "cores": 0,
    }

    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return info

    text = cpuinfo_path.read_text()
    blocks = text.strip().split("\n\n")

    info["cores"] = len(blocks)

    if blocks:
        first = blocks[0]
        for line in first.split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()

            if key == "model name":
                info["model_name"] = val
            elif key == "CPU implementer":
                info["implementer"] = val
            elif key == "CPU architecture":
                info["architecture"] = val
            elif key == "CPU variant":
                info["variant"] = val
            elif key == "CPU part":
                info["part"] = val
            elif key == "CPU revision":
                info["revision"] = val
            elif key == "BogoMIPS":
                info["bogomips"] = val
            elif key == "Features":
                info["features"] = val.split()

    # Map CPU part numbers to known ARM cores
    PART_MAP = {
        "0xd40": "Neoverse-N1 (Graviton2)",
        "0xd4f": "Neoverse-V2 (Graviton4)",
        "0xd4c": "Neoverse-V1 (Graviton3)",
        "0xd4a": "Neoverse-E1",
        "0xd0b": "Cortex-A78",
        "0xd0c": "Neoverse-N2",
        "0xd80": "Cortex-A520",
        "0xd81": "Cortex-A720",
        "0xd82": "Cortex-X4",
        "0xd4e": "Neoverse-V3 (Graviton5)",
    }

    implementer_map = {
        "0x41": "ARM",
        "0x48": "HiSilicon",
        "0x51": "Qualcomm",
        "0x53": "Samsung",
        "0xc0": "Ampere",
    }

    cpu_part = info["part"]
    cpu_impl = info["implementer"]

    if cpu_part in PART_MAP:
        info["model_name"] = PART_MAP[cpu_part]
    if cpu_impl in implementer_map:
        info["implementer"] = implementer_map[cpu_impl]

    return info


# ──────────────────────────────────────────────────────────────────────────────
# ARM ISA feature detection
# ──────────────────────────────────────────────────────────────────────────────
ARM_FEATURES = {
    "NEON":     "asimd",
    "DOTPROD":  "dotprod",
    "SVE":      "sve",
    "SVE2":     "sve2",
    "I8MM":     "i8mm",
    "BF16":     "bf16",
    "FMA":      "fphp",  # fp16 half-precision
    "AES":      "aes",
    "SHA1":     "sha1",
    "SHA2":     "sha2",
    "SHA3":     "sha3",
    "SM3":      "sm3",
    "SM4":      "sm4",
    "SME":      "sme",
    "FCMA":     "fcma",
    "RNG":      "rng",
    "CRC32":    "crc32",
    "LSE":      "atomics",
}


def detect_arm_features(cpu_features: list[str]) -> dict[str, bool]:
    """Map CPU feature flags to named ISA extensions."""
    results = {}
    for name, flag in ARM_FEATURES.items():
        results[name] = flag in cpu_features
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Memory detection
# ──────────────────────────────────────────────────────────────────────────────
def get_memory() -> dict[str, str]:
    """Read memory info from /proc/meminfo."""
    mem: dict[str, str] = {"total": "N/A", "available": "N/A", "free": "N/A"}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem["total"] = line.split(":")[1].strip()
                elif line.startswith("MemAvailable:"):
                    mem["available"] = line.split(":")[1].strip()
                elif line.startswith("MemFree:"):
                    mem["free"] = line.split(":")[1].strip()
    except OSError:
        pass
    return mem


# ──────────────────────────────────────────────────────────────────────────────
# Tool version detection
# ──────────────────────────────────────────────────────────────────────────────
def get_tool_version(command: str) -> str:
    """Extract the first version-like string from a tool's output."""
    output = run(command, fallback="")
    match = re.search(r"[\d]+\.[\d]+[\.\d]*", output)
    return match.group(0) if match else output[:60] if output else "not found"


# ──────────────────────────────────────────────────────────────────────────────
# Build detection
# ──────────────────────────────────────────────────────────────────────────────
def get_llama_build_info() -> dict[str, str]:
    """Detect llama.cpp build info from the CMake cache."""
    llama_dir = Path.home() / "llama.cpp"
    cache_file = llama_dir / "build" / "CMakeCache.txt"

    build: dict[str, str] = {
        "source_exists": "N/A",
        "kleidai_cpu": "N/A",
        "kleidai_top": "N/A",
        "build_type": "N/A",
        "llama_version": "N/A",
        "cmake_cpu_flags": "N/A",
    }

    if not cache_file.exists():
        build["source_exists"] = "NO (build directory not found)"
        return build

    build["source_exists"] = "YES"

    text = cache_file.read_text()

    for key, field in [
        ("GGML_CPU_KLEIDIAI", "kleidai_cpu"),
        ("GGML_KLEIDIAI", "kleidai_top"),
        ("CMAKE_BUILD_TYPE", "build_type"),
    ]:
        match = re.search(rf"^{key}:(\w+)=(.+)$", text, re.MULTILINE)
        if match:
            build[field] = match.group(2).strip()

    # CPU flags
    match = re.search(r"GGML_CPU_AARCH64_FLAGS:STRING=(.+)", text)
    if match:
        build["cmake_cpu_flags"] = match.group(1).strip()

    # Get llama version from binary
    llama_cli = llama_dir / "build" / "bin" / "llama-cli"
    if llama_cli.exists() and os.access(str(llama_cli), os.X_OK):
        ver = run(f"{llama_cli} --version 2>&1", fallback="")
        if ver:
            build["llama_version"] = ver.split("\n")[0].strip()

    # Git commit
    git_dir = llama_dir / ".git"
    if git_dir.exists():
        commit = run(f"git -C {llama_dir} log --oneline -1", fallback="N/A")
        build["llama_version"] = commit

    return build


# ──────────────────────────────────────────────────────────────────────────────
# OS detection
# ──────────────────────────────────────────────────────────────────────────────
def get_os_info() -> dict[str, str]:
    """Gather OS and kernel information."""
    info: dict[str, str] = {
        "name": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    # Ubuntu-specific
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    info["pretty"] = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("VERSION_ID="):
                    info["version_id"] = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass

    return info


# ──────────────────────────────────────────────────────────────────────────────
# Main probe
# ──────────────────────────────────────────────────────────────────────────────
def probe_all() -> dict[str, Any]:
    """Run all probes and return structured results."""
    cpu = get_cpuinfo()
    features = detect_arm_features(cpu["features"])
    mem = get_memory()
    os_info = get_os_info()
    build_info = get_llama_build_info()

    gcc_ver = get_tool_version("gcc --version")
    cmake_ver = get_tool_version("cmake --version")
    ninja_ver = get_tool_version("ninja --version")
    python_ver = get_tool_version("python3 --version")
    git_ver = get_tool_version("git --version")

    return {
        "hardware": {
            "cpu_model": cpu["model_name"],
            "cpu_implementer": cpu["implementer"],
            "cpu_part": cpu["part"],
            "architecture": cpu["architecture"],
            "cores": cpu["cores"],
            "bogomips": cpu["bogomips"],
            "memory": mem,
        },
        "arm_isa": features,
        "os": {
            **os_info,
            "gcc_version": gcc_ver,
            "cmake_version": cmake_ver,
            "ninja_version": ninja_ver,
            "python_version": python_ver,
            "git_version": git_ver,
        },
        "build": build_info,
    }


def print_report(data: dict[str, Any]) -> None:
    """Pretty-print the probe report."""
    hw = data["hardware"]
    isa = data["arm_isa"]
    os_info = data["os"]
    build = data["build"]

    print(f"\n{C.BOLD}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}║           SYSTEM CAPABILITY PROBE — llama.cpp               ║{C.RESET}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════════════════╝{C.RESET}")

    _section("Hardware")
    _status("CPU Model", hw["cpu_model"])
    _status("Implementer", hw["cpu_implementer"])
    _status("CPU Part", hw["part"] if hw["part"] != "N/A" else "N/A", ok=None)
    _status("Architecture", hw["architecture"])
    _status("Cores", str(hw["cores"]))
    _status("BogoMIPS", hw["bogomips"])
    _status("Memory Total", hw["memory"]["total"])
    _status("Memory Available", hw["memory"]["available"])

    _section("ARM ISA Extensions")
    for name, supported in isa.items():
        _status(name, "supported" if supported else "NOT supported", ok=supported)

    _section("Operating System")
    _status("OS", os_info.get("pretty", os_info["name"]))
    _status("Kernel", os_info["release"])
    _status("Architecture", os_info["machine"])

    _section("Build Tools")
    _status("GCC", os_info["gcc_version"])
    _status("G++", get_tool_version("g++ --version"))
    _status("CMake", os_info["cmake_version"])
    _status("Ninja", os_info["ninja_version"])
    _status("Python3", os_info["python_version"])
    _status("Git", os_info["git_version"])

    _section("llama.cpp Build")
    _status("Source", build["source_exists"])
    _status("Version", build["llama_version"])
    _status("Build Type", build["build_type"])
    _status("KleidiAI (top-level)", build["kleidai_top"])
    _status("KleidiAI (CPU backend)", build["kleidai_cpu"])
    _status("CPU Flags", build["cmake_cpu_flags"][:80] if build["cmake_cpu_flags"] != "N/A" else "N/A", ok=None)

    # Readiness assessment
    _section("Readiness Assessment")

    kleidai_ok = build["kleidai_cpu"] == "ON"
    i8mm_ok = isa.get("I8MM", False)
    sve_ok = isa.get("SVE", False) or isa.get("SVE2", False)
    source_ok = build["source_exists"] == "YES"

    checks = [
        ("ARM64 architecture", os_info["machine"] in ("aarch64", "arm64")),
        ("SVE/SVE2 available", sve_ok),
        ("I8MM available", i8mm_ok),
        ("KleidiAI enabled in build", kleidai_ok),
        ("llama.cpp source present", source_ok),
    ]

    all_pass = True
    for label, ok in checks:
        _status(label, "PASS" if ok else "FAIL", ok=ok)
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"  {C.GREEN}{C.BOLD}>>> SYSTEM IS FULLY READY FOR llama.cpp WITH KleidiAI <<<  {C.RESET}")
    else:
        print(f"  {C.YELLOW}{C.BOLD}>>> PARTIAL READINESS — review warnings above <<<{C.RESET}")
    print()


def print_compact(data: dict[str, Any]) -> None:
    """Single-line summary for scripts."""
    hw = data["hardware"]
    build = data["build"]
    isa = data["arm_isa"]

    features = []
    for f in ["NEON", "DOTPROD", "SVE", "SVE2", "I8MM"]:
        if isa.get(f, False):
            features.append(f)

    print(
        f"{hw['cpu_model']} | {hw['cores']} cores | "
        f"{hw['memory']['total']} RAM | "
        f"ISA: {','.join(features) or 'none'} | "
        f"KleidiAI: {build['kleidai_cpu']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe system capabilities for llama.cpp + KleidiAI."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (for CI/CD pipelines)"
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Print a single-line summary"
    )
    args = parser.parse_args()

    data = probe_all()

    if args.json:
        print(json.dumps(data, indent=2))
    elif args.compact:
        print_compact(data)
    else:
        print_report(data)


if __name__ == "__main__":
    main()
