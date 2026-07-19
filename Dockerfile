# Dockerfile — Production image for llama.cpp with KleidiAI on ARM64.
#
# Build:
#   docker build -t llama-kleidai .
#
# Run server:
#   docker run -p 8080:8080 -v ./models:/models llama-kleidai \
#     --host 0.0.0.0 --port 8080 -m /models/model.gguf
#
# Run CLI:
#   docker run -it -v ./models:/models llama-kleidai \
#     -m /models/model.gguf -p "Hello"
#
# Multi-stage build: builder stage compiles, final stage copies only binaries.
# Target: ARM64 only (use --platform linux/arm64 or build on native ARM64).

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ──────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=2
ARG GGML_KLEIDIAI=ON

# Install build dependencies in a single layer for cache efficiency.
# Combined into one RUN to minimize layers and image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
        python3 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp at a pinned commit for reproducible builds.
# Change LLAMA_CPP_COMMIT to update the version.
ARG LLAMA_CPP_COMMIT=b10068
RUN git clone --depth 1 --recursive https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && git checkout ${LLAMA_CPP_COMMIT} 2>/dev/null || true \
    && git submodule update --init --recursive

WORKDIR /opt/llama.cpp

# Configure with KleidiAI enabled.
# On non-ARM64 hosts, KleidiAI will be detected as unavailable at configure time
# and silently disabled — the build still succeeds, just without KleidiAI kernels.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_KLEIDIAI=${GGML_KLEIDIAI} \
        -DGGML_CPU_KLEIDIAI=${GGML_KLEIDIAI} \
    && cmake --build build --config Release -j${BUILD_JOBS}

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime (minimal image)
# ──────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Only runtime libraries needed — no compilers, no build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security.
RUN groupadd -r llama && useradd -r -g llama -m -s /bin/bash llama

# Copy only the shared libraries and binaries from the builder.
# This avoids carrying the entire build tree in the final image.
COPY --from=builder /opt/llama.cpp/build/bin/ /opt/llama.cpp/bin/
COPY --from=builder /opt/llama.cpp/build/ggml/src/libggml-base.so /opt/llama.cpp/bin/
COPY --from=builder /opt/llama.cpp/build/ggml/src/libggml.so /opt/llama.cpp/bin/
COPY --from=builder /opt/llama.cpp/build/ggml/src/libggml-cpu.so /opt/llama.cpp/bin/
COPY --from=builder /opt/llama.cpp/build/src/libllama.so /opt/llama.cpp/bin/
COPY --from=builder /opt/llama.cpp/build/common/libllama-common.so /opt/llama.cpp/bin/
COPY --from=builder /opt/llama.cpp/build/tools/mtmd/libmtmd.so /opt/llama.cpp/bin/

# Ensure shared libraries are findable.
ENV LD_LIBRARY_PATH=/opt/llama.cpp/bin

# Model directory — mount at runtime.
RUN mkdir -p /models && chown -R llama:llama /models

# Default to the server binary.
WORKDIR /opt/llama.cpp/bin

# Expose the default server port.
EXPOSE 8080

# Switch to non-root user.
USER llama

# Default entrypoint: llama-server.
# Override with `docker run ... llama-kleidai -m /models/x.gguf` for CLI.
ENTRYPOINT ["/opt/llama.cpp/bin/llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
