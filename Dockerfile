# GB10 (Grace Blackwell, SM121) benchmark environment.
#
# The base image must be aarch64 -- GB10's CPU is ARM v9.2, so x86_64 CUDA images will
# not run here. CUDA 13.0 matches the driver shipped on DGX Spark class systems.
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    # SM121 is the consumer/edge Blackwell variant; build CUDA extensions for it
    # explicitly, since many projects' default arch lists stop at SM100.
    TORCH_CUDA_ARCH_LIST="12.1" \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# uv manages both the Python toolchain and the dependency set. A uv-managed CPython is
# required rather than the distro one: Triton JIT-compiles a C shim at runtime and
# needs the Python development headers, which the distro `python3` package omits.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Core deps plus the pure-Triton linear-attention kernels (Mamba-2 SSD, KDA).
RUN uv sync --extra linear

# mamba-ssm and causal-conv1d have no aarch64 + CUDA 13 wheels, so they are compiled
# here against the local toolkit. This is the slow layer; keep it last so edits to
# src/ do not trigger a rebuild.
RUN uv pip install --no-build-isolation causal-conv1d mamba-ssm

COPY scripts ./scripts
RUN chmod +x scripts/*.sh

CMD ["/bin/bash"]
