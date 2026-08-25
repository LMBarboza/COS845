# Attention vs. SSM Benchmark Suite: Architectural Guidelines & Manual

This repository contains a two-stage empirical benchmark suite comparing traditional
attention mechanisms, optimized attention architectures, state-space models (SSMs),
and hybrid linear attention models.

The goal of this project is to test whether state-space models maintain their
theoretical memory and latency advantages over optimized attention mechanisms when
executed on modern hardware and deployed via production inference engines.

---

## Target Hardware & System Architecture

- **Primary Benchmark Hardware:** NVIDIA GB10 Grace Blackwell Superchip (e.g. DGX
  Spark or equivalent GB10-based system)
- **GPU compute capability:** SM121 (12.1) — this is the "consumer/edge" Blackwell
  variant, distinct from datacenter Blackwell (SM100, used in B100/B200/GB200). It
  lacks TMEM and the tcgen05 tensor-core instruction; its tensor cores use an
  extended `mma.sync` path closer to Ampere. **Kernels that hard-require SM100
  (e.g. FlashMLA's default build, some FlashAttention-3/4 Blackwell paths) may not
  run here — see "Known Kernel Compatibility Caveats" below before assuming a
  library "just works" because it lists Blackwell support.**
- **CPU:** ARM v9.2 (aarch64), not x86_64. Prebuilt pip wheels for some CUDA
  extension packages may not exist for this platform/architecture combination and
  may need to be built from source.
- **Memory:** 128 GB unified LPDDR5X, shared between CPU and GPU, ~273 GB/s
  raw bandwidth (NVIDIA-published figure; there is no separate discrete VRAM pool
  to size against). This is roughly an order of magnitude lower bandwidth than a
  datacenter GPU (H100/B200 use HBM3/HBM3e in the multiple-TB/s range), which is
  directly relevant to this study's thesis about memory-bandwidth-bound decode —
  don't describe this system as "high-bandwidth memory" in write-ups.
- **Target Precision:** `bfloat16` (BF16) as the primary, reliable dtype for all
  mechanisms. `float8` (FP8) may be attempted per-kernel, but FP8 block-scaled GEMM
  paths written for SM100 have been reported to fail on SM12x hardware — treat FP8
  results as best-effort/optional, not a required data point, and fall back to BF16
  if a kernel errors or silently mis-executes under FP8.
- **Primary Runtime Constraints:** CUDA 12.x+ / Triton. Note that Triton currently
  treats SM12x as SM80 (Ampere) for scheduling purposes, which disables
  Blackwell-specific Triton optimizations but should not block correctness for the
  Mamba-2/KDA selective-scan kernels, which are Ampere-compatible by design.

### Known Kernel Compatibility Caveats (check before benchmarking)

- **FlashAttention (naive install):** Standard `flash-attn` pip releases have
  historically only listed Ampere/Ada/Hopper support. If it fails to import or
  raises an architecture error on this hardware, do not treat that as a benchmark
  result — either build a compatible version, fall back to PyTorch's
  `scaled_dot_product_attention` (which can use the cuDNN SDPA backend on SM12x),
  or mark the mechanism as "unsupported on this hardware" for that hardware target.
- **DeepSeek MLA / FlashMLA:** FlashMLA's shipped kernels are built around SM90
  (WGMMA) and SM100 (tcgen05) backends specifically. Neither instruction path
  exists on SM12x. If a reference/eager MLA implementation isn't available, treat
  MLA as a mechanism that may only be benchmarkable on non-GB10 hardware, and say
  so explicitly in results rather than silently substituting a different kernel.
- **vLLM on SM12x:** may require the `--enforce-eager` flag if CUDA-graph capture
  paths assume datacenter Blackwell features. Check the installed vLLM version's
  release notes for current SM12x/GB10 status before assuming default flags work.
- Confirm actual installed kernel behavior (e.g. via a short correctness check
  against a reference PyTorch implementation) before trusting any mechanism's
  latency numbers — a kernel that silently falls back to a slow or incorrect path
  will otherwise look like a legitimate benchmark result.

---

## Benchmark Methodology & Experimental Design

The study is structured as a two-stage benchmark:

```
                               STAGING WORKFLOW
                                       |
            +--------------------------+--------------------------+
            v                                                     v
   STAGE 1: Microbenchmarks                             STAGE 2: Production Inference
  (Component Isolation)                                   (vLLM Model Serving)
            |                                                     |
 +----------+----------+----------+----------+----------+  +------+------+------+
 v          v          v          v          v          v  v      v      v
Naive   FlashAttn   DeepSeek   Mamba-2    Kimi KDA  Transformer Mamba-2  Kimi Linear
 MHA      (v2/v3)     MLA       (SSD)                (GQA/MLA)   Model     Hybrid
```

### Stage 1: Component-Level Synthetic Microbenchmarks

Isolates kernel execution overhead across synthetic input sequence lengths
S = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}.

Tested Mechanisms:

1. **Naive MHA:** Standard PyTorch O(S^2) scaled dot-product attention (baseline
   control).
2. **FlashAttention:** IO-aware tiled attention kernel (FlashAttention-2, and
   FlashAttention-3/4 only if a build with confirmed SM12x support is available —
   see compatibility caveats above).
3. **DeepSeek MLA (Multi-Head Latent Attention):** Low-rank latent-space KV-cache
   compression. If no SM12x-compatible MLA kernel is available, benchmark against a
   reference (non-fused) implementation and label it as such, rather than skipping
   the mechanism silently.
4. **Mamba-2:** State-Space Dual (SSD) selective-scan state compression.
5. **Kimi Delta Attention (KDA):** Hardware-efficient gated linear attention/
   state-space memory mechanism (from Moonshot AI's Kimi Linear architecture).

### Stage 2: Real-World Inference Engine Evaluation

Evaluates complete language models running on vLLM to capture batching,
scheduling, state management, and KV-cache overheads.

Tested Model Types:

1. **Modern Transformer:** LLM using Grouped-Query Attention (GQA) or Multi-Head
   Latent Attention (MLA).
2. **Mamba-2:** Pure SSM model (e.g. `state-spaces/mamba2-2.7b`).
3. **Kimi Linear:** Hybrid model combining KDA layers with periodic global MLA
   layers. Confirm the vLLM version installed actually supports this architecture
   before scripting the serving benchmark — hybrid-model support in vLLM has
   landed incrementally, model-by-model.

---

## Standard Metrics Captured

All benchmark outputs must record and save the following quantitative metrics into
structured JSON and CSV formats (`results/raw/`):

### Stage 1 Microbenchmark Metrics

| Metric Name              | Unit         | Definition & Measurement Method                                                                                     |
| ------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Prefill Latency           | ms           | Elapsed time to process the full prompt sequence S. Measured via `torch.cuda.Event` timing.                            |
| Decode Latency            | ms/token     | Time to process step S+1 given existing state/context of size S. Averaged across 128 generated tokens.                 |
| Peak Memory Allocated     | MB / GB      | Peak active memory allocated by tensors, via `torch.cuda.max_memory_allocated()`. Note: on GB10 this draws from the shared 128 GB unified pool, not a dedicated VRAM budget — record total system memory pressure too if relevant. |
| Peak Memory Reserved      | MB / GB      | Total memory reserved by the PyTorch allocator, via `torch.cuda.max_memory_reserved()`.                                |
| Effective Throughput      | tokens/sec   | Sequence Length / Latency in seconds.                                                                                    |
| Memory Bandwidth          | GB/s         | Estimated empirical throughput: (Bytes Read + Bytes Written) / Execution Time (seconds). Compare against the ~273 GB/s hardware ceiling, not an HBM-class ceiling. |

### Stage 2 Production Serving Metrics (vLLM)

| Metric Name                    | Unit         | Definition & Measurement Method                                                                                  |
| -------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| TTFT (Time To First Token)       | ms           | Duration from HTTP request dispatch until the first generated stream token arrives.                                  |
| TPOT (Time Per Output Token)     | ms/token     | (Total Time - TTFT) / (Output Tokens - 1).                                                                            |
| Inter-Token Latency (ITL)        | ms           | Per-token decode interval delta; track mean, standard deviation, and p95 tail latency.                               |
| Aggregate Output Throughput      | tokens/sec   | Total generated tokens across all concurrent client streams per unit time.                                           |
| Request Throughput               | req/sec      | Total completed inference requests per second under concurrency load.                                                |
| KV/State Memory Consumption      | GB           | Memory footprint occupied exclusively by the KV-cache (Transformers) or state structures (Mamba/KDA).                |

---

## Recommended Repository Structure

```
attention-vs-ssm-benchmark/
├── CLAUDE.md                   # This instruction file
├── pyproject.toml              # uv workspace dependencies & tools
├── Dockerfile                  # GB10 / CUDA 12.x+ environment container (aarch64 base image)
├── docker-compose.yml          # Container orchestration setup
├── src/
│   ├── __init__.py
│   ├── stage1_micro/           # Component-level synthetic tests
│   │   ├── __init__.py
│   │   ├── naive_mha.py
│   │   ├── flash_attn.py
│   │   ├── mla_attn.py
│   │   ├── mamba2_ssm.py
│   │   ├── kda_attn.py
│   │   └── runner.py
│   ├── stage2_serving/         # Production vLLM client & server scripts
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── benchmark_vllm.py
│   └── utils/                  # Metrics, CUDA timing, and plotting
│       ├── __init__.py
│       ├── cuda_timer.py
│       ├── metrics.py
│       └── plotting.py
├── scripts/
│   ├── run_stage1.sh           # Stage 1 sweep runner
│   └── run_stage2.sh           # Stage 2 sweep runner
└── results/
    ├── raw/                    # Raw JSON data outputs
    └── plots/                  # Rendered charts and figures
```

---

## Project Setup & Environment Management

Package and environment management is powered by `uv`.

### Local Setup (Host System)

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync virtual environment and dependencies
uv sync --all-extras

# If a CUDA extension package (flash-attn, mamba-ssm, causal-conv1d) fails to
# find a prebuilt wheel for this platform, it likely needs a source build against
# the local CUDA toolkit and aarch64 CPU — check the package's build docs before
# assuming the dependency is simply broken or unavailable.

# Validate formatting and codebase linting
uv run ruff check .
uv run ruff format --check .
```

### Docker Setup (NVIDIA Container Toolkit)

```bash
# Build docker container image (use an aarch64-compatible CUDA base image)
docker build -t attention-ssm-gb10:latest .

# Run interactive session with GPU exposed
docker run --gpus all --ipc=host -it --rm \
  -v $(pwd)/results:/app/results \
  attention-ssm-gb10:latest /bin/bash
```

---

## Execution Commands

### Running Stage 1 Microbenchmarks

```bash
# Full sequence length sweep (Prefill & Decode)
uv run python -m src.stage1_micro.runner \
  --device cuda \
  --dtype bfloat16 \
  --min-seq 512 \
  --max-seq 65536 \
  --output-dir results/raw/stage1/

# Run a single component test (e.g., DeepSeek MLA)
uv run python -m src.stage1_micro.runner --component mla --phase decode
```

### Running Stage 2 Production Serving Benchmarks

```bash
# Step 1: Launch vLLM server instance in background/terminal 1
uv run vllm serve state-spaces/mamba2-2.7b \
  --trust-remote-code \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536

# Step 2: Execute benchmark client sweep in terminal 2
uv run python -m src.stage2_serving.benchmark_vllm \
  --endpoint http://localhost:8000/v1 \
  --concurrency 1,2,4,8,16,32,64 \
  --input-tokens 4096 \
  --output-tokens 128 \
  --output-dir results/raw/stage2/
```

Note: `--max-model-len 65536` combined with `--gpu-memory-utilization 0.90` on a
128 GB unified-memory system leaves headroom for the OS and CPU-side processes —
don't assume the full 128 GB is available to the GPU allocator the way a discrete
GPU's dedicated VRAM would be.

---

## Scientific & Code Implementation Rules

When writing or modifying code in this repository, follow these rules:

1. **Strict CUDA Synchronization:** Never measure execution time using pure Python
   `time.time()`. Always use `torch.cuda.Event(enable_timing=True)`, or wrap
   execution steps in `torch.cuda.synchronize()` calls before starting and
   stopping timers.

2. **Memory Reset Protocols:** Before running any iteration benchmark:

   ```python
   torch.cuda.empty_cache()
   torch.cuda.reset_peak_memory_stats()
   ```

3. **Warmup Requirements:** Every microbenchmark step must execute at least 10
   warmup iterations (to trigger JIT compilation, Triton kernel instantiation, and
   CUDA context allocation) prior to collecting statistics across a minimum of 50
   measurement runs.

4. **Synthetic Tensor Consistency:** Unless otherwise specified, Stage 1 synthetic
   microbenchmarks should standardize on:

   - Batch Size = 1 (for sequence-scaling studies)
   - Hidden Size = 2048
   - Num Heads = 32
   - Head Dimension = 64 (32 x 64 = 2048, consistent with Hidden Size)
   - Data Type = `torch.bfloat16`

5. **Correctness Before Speed:** Before recording any latency/throughput number
   for a new kernel or mechanism, verify its output against a reference PyTorch
   implementation on a small sequence length. A kernel that silently runs an
   incorrect or degraded fallback path (see hardware caveats above) will otherwise
   produce a fast but meaningless number.

6. **Code Quality Standards:**
   - Type hints are required for all public function definitions.
   - Code style must adhere to standard `ruff` rules. Run `uv run ruff format .`
     prior to committing.
