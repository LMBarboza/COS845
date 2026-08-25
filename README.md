# Attention vs. SSM Benchmark Suite (NVIDIA GB10 / SM121)

A two-stage empirical benchmark comparing traditional attention, optimized attention,
state-space models, and hybrid linear attention, to test whether SSMs keep their
theoretical memory and latency advantages over optimized attention on modern hardware
and under a production inference engine.

See [`CLAUDE.md`](CLAUDE.md) for the full architectural guidelines, hardware notes, and
kernel-compatibility caveats this suite is built against.

## Layout

```
src/stage1_micro/    component-level synthetic microbenchmarks + runner
src/stage2_serving/  vLLM streaming client + concurrency sweep
src/utils/           CUDA-event timing, metric records, plotting
scripts/             sweep runners
results/raw/         JSON + CSV outputs
results/plots/       rendered figures
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra linear          # core deps + Triton kernels (Mamba-2 SSD, KDA)
uv run ruff check . && uv run ruff format --check .
```

`mamba-ssm` and `causal-conv1d` have no aarch64 + CUDA 13 wheels and must be built
from source against the local toolkit:

```bash
TORCH_CUDA_ARCH_LIST="12.1" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
  uv pip install --no-build-isolation causal-conv1d mamba-ssm
```

**The virtualenv must use a uv-managed CPython, not the distro `python3`.** Triton
JIT-compiles a small C shim at every kernel launch and needs `Python.h`; without the
development headers it silently falls back to CPU with a "Triton is not supported on
current platform" warning, and every Triton-backed mechanism would be measured on the
wrong device. `uv venv --python 3.12 --python-preference only-managed` gets this right.

Docker (aarch64 CUDA base, builds the CUDA extensions in-image):

```bash
docker compose build && docker compose run --rm benchmark
```

## Running

```bash
./scripts/run_stage1.sh                                    # full sweep + figures
uv run python -m src.stage1_micro.runner --component mla --phase decode

./scripts/run_stage2.sh state-spaces/mamba2-2.7b           # server + client sweep
```

## What is actually measured

**Stage 1** isolates five mechanisms over S = 512 … 65536 at the standard shape
(batch 1, hidden 2048, 32 heads x 64 head dim, bfloat16), with >= 10 warmup iterations
and >= 50 timed iterations, `torch.cuda.Event` timing throughout, and the allocator
caches and peak-memory counters cleared before every cell. Decode is measured as 128
consecutive single-token steps over a growing context, so the reported figure is
ms/token by construction.

Every mechanism is checked against a reference implementation before it is timed
(repository rule 5). A mechanism that fails its check is recorded with
`status="correctness_failed"` and **no** latency number — a kernel that quietly took a
degraded path must not produce a fast, meaningless result. The pass criterion is
relative error (`max|got-want| / max|want|`), because the SSD and delta-rule scans
produce outputs two orders of magnitude larger than an attention output and the same
bfloat16 rounding shows up as a very different absolute number.

Which kernel produced each number is recorded in every row:

| Mechanism | Kernel used on SM121 | Note |
|---|---|---|
| Naive MHA | eager `matmul` + `softmax` | the materialized-score control |
| FlashAttention | PyTorch SDPA, `FLASH_ATTENTION` backend | `flash-attn` has no aarch64/CUDA-13 wheel; the SDPA `MATH` backend is never selected, since that is the naive path under a different label |
| DeepSeek MLA | reference PyTorch, matrix-absorbed decode | FlashMLA needs SM90 WGMMA or SM100 tcgen05; neither exists on SM121 |
| Mamba-2 (SSD) | `mamba-ssm` Triton kernels, built for SM121 | verified against a sequential fp32 SSD recurrence |
| Kimi Delta Attention | `flash-linear-attention` Triton kernels | verified against `naive_recurrent_kda` |

**Stage 2** drives a live vLLM server over its OpenAI-compatible API, sweeping client
concurrency and recording TTFT, TPOT, per-token ITL (mean/std/p95), aggregate output
throughput, request throughput, and the KV/state cache footprint scraped from
`/metrics`. Timing there is client-side wall clock, which is the thing a served client
actually experiences; the CUDA-event rule governs Stage 1 kernel measurement.

## Results on this machine

NVIDIA GB10, SM12.1, 128 GB unified LPDDR5X, driver 580.173.02, CUDA 13.0,
torch 2.13.0+cu130, Triton 3.7.1, aarch64. All five mechanisms passed correctness
(relative error 4.4e-3 … 7.4e-3, consistent with bfloat16 rounding).

Decode latency, ms/token:

| S | Naive MHA | FlashAttention | MLA (ref) | Mamba-2 | KDA |
|---|---|---|---|---|---|
| 512 | 0.051 | 0.040 | 0.122 | **0.040** | 0.056 |
| 4096 | 0.154 | 0.134 | 0.120 | **0.040** | 0.054 |
| 65536 | 2.897 | 2.168 | 0.767 | **0.039** | 0.053 |

Decode state footprint, MB:

| S | Naive MHA / Flash (KV cache) | MLA latent cache | Mamba-2 | KDA |
|---|---|---|---|---|
| 512 | 5.1 | 0.7 | 1.0 | 0.5 |
| 65536 | 513.1 | 68.1 | **1.0** | **0.5** |

What the data says:

- **The SSM decode advantage is real and it is asymptotic, not constant.** Mamba-2 and
  KDA are flat to within noise across a 128x span of context (0.040 → 0.039 and 0.056 →
  0.053 ms/token), because their state is fixed-size. But at S = 512 FlashAttention
  matches Mamba-2 exactly (0.040 ms/token) and beats KDA. The crossover on this
  hardware is around S = 2048–4096; below it, an optimized attention kernel gives up
  nothing.
- **Decode is bandwidth-bound, as the thesis predicts.** Attention decode runs at
  180–250 GB/s against a ~273 GB/s unified-memory ceiling — it is reading the KV cache
  and little else, and its latency tracks cache size linearly. The recurrent
  mechanisms move ~1 MB per step regardless of S and never approach the ceiling; they
  are launch-latency-bound instead, which is why their curves are flat rather than
  merely shallow.
- **MLA sits between the two, by design.** Matrix-absorbed decode reads a 68 MB latent
  cache at S = 65536 where MHA reads 513 MB, and lands at 0.77 ms/token vs 2.17. Below
  S = 4096 its fixed projection cost dominates and it is the *slowest* mechanism
  measured — and this is a reference implementation, so its constant factor is an upper
  bound, not a property of MLA.
- **Prefill inverts the ranking.** FlashAttention prefills S = 65536 in 199 ms with
  1.06 GB peak; Mamba-2 does it in 17.7 ms with 793 MB. The materialized-score paths
  (naive MHA, reference MLA) cannot reach S = 32768 at all: their score matrix would
  need 64 GiB, and they are recorded as `oom` by an explicit guard rather than being
  allowed to invite the OOM killer on a shared 128 GB unified pool.

Reading the bandwidth figures: `bandwidth_gb_per_s` is an *analytic* estimate — the
bytes a mechanism must move, divided by measured time — so a value above the ~273 GB/s
ceiling (attention decode at S = 2048 reports ~400 GB/s) means the working set was
small enough to be served from cache, not that DRAM was exceeded.

Sequence lengths, mechanisms, and phases are chosen at the command line; the raw JSON
carries full environment provenance (GPU, driver, arch list, torch/Triton versions) so
each result file is self-describing.

## Stage 2 results on this machine

vLLM 0.27.1 (aarch64 wheel, `torch==2.13.0` pin matching the Stage 1 environment), seven
models, prompts of 2,048 and 4,096 tokens, 128 output tokens with `ignore_eos`,
concurrency 1→64 at four requests per client, `--gpu-memory-utilization 0.60` held
constant, one server per model. 98 concurrency levels, zero failed requests.

Run the whole matrix with `./scripts/run_stage2_sweeps.sh b a`.

**Prefix caching must be disabled** (`--no-enable-prefix-caching`, which the sweep driver
passes). The client sends one identical prompt to every request in a level, so with
caching on a transformer serves all but the first from a cached prefix and reports a
near-zero TTFT, while the SSMs — which have no prefix cache — pay full prefill every
time. Measured on Qwen2.5-3B at 2,048 tokens: 42 ms TTFT with caching on, 172 ms with it
off.

Aggregate output throughput and peak cache at 64 concurrent clients:

| Model | Class | 2,048 prompt | 4,096 prompt |
|---|---|---|---|
| Qwen2.5-3B-Instruct | GQA transformer | 379.0 tok/s · 4.28 GB | 155.6 tok/s · 8.27 GB |
| Mamba-2 2.7B | pure SSM | 376.7 tok/s · 5.13 GB | 245.6 tok/s · 5.13 GB |
| DeepSeek-V2-Lite | MLA transformer, MoE | 305.9 tok/s · 3.61 GB | 196.1 tok/s · 6.98 GB |
| Zamba2-2.7B-Instruct | hybrid | 156.6 tok/s · 27.03 GB | — (4,096-token window) |
| Qwen2.5-7B-Instruct | GQA transformer | 201.3 tok/s · 6.66 GB | 87.8 tok/s · 12.87 GB |
| Mamba-Codestral-7B | pure SSM | 162.6 tok/s · 8.27 GB | 100.8 tok/s · 8.27 GB |
| Falcon-H1-7B-Instruct | hybrid | 174.5 tok/s · 12.66 GB | 90.5 tok/s · 16.88 GB |

The crossover sits between 2,048 and 4,096 tokens at both scales. At 2,048 the
transformers win or tie and the SSMs are the *more* memory-hungry option; at 4,096 the
transformers' throughput turns over with added concurrency while the SSMs' keeps rising.
SSM state memory is length-independent by construction — vLLM allocates a fixed-size
recurrent state block per sequence — and measures as such: Mamba-2 differs by 0.03%
between the two prompt lengths, Mamba-Codestral by 0.05%.

Hybrid is not by itself a memory win. Zamba2's shared attention layers use 32 KV heads
(full MHA, no grouping) and it holds 6.3× the cache of a same-size GQA transformer.

Which kernel served each model on SM121:

| Model | vLLM backend | Note |
|---|---|---|
| Qwen2.5-3B / 7B | `FLASH_ATTN` (FlashAttention 2) | CUDA-graph capture works, no `--enforce-eager` needed |
| DeepSeek-V2-Lite | `TRITON_MLA` decode, `FLASH_ATTN` MLA prefill | FlashMLA's SM90/SM100 kernels are unavailable here, as expected on SM12x |
| Mamba-2 / Mamba-Codestral | Mamba2 SSD Triton kernels | same kernel family Stage 1 measured |
| Zamba2 / Falcon-H1 | Mamba2 SSD + the model's own attention backend | |

### Checkpoint and support caveats

- **Llama-3.1-8B is gated**; with no HF token available, Qwen2.5-7B-Instruct stands in as
  the 7B-scale GQA transformer.
- **Bamba-9B is not servable on vLLM 0.27.1** — `BambaForCausalLM` was removed in 0.23.0
  and is listed in `_PREVIOUSLY_SUPPORTED_MODELS`. Falcon-H1-7B-Instruct stands in as the
  7B-scale Mamba-2 + attention hybrid.
- **`state-spaces/mamba2-2.7b` ships raw `mamba_ssm` format** (`d_model`/`ssm_cfg`, no
  `architectures` key, no tokenizer) and vLLM cannot load it. The same weights in
  HF format (`AntonV/mamba2-2.7b-hf`) load once `architectures: ["Mamba2ForCausalLM"]`
  is added to the config.
- **`mistralai/Mamba-Codestral-7B-v0.1` fails to load as published**: the repo ships
  `consolidated.safetensors` (Mistral's own format, `model.*` names) alongside the HF
  shards (`backbone.*` names), vLLM globs both, and weight loading dies with
  `There is no module or parameter named 'model' in Mamba2ForCausalLM`. Serving from a
  directory that excludes the consolidated file works.
- **CUDA graphs made no measurable difference** where it was checked: DeepSeek-V2-Lite at
  2,048 tokens and 64 clients scored 305.9 tok/s with graphs and 318.6 eager. Both runs
  are kept (`deepseek-v2-lite__in2048.csv` and `..._in2048_eager.csv`).

### What Stage 2 does not measure

No quality or long-context-recall evaluation, two prompt lengths rather than a sweep (so
the crossover is bracketed, not located), and models that differ in more than their
sequence mixer — these are architecture families as shipped, not a controlled ablation.
