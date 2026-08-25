"""Stage 1 sweep driver: correctness gate, then timed prefill/decode measurement.

Every mechanism is measured under identical conditions -- same synthetic shapes, same
warmup/iteration counts, allocator caches and peak-memory counters cleared before each
measurement, CUDA-event timing throughout.

Correctness runs first (repository rule 5). A mechanism whose output does not match
its reference is recorded with ``status="correctness_failed"`` and is not timed: a
kernel that silently took a degraded path must not produce a latency number.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import torch

from ..utils.cuda_timer import reset_memory_stats, time_callable
from ..utils.metrics import (
    BenchConfig,
    BenchmarkRun,
    MemoryStats,
    PhaseResult,
    bandwidth_gb_per_s,
    collect_environment,
    effective_throughput,
)
from .base import Mechanism, PreparedPhase
from .flash_attn import FlashAttention
from .kda_attn import KimiDeltaAttention
from .mamba2_ssm import MambaTwoSSD
from .mla_attn import DeepSeekMLA
from .naive_mha import NaiveMHA

MECHANISMS: dict[str, type[Mechanism]] = {
    NaiveMHA.key: NaiveMHA,
    FlashAttention.key: FlashAttention,
    DeepSeekMLA.key: DeepSeekMLA,
    MambaTwoSSD.key: MambaTwoSSD,
    KimiDeltaAttention.key: KimiDeltaAttention,
}

PHASES = ("prefill", "decode")
DEFAULT_SEQ_LENS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
#: Sequence length used for the pre-timing correctness check.
VERIFY_SEQ_LEN = 256


def sequence_lengths(min_seq: int, max_seq: int) -> list[int]:
    """Powers of two between ``min_seq`` and ``max_seq``, inclusive."""
    lengths = [s for s in DEFAULT_SEQ_LENS if min_seq <= s <= max_seq]
    if not lengths:
        raise ValueError(f"no benchmark sequence length in [{min_seq}, {max_seq}]")
    return lengths


def _is_oom(exc: BaseException) -> bool:
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError | MemoryError)
        or "out of memory" in str(exc).lower()
    )


def measure_phase(
    mech: Mechanism,
    phase: str,
    seq_len: int,
    verification: tuple[float, float, str] | None,
) -> PhaseResult:
    """Prepare, time, and score a single (mechanism, phase, seq_len) cell."""
    cfg = mech.config
    result = PhaseResult(
        mechanism=mech.name,
        implementation=mech.implementation,
        phase=phase,
        seq_len=seq_len,
        warmup=cfg.warmup,
    )
    if verification is not None:
        (
            result.correctness_max_abs_err,
            result.correctness_rel_err,
            result.correctness_reference,
        ) = verification

    reset_memory_stats()
    prepared: PreparedPhase | None = None
    try:
        prepared = (
            mech.prepare_prefill(seq_len) if phase == "prefill" else mech.prepare_decode(seq_len)
        )
        iters = prepared.iters or cfg.iters
        stats = time_callable(
            prepared.fn,
            warmup=cfg.warmup,
            iters=iters,
            reset_fn=prepared.reset_fn,
            pre_measure_fn=prepared.pre_measure_fn,
        )
    except Exception as exc:  # noqa: BLE001 -- an OOM or kernel failure is a real datum
        result.status = "oom" if _is_oom(exc) else "error"
        result.error = f"{type(exc).__name__}: {exc}"
        del prepared
        reset_memory_stats()
        return result

    memory = MemoryStats.capture()
    result.iters = stats.iters
    result.latency_ms = stats.mean_ms
    result.latency_median_ms = stats.median_ms
    result.latency_std_ms = stats.std_ms
    result.latency_p95_ms = stats.p95_ms
    result.latency_min_ms = stats.min_ms
    result.latency_max_ms = stats.max_ms
    result.peak_allocated_mb = memory.peak_allocated_mb
    result.peak_reserved_mb = memory.peak_reserved_mb
    result.state_memory_mb = prepared.state_memory_mb
    result.bytes_moved = prepared.bytes_moved
    result.bandwidth_gb_per_s = bandwidth_gb_per_s(prepared.bytes_moved, stats.mean_ms)
    result.notes = prepared.notes

    # Prefill throughput is the whole prompt over the elapsed time; decode is measured
    # one token per timed iteration, so its mean latency is already ms/token.
    result.throughput_tokens_per_s = effective_throughput(prepared.tokens_processed, stats.mean_ms)

    del prepared
    reset_memory_stats()
    return result


def run_mechanism(
    mech_cls: type[Mechanism],
    config: BenchConfig,
    seq_lens: Sequence[int],
    phases: Sequence[str],
    skip_verify: bool,
    verbose: bool = True,
) -> list[PhaseResult]:
    """Instantiate, verify, and sweep one mechanism across sequence lengths."""
    avail = mech_cls.availability()
    if not avail.available:
        if verbose:
            print(f"  SKIP {mech_cls.name}: {avail.reason}", flush=True)
        return [
            PhaseResult(
                mechanism=mech_cls.name,
                implementation=avail.implementation,
                phase=phase,
                seq_len=s,
                status="unsupported",
                error=avail.reason,
            )
            for phase in phases
            for s in seq_lens
        ]

    if verbose:
        note = f" ({avail.reason})" if avail.reason else ""
        print(f"\n== {mech_cls.name} [{avail.implementation}]{note}", flush=True)

    try:
        mech = mech_cls(config)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  FAILED to construct: {exc}", flush=True)
        return [
            PhaseResult(
                mechanism=mech_cls.name,
                implementation=avail.implementation,
                phase=phase,
                seq_len=s,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            for phase in phases
            for s in seq_lens
        ]

    verification: tuple[float, float, str] | None = None
    if not skip_verify:
        try:
            check = mech.verify(seq_len=VERIFY_SEQ_LEN)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  correctness check raised: {exc}", flush=True)
                traceback.print_exc()
            return [
                PhaseResult(
                    mechanism=mech.name,
                    implementation=mech.implementation,
                    phase=phase,
                    seq_len=s,
                    status="correctness_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                for phase in phases
                for s in seq_lens
            ]
        verification = (check.max_abs_err, check.rel_err, check.reference)
        if verbose:
            verdict = "PASS" if check.passed else "FAIL"
            print(
                f"  correctness {verdict}: rel={check.rel_err:.3e} "
                f"max|err|={check.max_abs_err:.3e} vs {check.reference}",
                flush=True,
            )
        if not check.passed:
            return [
                PhaseResult(
                    mechanism=mech.name,
                    implementation=mech.implementation,
                    phase=phase,
                    seq_len=s,
                    status="correctness_failed",
                    correctness_max_abs_err=check.max_abs_err,
                    correctness_rel_err=check.rel_err,
                    correctness_reference=check.reference,
                    error="output did not match reference; latency not measured",
                )
                for phase in phases
                for s in seq_lens
            ]

    results: list[PhaseResult] = []
    for phase in phases:
        for seq_len in seq_lens:
            res = measure_phase(mech, phase, seq_len, verification)
            results.append(res)
            if verbose:
                _print_row(res)
    return results


def _print_row(res: PhaseResult) -> None:
    if res.status != "ok":
        print(f"  {res.phase:<7} S={res.seq_len:<6} {res.status.upper()}: {res.error}", flush=True)
        return
    unit = "ms" if res.phase == "prefill" else "ms/token"
    print(
        f"  {res.phase:<7} S={res.seq_len:<6} {res.latency_ms:9.4f} {unit:<9} "
        f"{res.throughput_tokens_per_s:12.1f} tok/s  "
        f"peak_alloc={res.peak_allocated_mb:9.1f} MB  "
        f"bw={res.bandwidth_gb_per_s:7.1f} GB/s",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.stage1_micro.runner",
        description="Stage 1 component-level microbenchmarks (attention vs. SSM).",
    )
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype (default: bfloat16)")
    parser.add_argument("--min-seq", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=65536)
    parser.add_argument(
        "--component",
        action="append",
        choices=sorted(MECHANISMS),
        help="restrict to one mechanism (repeatable); default: all",
    )
    parser.add_argument(
        "--phase", action="append", choices=PHASES, help="restrict to one phase (repeatable)"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10, help="untimed warmup iterations (>= 10)")
    parser.add_argument("--iters", type=int, default=50, help="timed iterations (>= 50)")
    parser.add_argument(
        "--decode-tokens", type=int, default=128, help="timed decode steps per sequence length"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw/stage1"))
    parser.add_argument("--tag", default="stage1", help="output filename stem")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the reference correctness check (not recommended; results become unverified)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("error: CUDA device unavailable", file=sys.stderr)
        return 1
    if args.warmup < 10 or args.iters < 50:
        print(
            "error: the benchmark protocol requires >= 10 warmup and >= 50 timed iterations",
            file=sys.stderr,
        )
        return 1

    config = BenchConfig(
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        decode_tokens=args.decode_tokens,
    )
    config.validate()

    seq_lens = sequence_lengths(args.min_seq, args.max_seq)
    phases = tuple(args.phase) if args.phase else PHASES
    selected = args.component or list(MECHANISMS)

    run = BenchmarkRun(config=config, environment=collect_environment())
    print(
        f"Stage 1: {len(selected)} mechanism(s) x {len(phases)} phase(s) x "
        f"{len(seq_lens)} sequence length(s) on {run.environment.get('gpu_name', args.device)} "
        f"(SM{run.environment.get('compute_capability', '?')})",
        flush=True,
    )

    for key in selected:
        run.results.extend(
            run_mechanism(MECHANISMS[key], config, seq_lens, phases, args.skip_verify)
        )

    json_path = run.to_json(args.output_dir / f"{args.tag}.json")
    csv_path = run.to_csv(args.output_dir / f"{args.tag}.csv")
    print(f"\nWrote {json_path}\nWrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
