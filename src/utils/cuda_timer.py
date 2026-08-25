"""CUDA-accurate timing helpers.

Rule 1 of the repository guidelines: never time GPU work with ``time.time()``.
Everything here is measured with ``torch.cuda.Event(enable_timing=True)`` pairs and
explicit ``torch.cuda.synchronize()`` barriers.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50


@dataclass
class TimingStats:
    """Latency statistics (all values in milliseconds) for a timed callable."""

    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    iters: int
    warmup: int
    samples_ms: list[float] = field(default_factory=list, repr=False)

    def as_dict(self, include_samples: bool = False) -> dict[str, float | int | list[float]]:
        out: dict[str, float | int | list[float]] = {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p95_ms": self.p95_ms,
            "iters": self.iters,
            "warmup": self.warmup,
        }
        if include_samples:
            out["samples_ms"] = self.samples_ms
        return out


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize(samples_ms: list[float], warmup: int) -> TimingStats:
    """Build a :class:`TimingStats` from a list of per-iteration millisecond samples."""
    ordered = sorted(samples_ms)
    return TimingStats(
        mean_ms=statistics.fmean(samples_ms),
        median_ms=statistics.median(samples_ms),
        std_ms=statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        min_ms=ordered[0],
        max_ms=ordered[-1],
        p95_ms=_percentile(ordered, 0.95),
        iters=len(samples_ms),
        warmup=warmup,
        samples_ms=samples_ms,
    )


class CudaTimer:
    """Context manager timing a single GPU region with CUDA events.

    ``elapsed_ms`` is only valid after the ``with`` block exits (the exit path
    synchronizes on the stop event).
    """

    def __init__(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: float = float("nan")

    def __enter__(self) -> CudaTimer:
        torch.cuda.synchronize()
        self.start_event.record()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop_event.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.stop_event)


def time_callable(
    fn: Callable[[], object],
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    reset_fn: Callable[[], None] | None = None,
    pre_measure_fn: Callable[[], None] | None = None,
) -> TimingStats:
    """Time ``fn`` with CUDA events after ``warmup`` untimed iterations.

    ``reset_fn`` runs (untimed) before every iteration, for mechanisms that mutate
    state per call. ``pre_measure_fn`` runs once between warmup and the timed loop --
    used by autoregressive decode, where the 128 timed steps must start from the
    same context length the warmup steps started from.
    """
    for _ in range(warmup):
        if reset_fn is not None:
            reset_fn()
        fn()
    torch.cuda.synchronize()

    if pre_measure_fn is not None:
        pre_measure_fn()
        torch.cuda.synchronize()

    samples: list[float] = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if reset_fn is not None:
            reset_fn()
            torch.cuda.synchronize()
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(stops[i]) for i in range(iters)]
    return summarize(samples, warmup)


def reset_memory_stats(device: torch.device | int | None = None) -> None:
    """Clear the caching allocator and peak-memory counters before a measurement."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
