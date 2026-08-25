"""Shared protocol for Stage 1 component-level microbenchmarks.

Every mechanism module exposes a :class:`Mechanism` subclass that knows how to

1. report whether it can run at all on this hardware (``availability``),
2. verify itself against a reference PyTorch implementation (``verify``) --
   repository rule 5, correctness before speed,
3. build a zero-argument callable for the prefill and decode phases, together with
   an analytic estimate of the bytes that callable moves.

The runner owns all timing, memory-stat resetting, and serialization so that every
mechanism is measured under identical conditions.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..utils.metrics import BYTES_PER_MB, BenchConfig

#: Fraction of the unified 128 GB pool a single materialized score matrix may claim.
#: GB10 shares this memory with the OS and CPU-side processes, so an allocation that
#: would "fit" a dedicated-VRAM budget can still take the whole machine down. The
#: guard turns that into a recorded ``oom`` row instead of an OOM-killed run.
SCORE_MATRIX_MEMORY_FRACTION = 0.4


class ScoreMatrixTooLarge(MemoryError):
    """Raised when a materialized S x S score matrix would exceed the memory guard."""


def guard_score_matrix(batch: int, heads: int, seq_len: int, itemsize: int) -> None:
    """Refuse to materialize a score matrix that would exhaust the unified pool."""
    if not torch.cuda.is_available():
        return
    needed = batch * heads * seq_len * seq_len * itemsize
    budget = torch.cuda.get_device_properties(0).total_memory * SCORE_MATRIX_MEMORY_FRACTION
    if needed > budget:
        raise ScoreMatrixTooLarge(
            f"materialized score matrix would need {needed / 1024**3:.1f} GiB "
            f"(> {budget / 1024**3:.1f} GiB guard on a shared 128 GB unified pool)"
        )


@dataclass
class Availability:
    """Whether a mechanism can be benchmarked here, and which kernel it would use."""

    available: bool
    implementation: str
    reason: str = ""


@dataclass
class PreparedPhase:
    """A ready-to-time callable plus the metadata the runner records alongside it.

    ``bytes_moved`` is an analytic estimate of DRAM traffic for one invocation: it
    is used for the empirical-bandwidth metric and is deliberately per-mechanism,
    since the whole point of the comparison is that a naive kernel moves the
    materialized S x S score matrix while an IO-aware or recurrent one does not.
    """

    fn: Callable[[], object]
    bytes_moved: int
    tokens_processed: int
    state_memory_mb: float | None = None
    reset_fn: Callable[[], None] | None = None
    pre_measure_fn: Callable[[], None] | None = None
    iters: int | None = None
    notes: str | None = None


@dataclass
class VerificationResult:
    """Outcome of a small-sequence check against a reference implementation."""

    max_abs_err: float
    rel_err: float
    reference: str
    passed: bool


class Mechanism(abc.ABC):
    """Base class for a Stage 1 mechanism under test."""

    #: Short key used on the command line and in result files (e.g. ``"mamba2"``).
    key: str = ""
    #: Human-readable mechanism name (e.g. ``"Mamba-2 (SSD)"``).
    name: str = ""

    def __init__(self, config: BenchConfig) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype
        #: Which kernel actually produced the numbers, recorded in every result row.
        self.implementation = self.availability().implementation

    # -- capability -------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def availability(cls) -> Availability:
        """Report whether this mechanism's kernel is importable and usable here."""

    # -- correctness ------------------------------------------------------------

    @abc.abstractmethod
    def verify(self, seq_len: int = 256, tolerance: float = 2e-2) -> VerificationResult:
        """Compare against a reference implementation at a small sequence length."""

    # -- phases -----------------------------------------------------------------

    @abc.abstractmethod
    def prepare_prefill(self, seq_len: int) -> PreparedPhase:
        """Build a callable that processes a full prompt of ``seq_len`` tokens."""

    @abc.abstractmethod
    def prepare_decode(self, seq_len: int) -> PreparedPhase:
        """Build a callable that processes token S+1 given existing state of size S."""

    # -- helpers ----------------------------------------------------------------

    @property
    def itemsize(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    def randn(self, *shape: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.randn(*shape, device=self.device, dtype=dtype or self.dtype)

    @staticmethod
    def tensor_mb(*tensors: torch.Tensor) -> float:
        return sum(t.numel() * t.element_size() for t in tensors) / BYTES_PER_MB

    @staticmethod
    def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a.float() - b.float()).abs().max().item()

    @classmethod
    def check(
        cls, got: torch.Tensor, want: torch.Tensor, reference: str, tolerance: float
    ) -> VerificationResult:
        """Score ``got`` against ``want`` on relative error.

        Absolute error alone is not a usable gate here: the SSD and delta-rule scans
        produce outputs two orders of magnitude larger than an attention output, so the
        same bfloat16 rounding shows up as a very different absolute number. The pass
        criterion is ``max|got - want| / max|want|``.
        """
        abs_err = cls.max_abs_err(got, want)
        denom = max(want.float().abs().max().item(), 1e-6)
        rel = abs_err / denom
        return VerificationResult(
            max_abs_err=abs_err, rel_err=rel, reference=reference, passed=rel <= tolerance
        )
