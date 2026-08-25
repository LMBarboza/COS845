"""Benchmark configuration, metric records, and structured result serialization."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

# NVIDIA-published raw bandwidth of the GB10 unified LPDDR5X pool. Used only as a
# reference ceiling in reports -- it is not an HBM-class number and must not be
# described as one.
GB10_PEAK_BANDWIDTH_GBPS = 273.0

BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class BenchConfig:
    """Standard Stage 1 synthetic tensor shape (see repository guidelines, rule 4)."""

    batch_size: int = 1
    hidden_size: int = 2048
    num_heads: int = 32
    head_dim: int = 64
    dtype: str = "bfloat16"
    device: str = "cuda"
    warmup: int = 10
    iters: int = 50
    decode_tokens: int = 128

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def validate(self) -> None:
        expected = self.num_heads * self.head_dim
        if expected != self.hidden_size:
            raise ValueError(
                f"num_heads * head_dim ({expected}) must equal hidden_size ({self.hidden_size})"
            )


@dataclass
class MemoryStats:
    """Peak allocator counters for one measured phase."""

    peak_allocated_mb: float
    peak_reserved_mb: float

    @classmethod
    def capture(cls, device: torch.device | int | None = None) -> MemoryStats:
        return cls(
            peak_allocated_mb=torch.cuda.max_memory_allocated(device) / BYTES_PER_MB,
            peak_reserved_mb=torch.cuda.max_memory_reserved(device) / BYTES_PER_MB,
        )


@dataclass
class PhaseResult:
    """One (mechanism, sequence length, phase) measurement."""

    mechanism: str
    implementation: str
    phase: str
    seq_len: int
    status: str = "ok"
    error: str | None = None

    latency_ms: float | None = None
    latency_median_ms: float | None = None
    latency_std_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_min_ms: float | None = None
    latency_max_ms: float | None = None

    throughput_tokens_per_s: float | None = None
    bytes_moved: int | None = None
    bandwidth_gb_per_s: float | None = None

    peak_allocated_mb: float | None = None
    peak_reserved_mb: float | None = None
    state_memory_mb: float | None = None

    correctness_max_abs_err: float | None = None
    correctness_rel_err: float | None = None
    correctness_reference: str | None = None

    iters: int | None = None
    warmup: int | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRun:
    """A full Stage 1 sweep: environment provenance plus every phase measurement."""

    config: BenchConfig
    results: list[PhaseResult] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def add(self, result: PhaseResult) -> None:
        self.results.append(result)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "config": asdict(self.config),
            "environment": self.environment,
            "results": [r.as_dict() for r in self.results],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def to_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.as_dict() for r in self.results]
        if not rows:
            path.write_text("")
            return path
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path


def effective_throughput(seq_len: int, latency_ms: float) -> float:
    """Tokens per second: sequence length divided by latency in seconds."""
    if latency_ms <= 0:
        return float("nan")
    return seq_len / (latency_ms / 1000.0)


def bandwidth_gb_per_s(bytes_moved: int, latency_ms: float) -> float:
    """Empirical (bytes read + written) / execution time, in GB/s."""
    if latency_ms <= 0:
        return float("nan")
    return (bytes_moved / BYTES_PER_GB) / (latency_ms / 1000.0)


def _nvidia_smi_field(field_name: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field_name}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else None


def collect_environment() -> dict[str, Any]:
    """Capture hardware/software provenance so a results file is self-describing."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / BYTES_PER_GB,
                "multi_processor_count": props.multi_processor_count,
                "torch_arch_list": torch.cuda.get_arch_list(),
                "driver_version": _nvidia_smi_field("driver_version"),
            }
        )
    env["reference_peak_bandwidth_gb_per_s"] = GB10_PEAK_BANDWIDTH_GBPS
    return env
