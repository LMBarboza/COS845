"""Kimi Delta Attention (KDA), the gated linear-attention core of Kimi Linear.

KDA is a delta-rule linear attention with *per-channel* (diagonal) forget gates:
each step decays the recurrent state elementwise along the key dimension, then
applies a delta-rule correction weighted by ``beta``. Like Mamba-2 it carries a
fixed-size ``K x V`` state per head, so decode cost does not grow with context; unlike
Mamba-2 its gate is fine-grained rather than a single scalar per head.

The kernels come from ``flash-linear-attention`` (pure Triton, Ampere-compatible by
design, which is what makes them usable on SM121 where Triton schedules SM12x as
SM80). Prefill uses the chunked scan; decode uses the fused recurrent step. Both are
checked against ``naive_recurrent_kda``, the reference sequential implementation.

The gate computation ``g = -exp(A_log) * softplus(raw + dt_bias)`` is deliberately
inside the timed region: it is part of the mechanism's per-token work, not setup.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.metrics import BYTES_PER_MB, BenchConfig
from .base import Availability, Mechanism, PreparedPhase, VerificationResult

#: Chunk length used by the KDA chunked scan (fla default).
CHUNK_SIZE = 64
_FLA_KDA = "fla-triton-kda"


def _kda_available() -> bool:
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda  # noqa: F401
    except (ImportError, RuntimeError, OSError):
        return False
    return True


class KimiDeltaAttention(Mechanism):
    key = "kda"
    name = "Kimi Delta Attention"

    def __init__(self, config: BenchConfig) -> None:
        super().__init__(config)
        self.num_heads = config.num_heads
        self.key_dim = config.head_dim
        self.value_dim = config.head_dim
        # Per-head decay magnitude and per-channel gate bias, as in Kimi Linear.
        self.a_log = torch.rand(self.num_heads, device=self.device, dtype=torch.float32)
        self.dt_bias = torch.rand(
            self.num_heads, self.key_dim, device=self.device, dtype=torch.float32
        )

    @classmethod
    def availability(cls) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "unavailable", "CUDA device unavailable")
        if not _kda_available():
            return Availability(
                False, "unavailable", "flash-linear-attention KDA kernels are not importable"
            )
        return Availability(True, _FLA_KDA, "")

    # -- input construction -----------------------------------------------------

    def _make_inputs(self, seq_len: int) -> dict[str, torch.Tensor]:
        b, h = self.config.batch_size, self.num_heads
        return {
            "q": self.randn(b, seq_len, h, self.key_dim),
            "k": self.randn(b, seq_len, h, self.key_dim),
            "v": self.randn(b, seq_len, h, self.value_dim),
            "g_raw": torch.randn(
                b, seq_len, h, self.key_dim, device=self.device, dtype=torch.float32
            ),
            "beta": torch.rand(b, seq_len, h, device=self.device, dtype=torch.float32),
        }

    def _log_decay(self, g_raw: torch.Tensor) -> torch.Tensor:
        """Fine-grained log-space forget gate, one value per (head, key channel)."""
        return -torch.exp(self.a_log).view(1, 1, -1, 1) * F.softplus(
            g_raw + self.dt_bias.view(1, 1, self.num_heads, self.key_dim)
        )

    # -- forward paths ----------------------------------------------------------

    def prefill_forward(self, inp: dict[str, torch.Tensor]) -> torch.Tensor:
        from fla.ops.kda import chunk_kda

        g = self._log_decay(inp["g_raw"])
        out, _ = chunk_kda(
            inp["q"], inp["k"], inp["v"], g, inp["beta"], use_qk_l2norm_in_kernel=True
        )
        return out

    def decode_forward(
        self, inp: dict[str, torch.Tensor], state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.ops.kda import fused_recurrent_kda

        g = self._log_decay(inp["g_raw"])
        return fused_recurrent_kda(
            inp["q"],
            inp["k"],
            inp["v"],
            g,
            inp["beta"],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    # -- correctness ------------------------------------------------------------

    def verify(self, seq_len: int = 256, tolerance: float = 2e-2) -> VerificationResult:
        """Check the chunked scan against fla's sequential reference recurrence."""
        from fla.ops.kda.naive import naive_recurrent_kda

        inp = self._make_inputs(seq_len)
        g = self._log_decay(inp["g_raw"])
        got = self.prefill_forward(inp)
        # The reference takes q/k already normalized; the kernels do it internally.
        q_n = F.normalize(inp["q"].float(), dim=-1).to(self.dtype)
        k_n = F.normalize(inp["k"].float(), dim=-1).to(self.dtype)
        want, _ = naive_recurrent_kda(q_n, k_n, inp["v"], g, inp["beta"])
        return self.check(got, want, "fla.ops.kda.naive.naive_recurrent_kda", tolerance)

    # -- phases -----------------------------------------------------------------

    def prepare_prefill(self, seq_len: int) -> PreparedPhase:
        inp = self._make_inputs(seq_len)

        def run() -> torch.Tensor:
            return self.prefill_forward(inp)

        cfg = self.config
        item = self.itemsize
        b, h, k, v = cfg.batch_size, self.num_heads, self.key_dim, self.value_dim
        io_bytes = (
            3 * b * seq_len * h * k * item  # q, k, v reads
            + b * seq_len * h * v * item  # output write
            + 3 * b * seq_len * h * k * 4  # g raw read, gate write, gate read (fp32)
            + 2 * b * seq_len * h * 4  # beta read + cumulative gate bookkeeping
        )
        num_chunks = max(1, seq_len // CHUNK_SIZE)
        chunk_state_bytes = 2 * b * num_chunks * h * k * v * 4
        return PreparedPhase(
            fn=run,
            bytes_moved=io_bytes + chunk_state_bytes,
            tokens_processed=seq_len,
            state_memory_mb=b * h * k * v * 4 / BYTES_PER_MB,
            notes=f"chunk={CHUNK_SIZE}, fine-grained per-channel gate, backend={self.implementation}",
        )

    def prepare_decode(self, seq_len: int) -> PreparedPhase:
        """Decode is O(1) in ``seq_len``; the ``K x V`` state per head is fixed-size."""
        cfg = self.config
        b, h, k, v = cfg.batch_size, self.num_heads, self.key_dim, self.value_dim
        step = self._make_inputs(1)
        state = torch.zeros(b, h, k, v, device=self.device, dtype=torch.float32)
        carry = {"state": state}

        def run() -> torch.Tensor:
            out, new_state = self.decode_forward(step, carry["state"])
            carry["state"] = new_state
            return out

        def reset_state() -> None:
            carry["state"] = torch.zeros(b, h, k, v, device=self.device, dtype=torch.float32)

        state_bytes = b * h * k * v * 4
        io_bytes = (
            2 * state_bytes  # state read + write
            + 3 * b * h * k * self.itemsize  # q, k, v
            + b * h * k * 4  # gate
        )
        return PreparedPhase(
            fn=run,
            bytes_moved=io_bytes,
            tokens_processed=1,
            state_memory_mb=state_bytes / BYTES_PER_MB,
            pre_measure_fn=reset_state,
            iters=cfg.decode_tokens,
            notes=(
                f"fixed {h}x{k}x{v} fp32 state, independent of S; backend={self.implementation}"
            ),
        )
