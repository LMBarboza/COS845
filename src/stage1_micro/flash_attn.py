"""IO-aware tiled attention (FlashAttention family).

On SM121 (GB10) the upstream ``flash-attn`` package has no prebuilt aarch64 wheel and
its Blackwell paths target SM100. Rather than report a missing kernel as a benchmark
result, this module resolves the best available IO-aware backend at runtime and
records exactly which one produced the numbers:

* ``flash-attn`` (``flash_attn_func``) when the package imports and runs, otherwise
* PyTorch SDPA pinned to a single tiled backend -- ``FLASH_ATTENTION``, then
  ``CUDNN_ATTENTION``, then ``EFFICIENT_ATTENTION``.

The ``MATH`` backend is never selected: it is the materialized-score path already
measured by :mod:`naive_mha`, so silently falling back to it would report the
baseline under the FlashAttention label.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..utils.metrics import BenchConfig
from .base import Availability, Mechanism, PreparedPhase, VerificationResult
from .naive_mha import naive_attention

#: Tiled SDPA backends in preference order. MATH is intentionally excluded.
SDPA_BACKENDS: tuple[tuple[str, SDPBackend], ...] = (
    ("torch-sdpa-flash", SDPBackend.FLASH_ATTENTION),
    ("torch-sdpa-cudnn", SDPBackend.CUDNN_ATTENTION),
    ("torch-sdpa-mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
)


def _flash_attn_package_works() -> bool:
    """Import ``flash_attn`` and actually run it -- an import alone proves nothing."""
    try:
        from flash_attn import flash_attn_func
    except (ImportError, RuntimeError, OSError):
        return False
    try:
        x = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(x, x, x, causal=True)
        torch.cuda.synchronize()
    except Exception:
        return False
    return True


def _working_sdpa_backend() -> tuple[str, SDPBackend] | None:
    """Return the first tiled SDPA backend that executes on this device."""
    if not torch.cuda.is_available():
        return None
    q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    for label, backend in SDPA_BACKENDS:
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize()
        except Exception:
            continue
        return label, backend
    return None


class FlashAttention(Mechanism):
    key = "flash"
    name = "FlashAttention (IO-aware)"

    @classmethod
    def availability(cls) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "unavailable", "CUDA device unavailable")
        if _flash_attn_package_works():
            return Availability(True, "flash-attn", "")
        resolved = _working_sdpa_backend()
        if resolved is None:
            return Availability(
                False,
                "unavailable",
                "no tiled attention backend (flash-attn absent; every SDPA tiled backend failed)",
            )
        label, _ = resolved
        return Availability(
            True,
            label,
            "flash-attn package unavailable on aarch64/SM121; using PyTorch SDPA tiled backend",
        )

    def __init__(self, config: BenchConfig) -> None:
        super().__init__(config)
        avail = self.availability()
        if not avail.available:
            raise RuntimeError(avail.reason)
        self.implementation = avail.implementation
        self._use_package = avail.implementation == "flash-attn"
        self._backend: SDPBackend | None = None
        if not self._use_package:
            resolved = _working_sdpa_backend()
            assert resolved is not None
            self._backend = resolved[1]

    # -- kernel dispatch --------------------------------------------------------

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
        """Attention over ``(B, H, S, D)`` tensors using the resolved backend."""
        if self._use_package:
            from flash_attn import flash_attn_func

            # flash_attn_func expects (B, S, H, D).
            out = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal
            )
            return out.transpose(1, 2)
        with sdpa_kernel(self._backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # -- correctness ------------------------------------------------------------

    def verify(self, seq_len: int = 256, tolerance: float = 2e-2) -> VerificationResult:
        cfg = self.config
        shape = (cfg.batch_size, cfg.num_heads, seq_len, cfg.head_dim)
        q, k, v = (self.randn(*shape) for _ in range(3))
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device).triu(1)
        got = self._attend(q, k, v, causal=True)
        want = naive_attention(q, k, v, mask)
        return self.check(
            got, want, "naive_mha.naive_attention (eager materialized softmax)", tolerance
        )

    # -- phases -----------------------------------------------------------------

    def prepare_prefill(self, seq_len: int) -> PreparedPhase:
        cfg = self.config
        b, h, d = cfg.batch_size, cfg.num_heads, cfg.head_dim
        q, k, v = (self.randn(b, h, seq_len, d) for _ in range(3))

        def run() -> torch.Tensor:
            return self._attend(q, k, v, causal=True)

        # Tiled attention never writes the S x S score matrix to DRAM: q, k, v in and
        # the output out is the whole of its traffic.
        return PreparedPhase(
            fn=run,
            bytes_moved=4 * b * h * seq_len * d * self.itemsize,
            tokens_processed=seq_len,
            state_memory_mb=self.tensor_mb(k, v),
            notes=f"backend={self.implementation}",
        )

    def prepare_decode(self, seq_len: int) -> PreparedPhase:
        cfg = self.config
        b, h, d = cfg.batch_size, cfg.num_heads, cfg.head_dim
        capacity = seq_len + cfg.decode_tokens + cfg.warmup + 1
        k_cache = self.randn(b, h, capacity, d)
        v_cache = self.randn(b, h, capacity, d)
        q_step = self.randn(b, h, 1, d)
        pos = [seq_len]

        def run() -> torch.Tensor:
            p = pos[0]
            k_cache[:, :, p : p + 1].copy_(q_step)
            v_cache[:, :, p : p + 1].copy_(q_step)
            out = self._attend(q_step, k_cache[:, :, : p + 1], v_cache[:, :, : p + 1], causal=False)
            pos[0] = p + 1
            return out

        def rewind() -> None:
            pos[0] = seq_len

        avg_ctx = seq_len + cfg.decode_tokens / 2
        return PreparedPhase(
            fn=run,
            bytes_moved=int(2 * b * h * avg_ctx * d * self.itemsize),
            tokens_processed=1,
            state_memory_mb=self.tensor_mb(k_cache, v_cache),
            pre_measure_fn=rewind,
            iters=cfg.decode_tokens,
            notes=f"backend={self.implementation}; KV cache capacity {capacity} tokens",
        )
