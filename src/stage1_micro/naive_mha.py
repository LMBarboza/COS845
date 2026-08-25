"""Naive multi-head attention: the O(S^2) materialized-score baseline control.

This is deliberately the *unoptimized* path -- the S x S score matrix is written to
DRAM, masked, softmaxed, and read back. It is the control against which the
IO-aware and recurrent mechanisms are compared, and it is also the correctness
reference for the other attention variants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import (
    Availability,
    Mechanism,
    PreparedPhase,
    VerificationResult,
    guard_score_matrix,
)

# Passes over the B x H x S x S score matrix in the eager formulation:
# matmul write, masked_fill_ read+write, softmax read+write, second matmul read.
NAIVE_SCORE_PASSES = 6


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager attention over ``(B, H, S, D)`` tensors with a materialized score matrix.

    ``causal_mask`` is a boolean ``(Sq, Sk)`` tensor that is ``True`` where a position
    must be masked out. Pass ``None`` for the single-query decode step, where every
    cached key is visible.
    """
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal_mask is not None:
        scores.masked_fill_(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class NaiveMHA(Mechanism):
    key = "naive"
    name = "Naive MHA"

    @classmethod
    def availability(cls) -> Availability:
        return Availability(
            available=torch.cuda.is_available(),
            implementation="torch-eager-matmul-softmax",
            reason="" if torch.cuda.is_available() else "CUDA device unavailable",
        )

    # -- correctness ------------------------------------------------------------

    def verify(self, seq_len: int = 256, tolerance: float = 2e-2) -> VerificationResult:
        """Cross-check the eager path against PyTorch's fused SDPA."""
        cfg = self.config
        shape = (cfg.batch_size, cfg.num_heads, seq_len, cfg.head_dim)
        q, k, v = (self.randn(*shape) for _ in range(3))
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device).triu(1)
        got = naive_attention(q, k, v, mask)
        want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.check(
            got,
            want,
            "torch.nn.functional.scaled_dot_product_attention (is_causal=True)",
            tolerance,
        )

    # -- phases -----------------------------------------------------------------

    def prepare_prefill(self, seq_len: int) -> PreparedPhase:
        cfg = self.config
        b, h, d = cfg.batch_size, cfg.num_heads, cfg.head_dim
        guard_score_matrix(b, h, seq_len, self.itemsize)
        q, k, v = (self.randn(b, h, seq_len, d) for _ in range(3))
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device).triu(1)

        def run() -> torch.Tensor:
            return naive_attention(q, k, v, mask)

        qkv_bytes = 4 * b * h * seq_len * d * self.itemsize  # q, k, v read + output write
        score_bytes = NAIVE_SCORE_PASSES * b * h * seq_len * seq_len * self.itemsize
        return PreparedPhase(
            fn=run,
            bytes_moved=qkv_bytes + score_bytes,
            tokens_processed=seq_len,
            state_memory_mb=self.tensor_mb(k, v),
            notes=f"materialized {b * h} x {seq_len} x {seq_len} score matrix",
        )

    def prepare_decode(self, seq_len: int) -> PreparedPhase:
        """Autoregressive decode: 128 timed single-token steps over a growing KV cache."""
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
            out = naive_attention(q_step, k_cache[:, :, : p + 1], v_cache[:, :, : p + 1])
            pos[0] = p + 1
            return out

        def rewind() -> None:
            pos[0] = seq_len

        # Average context length over the 128 timed steps.
        avg_ctx = seq_len + cfg.decode_tokens / 2
        kv_bytes = int(2 * b * h * avg_ctx * d * self.itemsize)
        score_bytes = int(4 * b * h * avg_ctx * self.itemsize)
        return PreparedPhase(
            fn=run,
            bytes_moved=kv_bytes + score_bytes,
            tokens_processed=1,
            state_memory_mb=self.tensor_mb(k_cache, v_cache),
            pre_measure_fn=rewind,
            iters=cfg.decode_tokens,
            notes=f"KV cache capacity {capacity} tokens",
        )
