"""DeepSeek Multi-Head Latent Attention (MLA), reference (non-fused) implementation.

FlashMLA ships SM90 (WGMMA) and SM100 (tcgen05) backends; neither instruction path
exists on SM121, so there is no fused MLA kernel to benchmark on GB10. Per the
repository guidelines the mechanism is benchmarked against a reference PyTorch
implementation and labelled as such, rather than skipped silently or substituted
with a different kernel.

Two paths are implemented, because they are what MLA is actually about:

* **Prefill** projects the compressed latent ``c_kv`` back up to per-head keys and
  values and runs ordinary attention over them.
* **Decode** uses the matrix-absorption form: ``W_UK`` is folded into the query and
  ``W_UV`` into the output, so the attention runs directly against the latent cache.
  The KV cache holds ``kv_lora_rank + qk_rope_head_dim`` values per token instead of
  ``2 * num_heads * head_dim`` -- the compression this mechanism exists to provide.

``decode`` is verified against the un-absorbed path at a small sequence length, so a
mis-absorbed projection cannot pass as a fast result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..utils.metrics import BYTES_PER_MB, BenchConfig
from .base import (
    Availability,
    Mechanism,
    PreparedPhase,
    VerificationResult,
    guard_score_matrix,
)


@dataclass(frozen=True)
class MLAShape:
    """Low-rank shape parameters, scaled to this study's standard hidden size.

    DeepSeek-V3 uses a 7168-wide model; these ranks keep the same structure and
    compression character at the 2048-wide standard shape used throughout Stage 1.
    """

    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def cache_per_token(self) -> int:
        """Latent values cached per token (vs ``2 * H * D`` for standard MHA)."""
        return self.kv_lora_rank + self.qk_rope_head_dim


def _rope_tables(
    seq_len: int, dim: int, device: torch.device, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute ``(cos, sin)`` of shape ``(seq_len, dim)`` for rotary embedding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to the trailing dimension of ``x``.

    ``cos``/``sin`` broadcast against ``x``; the caller is responsible for slicing
    them to the positions ``x`` occupies.
    """
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


class MLAWeights(nn.Module):
    """The projection stack of one MLA layer."""

    def __init__(self, hidden: int, num_heads: int, shape: MLAShape) -> None:
        super().__init__()
        self.shape = shape
        self.num_heads = num_heads
        self.q_down = nn.Linear(hidden, shape.q_lora_rank, bias=False)
        self.q_up = nn.Linear(shape.q_lora_rank, num_heads * shape.qk_head_dim, bias=False)
        self.kv_down = nn.Linear(hidden, shape.kv_lora_rank, bias=False)
        self.k_rope = nn.Linear(hidden, shape.qk_rope_head_dim, bias=False)
        self.k_up = nn.Linear(shape.kv_lora_rank, num_heads * shape.qk_nope_head_dim, bias=False)
        self.v_up = nn.Linear(shape.kv_lora_rank, num_heads * shape.v_head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * shape.v_head_dim, hidden, bias=False)

    @property
    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def k_up_absorb(self) -> torch.Tensor:
        """``W_UK`` as ``(H, qk_nope_head_dim, kv_lora_rank)`` for query absorption."""
        return self.k_up.weight.view(self.num_heads, self.shape.qk_nope_head_dim, -1)

    def v_up_absorb(self) -> torch.Tensor:
        """``W_UV`` as ``(H, v_head_dim, kv_lora_rank)`` for output absorption."""
        return self.v_up.weight.view(self.num_heads, self.shape.v_head_dim, -1)


class DeepSeekMLA(Mechanism):
    key = "mla"
    name = "DeepSeek MLA (reference)"

    def __init__(self, config: BenchConfig, shape: MLAShape | None = None) -> None:
        super().__init__(config)
        self.shape = shape or MLAShape()
        self.weights = MLAWeights(config.hidden_size, config.num_heads, self.shape).to(
            device=self.device, dtype=self.dtype
        )
        self.weights.eval()
        self.scale = self.shape.qk_head_dim**-0.5

    @classmethod
    def availability(cls) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "unavailable", "CUDA device unavailable")
        return Availability(
            True,
            "reference-pytorch-mla",
            "no fused MLA kernel on SM121 (FlashMLA requires SM90 WGMMA or SM100 tcgen05)",
        )

    # -- forward paths ----------------------------------------------------------

    def _project_queries(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q_nope, q_pe)`` of shape ``(B, H, S, *)``."""
        b, s, _ = x.shape
        w, sh = self.weights, self.shape
        q = w.q_up(w.q_down(x)).view(b, s, self.config.num_heads, sh.qk_head_dim)
        q = q.transpose(1, 2)
        q_nope, q_pe = q.split([sh.qk_nope_head_dim, sh.qk_rope_head_dim], dim=-1)
        return q_nope, apply_rope(q_pe, cos, sin)

    def _project_latents(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached pair ``(c_kv, k_pe)`` of shape ``(B, S, *)``."""
        w = self.weights
        c_kv = w.kv_down(x)
        k_pe = apply_rope(w.k_rope(x).unsqueeze(1), cos, sin).squeeze(1)
        return c_kv, k_pe

    def prefill_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-prompt MLA: decompress the latent cache and attend over per-head KV."""
        b, s, _ = x.shape
        h, sh, w = self.config.num_heads, self.shape, self.weights
        cos, sin = _rope_tables(s, sh.qk_rope_head_dim, self.device)
        q_nope, q_pe = self._project_queries(x, cos, sin)
        c_kv, k_pe = self._project_latents(x, cos, sin)

        k_nope = w.k_up(c_kv).view(b, s, h, sh.qk_nope_head_dim).transpose(1, 2)
        v = w.v_up(c_kv).view(b, s, h, sh.v_head_dim).transpose(1, 2)
        k = torch.cat([k_nope, k_pe.unsqueeze(1).expand(b, h, s, sh.qk_rope_head_dim)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores.masked_fill_(
            torch.ones(s, s, dtype=torch.bool, device=self.device).triu(1), float("-inf")
        )
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return w.o_proj(out.transpose(1, 2).reshape(b, s, h * sh.v_head_dim))

    def decode_absorbed(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step decode attending directly against the compressed latent cache.

        ``W_UK`` is absorbed into the query and ``W_UV`` into the output, so no
        per-head key or value is ever materialized: only ``c_kv`` and ``k_pe`` are read.
        """
        w = self.weights
        q_absorbed = torch.einsum("bhqd,hdc->bhqc", q_nope, w.k_up_absorb())
        scores = torch.einsum("bhqc,bsc->bhqs", q_absorbed, c_kv)
        scores = scores + torch.einsum("bhqr,bsr->bhqs", q_pe, k_pe)
        probs = torch.softmax(scores * self.scale, dim=-1)
        out_latent = torch.einsum("bhqs,bsc->bhqc", probs, c_kv)
        out = torch.einsum("bhqc,hvc->bhqv", out_latent, w.v_up_absorb())
        b, h, q_len, _ = out.shape
        return w.o_proj(out.transpose(1, 2).reshape(b, q_len, h * self.shape.v_head_dim))

    def _decode_unabsorbed(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Reference for :meth:`decode_absorbed`: decompress, then attend normally."""
        b, s, _ = c_kv.shape
        h, sh, w = self.config.num_heads, self.shape, self.weights
        k_nope = w.k_up(c_kv).view(b, s, h, sh.qk_nope_head_dim).transpose(1, 2)
        v = w.v_up(c_kv).view(b, s, h, sh.v_head_dim).transpose(1, 2)
        k = torch.cat([k_nope, k_pe.unsqueeze(1).expand(b, h, s, sh.qk_rope_head_dim)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        probs = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(probs, v)
        q_len = out.shape[2]
        return w.o_proj(out.transpose(1, 2).reshape(b, q_len, h * sh.v_head_dim))

    # -- correctness ------------------------------------------------------------

    def verify(self, seq_len: int = 256, tolerance: float = 2e-2) -> VerificationResult:
        """Check the absorbed decode path against the decompressed reference path."""
        cfg, sh = self.config, self.shape
        b = cfg.batch_size
        x = self.randn(b, seq_len, cfg.hidden_size)
        cos, sin = _rope_tables(seq_len, sh.qk_rope_head_dim, self.device)
        c_kv, k_pe = self._project_latents(x, cos, sin)
        q_step = self.randn(b, cfg.num_heads, 1, sh.qk_nope_head_dim)
        q_pe = self.randn(b, cfg.num_heads, 1, sh.qk_rope_head_dim)

        with torch.no_grad():
            got = self.decode_absorbed(q_step, q_pe, c_kv, k_pe)
            want = self._decode_unabsorbed(q_step, q_pe, c_kv, k_pe)
        return self.check(
            got,
            want,
            "MLA decode without matrix absorption (decompressed per-head KV)",
            tolerance,
        )

    # -- phases -----------------------------------------------------------------

    def prepare_prefill(self, seq_len: int) -> PreparedPhase:
        cfg, sh = self.config, self.shape
        b, h = cfg.batch_size, cfg.num_heads
        guard_score_matrix(b, h, seq_len, self.itemsize)
        x = self.randn(b, seq_len, cfg.hidden_size)

        def run() -> torch.Tensor:
            with torch.no_grad():
                return self.prefill_forward(x)

        item = self.itemsize
        act = b * seq_len * item
        activation_bytes = act * (
            cfg.hidden_size  # x read
            + 2 * sh.q_lora_rank  # q latent write + read
            + h * sh.qk_head_dim  # q write
            + 2 * sh.kv_lora_rank  # c_kv write + read
            + 2 * sh.qk_rope_head_dim  # k_pe write + read
            + h * sh.qk_nope_head_dim  # k_nope write
            + 2 * h * sh.v_head_dim  # v write + attention output write
            + cfg.hidden_size  # o_proj write
        )
        # The reference path materializes the score matrix, like the naive baseline.
        score_bytes = 6 * b * h * seq_len * seq_len * item
        return PreparedPhase(
            fn=run,
            bytes_moved=activation_bytes + score_bytes + self.weights.weight_bytes,
            tokens_processed=seq_len,
            state_memory_mb=b * seq_len * sh.cache_per_token * item / BYTES_PER_MB,
            notes=(
                f"latent cache {sh.cache_per_token} values/token vs "
                f"{2 * h * cfg.head_dim} for MHA; reference (non-fused) kernel"
            ),
        )

    def prepare_decode(self, seq_len: int) -> PreparedPhase:
        cfg, sh = self.config, self.shape
        b, h = cfg.batch_size, cfg.num_heads
        capacity = seq_len + cfg.decode_tokens + cfg.warmup + 1
        c_kv_cache = self.randn(b, capacity, sh.kv_lora_rank)
        k_pe_cache = self.randn(b, capacity, sh.qk_rope_head_dim)
        q_nope = self.randn(b, h, 1, sh.qk_nope_head_dim)
        q_pe = self.randn(b, h, 1, sh.qk_rope_head_dim)
        pos = [seq_len]

        def run() -> torch.Tensor:
            p = pos[0]
            with torch.no_grad():
                out = self.decode_absorbed(
                    q_nope, q_pe, c_kv_cache[:, : p + 1], k_pe_cache[:, : p + 1]
                )
            pos[0] = p + 1
            return out

        def rewind() -> None:
            pos[0] = seq_len

        item = self.itemsize
        avg_ctx = seq_len + cfg.decode_tokens / 2
        # The latent cache is read twice per step: once for scores, once for output.
        cache_bytes = int(
            2 * b * avg_ctx * sh.kv_lora_rank * item + b * avg_ctx * sh.qk_rope_head_dim * item
        )
        absorb_bytes = (self.weights.k_up.weight.numel() + self.weights.v_up.weight.numel()) * item
        return PreparedPhase(
            fn=run,
            bytes_moved=cache_bytes + absorb_bytes,
            tokens_processed=1,
            state_memory_mb=self.tensor_mb(c_kv_cache, k_pe_cache),
            pre_measure_fn=rewind,
            iters=cfg.decode_tokens,
            notes=(
                f"matrix-absorbed decode over latent cache ({sh.cache_per_token} values/token); "
                f"capacity {capacity} tokens"
            ),
        )
