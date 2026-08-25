"""Mamba-2 State-Space Dual (SSD) selective scan.

The mechanism under test is the SSD core itself -- the chunked selective scan for
prefill and the single-step state update for decode -- isolated from the surrounding
convolution and gating so that it is directly comparable to the attention kernels.

Two backends are supported, in preference order:

* ``mamba-ssm``'s Triton SSD kernels (``mamba_chunk_scan_combined`` /
  ``selective_state_update``), built from source for aarch64 + SM121;
* ``flash-linear-attention``'s ``simple_gla`` kernels, which implement exactly the
  SSD recurrence (scalar per-head decay) in pure Triton.

Both are checked against a sequential PyTorch reference of the SSD recurrence before
any latency number is recorded.

The decisive property this measures: the recurrent state is
``num_heads x head_dim x d_state`` regardless of sequence length, so decode cost is
flat in S where an attention KV cache grows linearly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.metrics import BYTES_PER_MB, BenchConfig
from .base import Availability, Mechanism, PreparedPhase, VerificationResult

#: Mamba-2 default SSM state width per head.
D_STATE = 128
#: Number of B/C projection groups shared across heads (Mamba-2 default).
N_GROUPS = 1
#: Chunk length used by the SSD chunked scan.
CHUNK_SIZE = 256

_MAMBA_SSM = "mamba-ssm-triton-ssd"
_FLA = "fla-triton-simple-gla"


def _mamba_ssm_available() -> bool:
    try:
        from mamba_ssm.ops.triton.selective_state_update import (  # noqa: F401
            selective_state_update,
        )
        from mamba_ssm.ops.triton.ssd_combined import (  # noqa: F401
            mamba_chunk_scan_combined,
        )
    except (ImportError, RuntimeError, OSError):
        return False
    return True


def _fla_available() -> bool:
    try:
        from fla.ops.simple_gla import chunk_simple_gla  # noqa: F401
    except (ImportError, RuntimeError, OSError):
        return False
    return True


def ssd_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    a_log: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential float32 reference for the SSD recurrence.

    Shapes follow the ``mamba-ssm`` convention: ``x`` is ``(B, S, H, P)``, ``dt`` is
    ``(B, S, H)`` (already softplus-activated), ``a_log`` is ``(H,)`` (the negative
    ``A``), and ``b``/``c`` are ``(B, S, G, N)`` with ``G`` groups broadcast across
    heads. Returns ``(y, final_state)`` with ``final_state`` of shape ``(B, H, P, N)``.
    """
    bsz, seq, heads, p_dim = x.shape
    groups, n_dim = b.shape[2], b.shape[3]
    heads_per_group = heads // groups

    xf, dtf = x.float(), dt.float()
    bf, cf = b.float(), c.float()
    state = (
        torch.zeros(bsz, heads, p_dim, n_dim, device=x.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    decay = torch.exp(dtf * a_log.float())  # (B, S, H)
    out = torch.empty(bsz, seq, heads, p_dim, device=x.device, dtype=torch.float32)
    for t in range(seq):
        b_t = bf[:, t].repeat_interleave(heads_per_group, dim=1)  # (B, H, N)
        c_t = cf[:, t].repeat_interleave(heads_per_group, dim=1)  # (B, H, N)
        update = torch.einsum("bhp,bhn->bhpn", xf[:, t] * dtf[:, t].unsqueeze(-1), b_t)
        state = state * decay[:, t].view(bsz, heads, 1, 1) + update
        out[:, t] = torch.einsum("bhpn,bhn->bhp", state, c_t)
    if d is not None:
        out = out + xf * d.float().view(1, 1, heads, 1)
    return out.to(x.dtype), state


class MambaTwoSSD(Mechanism):
    key = "mamba2"
    name = "Mamba-2 (SSD)"

    def __init__(self, config: BenchConfig, d_state: int = D_STATE) -> None:
        super().__init__(config)
        self.d_state = d_state
        self.n_groups = N_GROUPS
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        # A is parameterized as a negative per-head scalar, as in Mamba-2.
        self.a_log = -torch.rand(self.num_heads, device=self.device, dtype=torch.float32).clamp_min(
            0.1
        )
        self.d_skip = torch.rand(self.num_heads, device=self.device, dtype=torch.float32)
        # dt is pre-activated in these microbenchmarks, so the bias is zero; it still has
        # to be materialized for the decode kernel (see `_decode_step`).
        self._dt_bias = torch.zeros(self.num_heads, device=self.device, dtype=torch.float32)

    @classmethod
    def availability(cls) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "unavailable", "CUDA device unavailable")
        if _mamba_ssm_available():
            return Availability(True, _MAMBA_SSM, "")
        if _fla_available():
            return Availability(
                True,
                _FLA,
                "mamba-ssm unavailable; using the equivalent simple_gla Triton SSD kernels",
            )
        return Availability(
            False, "unavailable", "neither mamba-ssm nor flash-linear-attention is importable"
        )

    # -- input construction -----------------------------------------------------

    def _make_inputs(self, seq_len: int) -> dict[str, torch.Tensor]:
        b = self.config.batch_size
        return {
            "x": self.randn(b, seq_len, self.num_heads, self.head_dim),
            "dt": F.softplus(
                torch.randn(b, seq_len, self.num_heads, device=self.device, dtype=torch.float32)
                - 2.0
            ),
            "B": self.randn(b, seq_len, self.n_groups, self.d_state),
            "C": self.randn(b, seq_len, self.n_groups, self.d_state),
        }

    # -- backend dispatch -------------------------------------------------------

    def _prefill_mamba_ssm(self, inp: dict[str, torch.Tensor]) -> torch.Tensor:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        return mamba_chunk_scan_combined(
            inp["x"],
            inp["dt"],
            self.a_log,
            inp["B"],
            inp["C"],
            chunk_size=CHUNK_SIZE,
            D=self.d_skip,
            dt_softplus=False,
        )

    def _prefill_fla(self, inp: dict[str, torch.Tensor]) -> torch.Tensor:
        """SSD via ``simple_gla``: q=C, k=B*dt, v=x, log-decay g = A * dt."""
        from fla.ops.simple_gla import chunk_simple_gla

        heads_per_group = self.num_heads // self.n_groups
        dt = inp["dt"]
        q = inp["C"].repeat_interleave(heads_per_group, dim=2)
        k = inp["B"].repeat_interleave(heads_per_group, dim=2) * dt.unsqueeze(-1).to(self.dtype)
        g = dt * self.a_log.view(1, 1, -1)
        out, _ = chunk_simple_gla(q, k, inp["x"], g, scale=1.0, output_final_state=True)
        return out + inp["x"] * self.d_skip.view(1, 1, -1, 1).to(self.dtype)

    def prefill_forward(self, inp: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.implementation == _MAMBA_SSM:
            return self._prefill_mamba_ssm(inp)
        return self._prefill_fla(inp)

    def _decode_step(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        dt_t: torch.Tensor,
        b_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """One recurrent state update, in place on ``state``.

        ``state`` is ``(B, H, P, N)``; ``x_t`` is ``(B, H, P)``; ``dt_t`` is ``(B, H)``;
        ``b_t``/``c_t`` are ``(B, G, N)``.
        """
        if self.implementation == _MAMBA_SSM:
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update

            heads, p_dim = self.num_heads, self.head_dim
            # Broadcast the per-head scalars across head_dim with stride-0 expands: that
            # is what selects the kernel's `tie_hdim` fast path, which is the layout
            # Mamba-2 actually uses. `dt_bias` must be a real tensor -- the kernel reads
            # its stride unconditionally, so passing None raises inside mamba-ssm.
            a_full = self.a_log.view(heads, 1, 1).expand(heads, p_dim, self.d_state)
            return selective_state_update(
                state,
                x_t,
                dt_t.unsqueeze(-1).expand(-1, -1, p_dim),
                a_full,
                b_t,
                c_t,
                D=self.d_skip.view(heads, 1).expand(heads, p_dim),
                dt_bias=self._dt_bias.view(heads, 1).expand(heads, p_dim),
                dt_softplus=False,
            )
        return self._decode_step_torch(state, x_t, dt_t, b_t, c_t)

    def _decode_step_torch(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        dt_t: torch.Tensor,
        b_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """Backend-independent single-step update, used by the fla path and as reference."""
        heads_per_group = self.num_heads // self.n_groups
        bsz = state.shape[0]
        b_h = b_t.float().repeat_interleave(heads_per_group, dim=1)
        c_h = c_t.float().repeat_interleave(heads_per_group, dim=1)
        decay = torch.exp(dt_t.float() * self.a_log.view(1, -1))
        update = torch.einsum("bhp,bhn->bhpn", x_t.float() * dt_t.float().unsqueeze(-1), b_h)
        state.mul_(decay.view(bsz, self.num_heads, 1, 1)).add_(update)
        out = torch.einsum("bhpn,bhn->bhp", state, c_h)
        return (out + x_t.float() * self.d_skip.view(1, -1, 1)).to(x_t.dtype)

    # -- correctness ------------------------------------------------------------

    def verify(self, seq_len: int = 256, tolerance: float = 2e-2) -> VerificationResult:
        """Check the chunked scan against the sequential float32 SSD recurrence."""
        inp = self._make_inputs(seq_len)
        got = self.prefill_forward(inp)
        want, _ = ssd_reference(inp["x"], inp["dt"], self.a_log, inp["B"], inp["C"], self.d_skip)
        return self.check(got, want, "ssd_reference (sequential float32 SSD recurrence)", tolerance)

    # -- phases -----------------------------------------------------------------

    def prepare_prefill(self, seq_len: int) -> PreparedPhase:
        inp = self._make_inputs(seq_len)

        def run() -> torch.Tensor:
            return self.prefill_forward(inp)

        cfg = self.config
        item = self.itemsize
        b, h, p, n = cfg.batch_size, self.num_heads, self.head_dim, self.d_state
        io_bytes = (
            2 * b * seq_len * h * p * item  # x read + y write
            + 2 * b * seq_len * self.n_groups * n * item  # B and C reads
            + b * seq_len * h * 4  # dt read (fp32)
        )
        # The chunked scan writes and re-reads one (P x N) state per head per chunk.
        num_chunks = max(1, seq_len // CHUNK_SIZE)
        chunk_state_bytes = 2 * b * num_chunks * h * p * n * 4
        state_mb = b * h * p * n * 4 / BYTES_PER_MB
        return PreparedPhase(
            fn=run,
            bytes_moved=io_bytes + chunk_state_bytes,
            tokens_processed=seq_len,
            state_memory_mb=state_mb,
            notes=f"d_state={n}, chunk={CHUNK_SIZE}, backend={self.implementation}",
        )

    def prepare_decode(self, seq_len: int) -> PreparedPhase:
        """Decode cost is independent of ``seq_len``: the state is fixed-size.

        ``seq_len`` still selects the row so results line up with the attention
        mechanisms; the recurrent state is simply carried in from a prefill of that
        length rather than a growing cache.
        """
        cfg = self.config
        b, h, p, n = cfg.batch_size, self.num_heads, self.head_dim, self.d_state
        state = torch.zeros(b, h, p, n, device=self.device, dtype=torch.float32)
        x_t = self.randn(b, h, p)
        dt_t = F.softplus(torch.randn(b, h, device=self.device, dtype=torch.float32) - 2.0)
        b_t = self.randn(b, self.n_groups, n)
        c_t = self.randn(b, self.n_groups, n)

        def run() -> torch.Tensor:
            return self._decode_step(state, x_t, dt_t, b_t, c_t)

        def reset_state() -> None:
            state.zero_()

        state_bytes = b * h * p * n * 4
        io_bytes = (
            2 * state_bytes
            + 2 * b * h * p * self.itemsize
            + 2 * b * self.n_groups * n * self.itemsize
        )
        return PreparedPhase(
            fn=run,
            bytes_moved=io_bytes,
            tokens_processed=1,
            state_memory_mb=state_bytes / BYTES_PER_MB,
            pre_measure_fn=reset_state,
            iters=cfg.decode_tokens,
            notes=(
                f"fixed {h}x{p}x{n} fp32 recurrent state, independent of S; "
                f"backend={self.implementation}"
            ),
        )
