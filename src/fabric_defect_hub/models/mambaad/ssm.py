"""The selective-scan recurrence and `SS2D` (2D selective-scan attention).

MambaAD's scan core is `mamba_ssm`'s fused CUDA kernel (`selective_scan_fn`),
which only builds against a matching CUDA/nvcc toolchain -- it does not
install at all on this project's dev machine (Apple Silicon). Depending on
it unconditionally would gate the whole backend to a CUDA host, which is
one of the two reasons MambaAD was dropped as a vendored component before
(see `components/README.md`). So `selective_scan` dispatches across three
implementations of the *same* recurrence:

1. `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` -- used
   automatically whenever the package imports and the tensors are on CUDA.
   This is upstream's exact kernel: on the training box you get upstream's
   exact semantics *and* its speed, with no configuration.
2. `selective_scan_chunked` -- the portable fallback (CPU/MPS, or CUDA
   without `mamba_ssm`). Mathematically identical to (3), but evaluates
   each chunk's intra-chunk contributions as one batched matmul instead of
   one Python-level step per timestep, cutting the sequential step count by
   the chunk length. That matters enormously: the deepest decoder stage
   scans L=4096 tokens 8 ways per block, and at one Python step per token
   a single 256px forward pass takes ~4s on CPU (measured) -- days per
   training run. See `_chunk_length` for how the chunk size is picked.
3. `selective_scan_reference` -- the plain sequential loop, i.e. the
   textbook statement of the recurrence (Gu & Dao, 2023), matching
   `mamba_ssm`'s own `selective_scan_ref`. Kept as the oracle the chunked
   path is tested against (`tests/test_mambaad.py`), not for production use.

All three take and return the same shapes, so which one ran is invisible to
the rest of the model -- only the wall-clock differs.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from fabric_defect_hub.models.mambaad.scan import scan_order

# Peak extra memory the chunked scan's decay matrix may take, in floats
# (~256MB at 4 bytes each). The chunk length is derived from this and the
# actual tensor shape rather than fixed, because the decay matrix is
# (batch, d_inner, d_state, chunk, chunk) -- a chunk length that is
# comfortable for the deepest decoder stage (small d_inner, huge length)
# would blow up on the shallowest (large d_inner, tiny length).
_CHUNK_BUDGET_ELEMENTS = int(os.environ.get("FDH_MAMBAAD_SCAN_BUDGET", 64_000_000))
_MAX_CHUNK = 128


def _mamba_ssm_scan_fn():
    """`mamba_ssm`'s fused kernel if importable, else None (cached)."""

    if not hasattr(_mamba_ssm_scan_fn, "_cached"):
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            _mamba_ssm_scan_fn._cached = selective_scan_fn
        except Exception:
            _mamba_ssm_scan_fn._cached = None
    return _mamba_ssm_scan_fn._cached


def _prepare(u, delta, A, B, C, delta_bias, delta_softplus):
    """Coerce to float32, apply delta_bias/softplus, and normalize B/C to a
    grouped `(batch, groups, d_state, length)` layout.

    B/C may arrive either ungrouped -- `(batch, d_state, length)`, one
    shared state projection for every channel, the plain single-scan case --
    or grouped, `(batch, groups, d_state, length)`, where channel `d`
    belongs to group `d // (d_inner // groups)`. Grouping is what lets all
    of `SS2D`'s scan directions run as *one* call: each direction has its
    own B/C but they can share one pass over the sequence (see
    `SS2D.forward_core`).
    """

    u, delta, A, B, C = (t.float() for t in (u, delta, A, B, C))
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    if B.dim() == 3:
        B, C = B.unsqueeze(1), C.unsqueeze(1)
    d_inner = u.shape[1]
    groups = B.shape[1]
    if d_inner % groups:
        raise ValueError(f"d_inner={d_inner} must be divisible by the B/C group count {groups}.")
    return u, delta, A, B, C, groups


def _scan_loop_eager(deltaA: torch.Tensor, deltaB_u: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """The sequential recurrence.

    deltaA, deltaB_u: (batch, groups, per_group, length, d_state).
    C: (batch, groups, d_state, length). Returns (batch, groups, per_group, length).
    """

    batch, groups, per_group, length, d_state = deltaA.shape
    state = torch.zeros((batch, groups, per_group, d_state), dtype=deltaA.dtype, device=deltaA.device)
    outputs = []
    for step in range(length):
        state = deltaA[:, :, :, step] * state + deltaB_u[:, :, :, step]
        outputs.append(torch.einsum("bgpn,bgn->bgp", state, C[:, :, :, step]))
    return torch.stack(outputs, dim=-1)


def _compile_scan_loop():
    """TorchScript-compile the step loop when that is actually available.

    The loop is latency-bound on tiny per-step tensors, so removing Python
    dispatch overhead is worth ~1.2x. But `torch.jit.script` is deprecated
    on Python 3.14+ (it warns that it "may break"), and a ~1.2x speedup is
    not worth a hard dependency on a deprecated path -- so compilation is
    attempted once and silently falls back to the eager loop, which is the
    same function, just slower.
    """

    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.jit.script(_scan_loop_eager)
    except Exception:
        return _scan_loop_eager


_scan_loop = _compile_scan_loop()


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """The recurrence stated directly: one step per timestep.

    u, delta: (batch, d_inner, length). A: (d_inner, d_state).
    B, C: (batch, d_state, length) or (batch, groups, d_state, length).
    D: (d_inner,) or None. Returns: (batch, d_inner, length).

    The step loop itself is TorchScript-compiled (`_scan_loop`); this is
    still the O(L)-sequential formulation and the semantic oracle the
    chunked path is tested against.
    """

    dtype_in = u.dtype
    u, delta, A, B, C, groups = _prepare(u, delta, A, B, C, delta_bias, delta_softplus)
    batch, d_inner, length = u.shape
    d_state = A.shape[1]
    per_group = d_inner // groups

    # Discretize: deltaA_t = exp(delta_t * A), deltaB_u_t = delta_t * B_t * u_t.
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bgnl,bdl->bdln", delta, B, u) if groups == 1 else (
        delta.view(batch, groups, per_group, length, 1)
        * B.permute(0, 1, 3, 2).unsqueeze(2)
        * u.view(batch, groups, per_group, length, 1)
    )
    deltaA = deltaA.view(batch, groups, per_group, length, d_state)
    deltaB_u = deltaB_u.view(batch, groups, per_group, length, d_state)

    y = _scan_loop(deltaA, deltaB_u, C).reshape(batch, d_inner, length)
    if D is not None:
        y = y + u * D[..., None]
    return y.to(dtype=dtype_in)


def _chunk_length(batch: int, d_inner: int, d_state: int, length: int) -> int:
    """Largest chunk whose decay matrix fits `_CHUNK_BUDGET_ELEMENTS`."""

    per_step = batch * d_inner * d_state
    if per_step <= 0:
        return length
    max_by_memory = int(math.sqrt(max(_CHUNK_BUDGET_ELEMENTS / per_step, 1.0)))
    return max(1, min(length, _MAX_CHUNK, max_by_memory))


def selective_scan_chunked(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Blocked evaluation of the same recurrence: sequential across chunks,
    fully parallel within one.

    Unrolling `h[t] = a[t] * h[t-1] + bu[t]` inside a chunk gives

        h[t] = exp(cs[t]) * h_chunk_start + sum_{s <= t} exp(cs[t] - cs[s]) * bu[s]

    where `cs = cumsum(log a)` within the chunk, so the whole chunk is one
    masked matmul against the `(chunk, chunk)` decay matrix
    `exp(cs[t] - cs[s])`. That matrix is built from *differences* of a
    decreasing cumulative sum, so every entry is `exp(<= 0) <= 1` -- no
    overflow, which is what makes this exact rather than merely close (the
    tempting `exp(cs[t]) * cumsum(exp(-cs[s]) * bu[s])` one-liner is O(L)
    memory but overflows, since `exp(-cs[s])` grows without bound).

    `chunk_size=None` derives it from a memory budget (see `_chunk_length`).
    """

    dtype_in = u.dtype
    u, delta, A, B, C, groups = _prepare(u, delta, A, B, C, delta_bias, delta_softplus)
    batch, d_inner, length = u.shape
    d_state = A.shape[1]
    per_group = d_inner // groups

    log_a = torch.einsum("bdl,dn->bdnl", delta, A).view(batch, groups, per_group, d_state, length)
    bu = (
        delta.view(batch, groups, per_group, 1, length)
        * B.unsqueeze(2)
        * u.view(batch, groups, per_group, 1, length)
    )

    step = chunk_size or _chunk_length(batch, d_inner, d_state, length)
    state = log_a.new_zeros((batch, groups, per_group, d_state))

    def one_chunk(log_a_c, bu_c, C_c, state_in):
        """One chunk's contribution plus the state handed to the next."""

        cumulative = log_a_c.cumsum(dim=-1)  # (B, G, P, N, Lc)
        span = log_a_c.shape[-1]

        # decay[..., t, s] = exp(cs[t] - cs[s]), lower-triangular (s <= t).
        differences = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
        causal = torch.ones(span, span, dtype=torch.bool, device=log_a_c.device).tril()
        decay = torch.where(causal, differences.exp(), differences.new_zeros(()))

        within = (decay * bu_c.unsqueeze(-2)).sum(dim=-1)
        states = within + cumulative.exp() * state_in.unsqueeze(-1)  # (B, G, P, N, Lc)
        return torch.einsum("bgpnl,bgnl->bgpl", states, C_c), states[..., -1]

    # The `(B, G, P, N, Lc, Lc)` decay matrix is budgeted *per chunk*
    # (`_chunk_length`), but autograd would keep every chunk's copy alive
    # until backward -- turning a ~256MB working set into 256MB x
    # (length / step), which is tens of GB for a detection-sized feature map
    # and OOMs a 32GB card on the first iteration. Checkpointing recomputes
    # each chunk during backward instead, so peak memory is one chunk's
    # worth regardless of sequence length, at the cost of one extra forward.
    # Only when building a graph: under `no_grad` nothing is retained
    # anyway, and recomputation would be pure overhead.
    use_checkpoint = torch.is_grad_enabled() and length > step

    chunk_outputs = []
    for start in range(0, length, step):
        stop = min(start + step, length)
        args = (log_a[..., start:stop], bu[..., start:stop], C[..., start:stop], state)
        if use_checkpoint:
            chunk_y, state = torch.utils.checkpoint.checkpoint(one_chunk, *args, use_reentrant=False)
        else:
            chunk_y, state = one_chunk(*args)
        chunk_outputs.append(chunk_y)

    y = torch.cat(chunk_outputs, dim=-1).reshape(batch, d_inner, length)
    if D is not None:
        y = y + u * D[..., None]
    return y.to(dtype=dtype_in)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """Dispatch to the best implementation for the device in hand.

    * **CUDA + `mamba_ssm`** -> upstream's fused kernel. Exact upstream
      semantics and speed, picked up automatically with no configuration --
      this is what a real training run on a GPU box uses.
    * **CUDA without `mamba_ssm`** -> the chunked scan. A GPU has FLOPs to
      spare but pays a kernel launch per sequential step, so trading extra
      arithmetic for `chunk_size`x fewer launches is the right side of that
      tradeoff.
    * **CPU / MPS** -> the sequential scan. Measured: chunking is *slower*
      here, because without launch overhead to amortize, its extra
      arithmetic is pure cost (see `tests/test_mambaad.py` and the module
      docstring).
    """

    if u.is_cuda:
        fused = _mamba_ssm_scan_fn()
        if fused is not None and B.dim() == 4:
            # The fused kernel takes grouped B/C as (batch, groups, d_state,
            # length) -- exactly the layout SS2D builds -- and applies
            # delta_bias/softplus itself, the same call upstream makes.
            return fused(
                u, delta, A, B, C, D, z=None,
                delta_bias=delta_bias, delta_softplus=delta_softplus,
                return_last_state=False,
            )
        return selective_scan_chunked(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=delta_softplus
        )
    return selective_scan_reference(
        u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=delta_softplus
    )


class HSCANS(nn.Module):
    """Precomputed (encode, decode) index permutations for one scan order
    over an `size x size` grid -- see `scan.py`. A `nn.Module` (not a plain
    function) purely so the index tensors move with `.to(device)`/state_dict
    like upstream's, even though they hold no trainable weights.
    """

    def __init__(self, size: int, scan_type: str = "scan"):
        super().__init__()
        locs_flat = scan_order(int(size), scan_type)
        locs_flat_inv = np.argsort(locs_flat)
        self.index_flat = nn.Parameter(
            torch.as_tensor(locs_flat, dtype=torch.long).unsqueeze(0).unsqueeze(1), requires_grad=False
        )
        self.index_flat_inv = nn.Parameter(
            torch.as_tensor(locs_flat_inv, dtype=torch.long).unsqueeze(0).unsqueeze(1), requires_grad=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Grid order -> scan order (scatter each cell to its scan step)."""

        return torch.zeros_like(x).scatter_(2, self.index_flat_inv.expand(x.shape), x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Scan order -> grid order (inverse of `encode`)."""

        return torch.zeros_like(x).scatter_(2, self.index_flat.expand(x.shape), x)


class SS2D(nn.Module):
    """2D selective-scan self-attention: project to `d_inner`, depthwise
    conv, scan the `H x W` grid along `num_direction` (scan order, its
    reverse, and — for `num_direction >= 4`/`8` — the transpose and
    90-degree rotation, each forward and reversed) 1D orderings, run the
    selective-scan recurrence independently per direction, sum the
    decoded-back-to-grid results, then gate and project out.

    A straight reimplementation of upstream's `SS2D`, with `selective_scan`
    (above) standing in for `mamba_ssm`'s fused kernel and `HSCANS`
    (above) standing in for its `pyzorder`/`hilbert`-backed one.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 8,
    ):
        super().__init__()
        if num_direction not in (2, 4, 8):
            raise ValueError(f"num_direction must be 2, 4 or 8, got {num_direction}")
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.num_direction = num_direction

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2,
            groups=self.d_inner, bias=conv_bias,
        )
        self.act = nn.SiLU()

        x_proj_weight = [
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False).weight
            for _ in range(num_direction)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))

        dt_projs = [
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(num_direction)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dt_projs], dim=0))

        self.A_logs = self._A_log_init(d_state, self.d_inner, copies=num_direction)
        self.Ds = self._D_init(self.d_inner, copies=num_direction)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.scans = HSCANS(size=size, scan_type=scan_type)

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"unknown dt_init {dt_init!r}")
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^-1
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def _A_log_init(d_state: int, d_inner: int, copies: int) -> nn.Parameter:
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        A_log = torch.log(A).unsqueeze(0).repeat(copies, 1, 1).flatten(0, 1)
        return nn.Parameter(A_log)

    @staticmethod
    def _D_init(d_inner: int, copies: int) -> nn.Parameter:
        return nn.Parameter(torch.ones(copies, d_inner).flatten(0, 1))

    def _multi_directional_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Build the `num_direction` 1D views of the (B, d_inner, H, W)
        grid: scan-order, its reverse, and — from `num_direction=4` up —
        the transpose, and — from `8` up — the 90-degree rotation and its
        transpose, each contributing a forward and a reversed sequence.
        """

        batch, _channels, height, width = x.shape
        length = height * width
        views = [self.scans.encode(x.view(batch, -1, length))]
        if self.num_direction >= 4:
            views.append(self.scans.encode(x.transpose(2, 3).contiguous().view(batch, -1, length)))
        if self.num_direction >= 8:
            rotated = torch.rot90(x, k=1, dims=(2, 3))
            views.append(self.scans.encode(rotated.contiguous().view(batch, -1, length)))
            views.append(self.scans.encode(rotated.transpose(2, 3).contiguous().view(batch, -1, length)))
        stacked = torch.stack(views, dim=1).view(batch, self.num_direction // 2, -1, length)
        return torch.cat([stacked, torch.flip(stacked, dims=[-1])], dim=1)

    def _undo_multi_directional_scan(self, y: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch = y.shape[0]
        length = height * width
        k = self.num_direction
        forward_reversed = torch.flip(y[:, k // 2 : k], dims=[-1]).view(batch, k // 2, -1, length)

        parts = [self.scans.decode(y[:, 0]), self.scans.decode(forward_reversed[:, 0])]
        if k >= 4:
            parts.append(
                self.scans.decode(y[:, 1]).view(batch, -1, width, height).transpose(2, 3).contiguous().view(batch, -1, length)
            )
            parts.append(
                self.scans.decode(forward_reversed[:, 1]).view(batch, -1, width, height)
                .transpose(2, 3).contiguous().view(batch, -1, length)
            )
        if k >= 8:
            def undo_rotation(decoded: torch.Tensor) -> torch.Tensor:
                return torch.rot90(decoded.view(batch, -1, width, height), k=3, dims=(2, 3)).contiguous().view(batch, -1, length)

            def undo_rotation_transpose(decoded: torch.Tensor) -> torch.Tensor:
                grid_ = decoded.view(batch, -1, width, height).transpose(2, 3)
                return torch.rot90(grid_, k=3, dims=(2, 3)).contiguous().view(batch, -1, length)

            parts.append(undo_rotation(self.scans.decode(y[:, 2])))
            parts.append(undo_rotation(self.scans.decode(forward_reversed[:, 2])))
            parts.append(undo_rotation_transpose(self.scans.decode(y[:, 3])))
            parts.append(undo_rotation_transpose(self.scans.decode(forward_reversed[:, 3])))
        return sum(parts)

    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = x.shape
        length = height * width
        k = self.num_direction

        xs = self._multi_directional_scan(x)  # (batch, k, d_inner, length)

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs.view(batch, k, -1, length), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts.view(batch, k, -1, length), self.dt_projs_weight)

        # All `k` directions run as ONE scan, with direction folded into the
        # channel axis (k * d_inner) and each direction's own B_t/C_t passed
        # as its own group. The recurrence is sequential in `length`, so the
        # cost is dominated by the number of steps, not their width: one
        # k-times-wider pass is far cheaper than k passes (measured ~3x on
        # CPU), and it is also exactly the layout `mamba_ssm`'s fused kernel
        # expects, so the CUDA fast path needs no reshaping either.
        As = -torch.exp(self.A_logs.float())  # (k * d_inner, d_state)
        out = selective_scan(
            xs.reshape(batch, k * self.d_inner, length),
            dts.reshape(batch, k * self.d_inner, length),
            As,
            Bs.float().view(batch, k, self.d_state, length),
            Cs.float().view(batch, k, self.d_state, length),
            self.Ds.float(),
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
        ).view(batch, k, self.d_inner, length)
        return self._undo_multi_directional_scan(out, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _channels = x.shape
        xz = self.in_proj(x)
        x_in, gate = xz.chunk(2, dim=-1)
        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_in = self.act(self.conv2d(x_in))
        y = self.forward_core(x_in)
        y = y.transpose(1, 2).contiguous().view(batch, height, width, -1)
        y = self.out_norm(y)
        y = y * F.silu(gate)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
