"""Clean-room building blocks for FabricMamba (Bao et al., 2025,
doi:10.1016/j.engappai.2025.112558): P-LSKA, MVSS, and DySample, written to
plug into Ultralytics' YOLOv8 graph via `register_with_ultralytics`.

The paper ships no code (its ESQUEL dataset and implementation are
proprietary), so everything here is reimplemented from the paper's text and
pseudocode tables:

* `LSKA` / `PLSKA` follow Table 1's pseudocode line by line (kernel sizes,
  dilation, pooling cascade, the two parallel branch inputs).
* `MVSS` / `MVSSBlock` follow Table 2 and Fig. 4: a C2f-shaped wrapper whose
  inner blocks are LayerNorm -> SS2D -> DropPath residual -> LayerNorm ->
  MLP -> DropPath residual. The SS2D core is reused from this project's
  MambaAD reimplementation (`models/mambaad/ssm.py`), subclassed to the
  four-direction raster cross-scan (VMamba-style) the paper describes,
  with dynamic feature-map sizes instead of MambaAD's fixed grids.
* `DySample` implements the point-sampling upsampler of Liu et al. (ICCV
  2023, arXiv:2308.15085) in its simplest published form ("lp" grouping,
  static scope), which is all the FabricMamba paper specifies.

Where the paper is silent on a hyperparameter (SS2D state size, MLP ratio,
DropPath rate), the value is chosen here and named in the constructor
signature rather than hidden — see each class's docstring. These are
assumptions, not reproductions; revisit against upstream if the authors
ever release code.

Every module is deliberately channel-preserving (`c_out == c_in`), because
Ultralytics' `parse_model` routes unknown modules through its generic
branch, which passes YAML args verbatim and assumes the output width equals
the input width. Channel changes in the FabricMamba head are done by
standard `Conv` layers in the YAML instead (see `fabricmamba_n.yaml`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fabric_defect_hub.models.mambaad.ssm import SS2D


class LSKA(nn.Module):
    """Large Separable Kernel Attention, per Table 1's `LSKA` function:
    depthwise 1x3 + 3x1, then dilated (rate 2) depthwise 1x5 + 5x1, then a
    1x1 conv producing the attention map that gates the input.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv_h_dilated = nn.Conv2d(dim, dim, (1, 5), padding=(0, 4), dilation=2, groups=dim)
        self.conv_v_dilated = nn.Conv2d(dim, dim, (5, 1), padding=(4, 0), dilation=2, groups=dim)
        self.conv_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv_h(x)
        attn = self.conv_v(attn)
        attn = self.conv_h_dilated(attn)
        attn = self.conv_v_dilated(attn)
        attn = self.conv_out(attn)
        return x * attn


class PLSKA(nn.Module):
    """Parallel LSKA: FabricMamba's replacement for YOLOv8's SPPF, per
    Table 1. A 1x1 conv halves the channels, a cascade of three 5x5 max
    pools builds the multi-scale pyramid, and two LSKA branches attend over
    two different four-way concatenations of that pyramid before a 1x1 conv
    fuses both branches back to the input width.

    Channel-preserving on purpose (see module docstring): `c` is both the
    input and output width, exactly like SPPF's use in YOLOv8 where the
    bottleneck stage keeps its width through the pooling block.
    """

    def __init__(self, c: int, pool_kernel: int = 5):
        super().__init__()
        hidden = c // 2
        self.cv1 = nn.Conv2d(c, hidden, 1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
        self.lska1 = LSKA(hidden * 4)
        self.lska2 = LSKA(hidden * 4)
        self.cv2 = nn.Conv2d(hidden * 8, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.cv1(x)
        y1 = self.pool(xp)
        y2 = self.pool(y1)
        out1 = self.lska1(torch.cat((xp, y1, y2, self.pool(y2)), dim=1))
        out2 = self.lska2(torch.cat((xp, y1, y2, self.pool(xp)), dim=1))
        return self.cv2(torch.cat((out1, out2), dim=1))


class CrossScanSS2D(SS2D):
    """MambaAD's `SS2D` with its fixed-size, precomputed scan orders
    replaced by the plain four-direction raster cross-scan (row-major and
    column-major, each forward and reversed) that VMamba — and Fig. 4's
    SS2D — use. Works at any feature-map size, which the detection FPN
    needs (P3/P4/P5 differ, and inference size is not fixed).

    Only the scan-view construction is overridden; parameter layout,
    initialisation, and the selective-scan recurrence (with its CUDA fast
    path) are inherited unchanged.
    """

    def __init__(self, d_model: int, **kwargs):
        kwargs.setdefault("num_direction", 4)
        # The parent builds `HSCANS` index tables for a fixed grid; they are
        # never used here (both scan hooks are overridden), so keep them at
        # the smallest legal size instead of the parent's default.
        kwargs.setdefault("size", 2)
        if kwargs["num_direction"] != 4:
            raise ValueError("CrossScanSS2D implements exactly the 4-direction raster cross-scan")
        super().__init__(d_model, **kwargs)

    def _multi_directional_scan(self, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = x.shape
        length = height * width
        rows = x.view(batch, -1, length)
        cols = x.transpose(2, 3).contiguous().view(batch, -1, length)
        stacked = torch.stack((rows, cols), dim=1)  # (batch, 2, d_inner, length)
        return torch.cat((stacked, torch.flip(stacked, dims=[-1])), dim=1)

    def _undo_multi_directional_scan(self, y: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch = y.shape[0]
        length = height * width
        reversed_back = torch.flip(y[:, 2:4], dims=[-1])

        def undo_transpose(seq: torch.Tensor) -> torch.Tensor:
            return seq.view(batch, -1, width, height).transpose(2, 3).contiguous().view(batch, -1, length)

        return (
            y[:, 0]
            + undo_transpose(y[:, 1])
            + reversed_back[:, 0]
            + undo_transpose(reversed_back[:, 1])
        )


class DropPath(nn.Module):
    """Stochastic depth (Huang et al., 2016): drop the whole residual branch
    per sample with probability `p` during training. Same semantics as
    `timm.layers.DropPath`, inlined to keep this backend's imports to torch.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(mask_shape, dtype=x.dtype, device=x.device) < keep
        return x * mask / keep


class MVSSBlock(nn.Module):
    """One block of Fig. 4 / Table 2's `MVSS Block` function: NHWC layout,
    LayerNorm -> SS2D -> DropPath residual, then LayerNorm -> MLP ->
    DropPath residual.

    The paper does not state the SS2D state size, the MLP expansion, or the
    DropPath rate; `d_state=16` and `mlp_ratio=2.0` follow VMamba's small
    configurations, `drop_path=0.0` because FabricMamba's training recipe
    never mentions stochastic depth.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        ssm_expand: int = 1,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = CrossScanSS2D(dim, d_state=d_state, expand=ssm_expand)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, channels, H, W) -> NHWC for LayerNorm/SS2D, back at the end.
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + self.drop_path(self.ss2d(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2).contiguous()


class MVSS(nn.Module):
    """FabricMamba's replacement for the FPN's C2f blocks, per Table 2: a
    1x1 conv to `2 * hidden` channels, a two-way split, a cascade of
    `n` MVSS blocks each feeding the next (every intermediate kept), and a
    1x1 fuse conv over the concatenation — C2f's gradient-rich wiring with
    state-space blocks inside.

    Channel-preserving (`c` in, `c` out); the YAML pairs it with a plain
    `Conv` wherever the original C2f also changed widths.
    """

    def __init__(
        self,
        c: int,
        n: int = 1,
        d_state: int = 16,
        ssm_expand: int = 1,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = c // 2
        self.cv1 = nn.Conv2d(c, hidden * 2, 1)
        self.blocks = nn.ModuleList(
            MVSSBlock(hidden, d_state=d_state, ssm_expand=ssm_expand, mlp_ratio=mlp_ratio, drop_path=drop_path)
            for _ in range(n)
        )
        self.cv2 = nn.Conv2d(hidden * (2 + n), c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class DySample(nn.Module):
    """Dynamic upsampling by point sampling (Liu et al., ICCV 2023), in the
    paper's simplest "lp" configuration with a static scope factor: a 1x1
    conv predicts per-group sampling offsets at the low resolution, they are
    pixel-shuffled up to the target resolution, added to the regular
    upsampling grid, and the input is bilinearly point-sampled there.

    Replaces `nn.Upsample(scale_factor=2, mode="nearest")` in the FPN;
    channel-preserving like the module it replaces.
    """

    _SCOPE = 0.25  # static offset scope, the DySample paper's default

    def __init__(self, c: int, scale: int = 2, groups: int = 4):
        super().__init__()
        if c % groups:
            raise ValueError(f"channels ({c}) must divide evenly into groups ({groups})")
        self.scale = int(scale)
        self.groups = int(groups)
        self.offset = nn.Conv2d(c, 2 * groups * self.scale**2, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        out_h, out_w = height * self.scale, width * self.scale

        # Offsets in input-pixel units: (batch, 2 * groups, out_h, out_w).
        offset = F.pixel_shuffle(self.offset(x) * self._SCOPE, self.scale)
        offset = offset.view(batch * self.groups, 2, out_h, out_w)

        # The regular bilinear-upsampling sample positions, in input-pixel
        # coordinates, that a zero offset must reproduce exactly.
        device, dtype = x.device, x.dtype
        ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) / self.scale - 0.5
        xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) / self.scale - 0.5
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")

        pos_x = base_x.unsqueeze(0) + offset[:, 0]
        pos_y = base_y.unsqueeze(0) + offset[:, 1]
        # Normalise to grid_sample's [-1, 1] coordinate convention.
        grid = torch.stack(
            (
                2.0 * pos_x / max(width - 1, 1) - 1.0,
                2.0 * pos_y / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        )

        grouped = x.view(batch * self.groups, channels // self.groups, height, width)
        sampled = F.grid_sample(grouped, grid, mode="bilinear", align_corners=True, padding_mode="border")
        return sampled.view(batch, channels, out_h, out_w)


#: The classes the architecture YAML refers to by bare name.
ULTRALYTICS_MODULES: tuple[type[nn.Module], ...] = (PLSKA, MVSS, DySample)


def register_with_ultralytics() -> None:
    """Make the FabricMamba modules resolvable by Ultralytics.

    `parse_model` looks module names up in `ultralytics.nn.tasks`'s module
    globals, so injecting the classes there lets `fabricmamba_n.yaml` name
    them like any built-in. Idempotent; refuses to shadow a genuine
    Ultralytics attribute of the same name so an upstream release that
    gains e.g. its own `DySample` fails loudly here instead of silently
    training a different network.
    """

    import ultralytics.nn.tasks as tasks

    for cls in ULTRALYTICS_MODULES:
        existing = getattr(tasks, cls.__name__, None)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"ultralytics.nn.tasks already defines {cls.__name__!r}; "
                "refusing to shadow it — rename the FabricMamba module."
            )
        setattr(tasks, cls.__name__, cls)

    # Trained checkpoints pickle these classes by reference; torch >= 2.6
    # only unpickles allowlisted globals under `weights_only=True` loads.
    try:
        torch.serialization.add_safe_globals(list(ULTRALYTICS_MODULES))
    except AttributeError:  # older torch without the allowlist API
        pass
