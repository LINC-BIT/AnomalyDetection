"""The Mamba decoder side of MambaAD: `HSSBlock` (one `SS2D` selective-scan
attention + residual), `LSSModule` (the paper's "Locality-Enhanced State
Space" block -- an `HSSBlock` stack fused with two parallel depthwise-conv
branches so the decoder captures both long-range and local structure),
`LSSLayer_up` (a stage of `LSSModule`s plus an upsample), and
`MambaUPNet`, which stacks four such stages to reconstruct multi-scale
features from the fused teacher embedding.

Straight reimplementation of upstream's same-named classes; only the `SS2D`
they build on (see `ssm.py`) differs in how it computes the scan.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange

from fabric_defect_hub.models.mambaad.ssm import SS2D


class PatchExpand2D(nn.Module):
    """2x spatial upsample via a learned channel-to-space rearrange
    (pixel-shuffle-style), halving channel count.
    """

    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, _height, _width, channels = x.shape
        x = self.expand(x)
        x = rearrange(
            x, "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale, p2=self.dim_scale, c=channels // self.dim_scale,
        )
        return self.norm(x)


class HSSBlock(nn.Module):
    """One selective-scan attention block: pre-norm `SS2D` + residual +
    stochastic depth, `(B, H, W, C)` channels-last throughout.
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 4,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
            size=size, scan_type=scan_type, num_direction=num_direction,
        )
        # DropPath (stochastic depth): identity at drop_path=0, else drops
        # the whole residual branch per-sample with that probability.
        self.drop_path_rate = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._drop_path(self.self_attention(self.norm(x)))


class LSSModule(nn.Module):
    """Locality-Enhanced State Space block: an `HSSBlock` stack (global,
    long-range structure via the multi-directional scan) fused with two
    parallel depthwise 5x5/7x7 conv branches (local structure), concatenated
    and mixed back down to `hidden_dim` with a residual connection.
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int = 2,
        drop_path: float | list[float] = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 8,
    ):
        super().__init__()
        drop_paths = drop_path if isinstance(drop_path, list) else [drop_path] * depth
        self.attn_blocks = nn.ModuleList(
            [
                HSSBlock(
                    hidden_dim=hidden_dim, drop_path=drop_paths[i], norm_layer=norm_layer,
                    attn_drop_rate=attn_drop_rate, d_state=d_state, size=size,
                    scan_type=scan_type, num_direction=num_direction,
                )
                for i in range(depth)
            ]
        )

        def local_branch(kernel_size: int) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.InstanceNorm2d(hidden_dim), nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding,
                          groups=hidden_dim, bias=False),
                nn.InstanceNorm2d(hidden_dim), nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.InstanceNorm2d(hidden_dim), nn.SiLU(),
            )

        self.local_5x5 = local_branch(5)
        self.local_7x7 = local_branch(7)
        self.merge = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1)
        self.apply(self._init_conv_weights)

    @staticmethod
    def _init_conv_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            import math

            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, H, W, C)` channels-last, matching `HSSBlock`."""

        global_features = x
        for block in self.attn_blocks:
            global_features = block(global_features)

        spatial = x.permute(0, 3, 1, 2).contiguous()
        local_5 = self.local_5x5(spatial)
        local_7 = self.local_7x7(spatial)

        merged = torch.cat([global_features.permute(0, 3, 1, 2).contiguous(), local_5, local_7], dim=1)
        merged = self.merge(merged).permute(0, 2, 3, 1).contiguous()
        return merged + x


class LSSLayer_up(nn.Module):
    """One decoder stage: an optional 2x upsample followed by
    `depth // {2,3}` `LSSModule`s (each itself `depth`-deep in `HSSBlock`s
    -- upstream groups 3 attention blocks per `LSSModule` when `depth` is a
    multiple of 3, else 2; kept as-is rather than simplified since it is
    what the paper's published depth schedules assume).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: Callable = nn.LayerNorm,
        upsample: type[nn.Module] | None = None,
        d_state: int = 16,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 4,
    ):
        super().__init__()
        module_depth = 3 if depth % 3 == 0 else 2
        num_modules = depth // module_depth
        drop_paths = drop_path if isinstance(drop_path, list) else [drop_path] * num_modules

        self.blocks = nn.ModuleList(
            [
                LSSModule(
                    hidden_dim=dim, depth=module_depth, drop_path=drop_paths[i], norm_layer=norm_layer,
                    attn_drop_rate=attn_drop, d_state=d_state, size=size,
                    scan_type=scan_type, num_direction=num_direction,
                )
                for i in range(num_modules)
            ]
        )
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class MambaUPNet(nn.Module):
    """Four-stage Mamba decoder: from the fused teacher embedding
    (`dims_decoder[0]` channels, deepest/smallest spatial resolution) up
    to `dims_decoder[-1]` channels at 8x the spatial size, returning the
    three shallower stages' outputs (channels-first) -- the scales the
    student is compared against the teacher's own three feature levels.
    """

    def __init__(
        self,
        dims_decoder: list[int],
        depths_decoder: list[int],
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        base_size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 4,
    ):
        super().__init__()
        total_depth = sum(depths_decoder)
        drop_path_schedule = torch.linspace(0, drop_path_rate, total_depth).tolist()[::-1]

        self.layers_up = nn.ModuleList()
        depth_offset = 0
        for stage, (dim, depth) in enumerate(zip(dims_decoder, depths_decoder)):
            layer = LSSLayer_up(
                dim=dim, depth=depth, d_state=d_state,
                drop_path=drop_path_schedule[depth_offset : depth_offset + depth],
                upsample=PatchExpand2D if stage != 0 else None,
                size=base_size * (2**stage),
                scan_type=scan_type, num_direction=num_direction,
            )
            self.layers_up.append(layer)
            depth_offset += depth
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        from timm.layers import trunc_normal_

        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """`x`: `(B, C, H, W)`. Returns the shallower 3 stages' outputs,
        each `(B, C_i, H_i, W_i)`, deepest-first-discarded (stage 0's
        output only upsamples into stage 1, never compared to the teacher).
        """

        x = rearrange(x, "b c h w -> b h w c")
        outputs = []
        for stage, layer in enumerate(self.layers_up):
            x = layer(x)
            if stage != 0:
                outputs.insert(0, rearrange(x, "b h w c -> b c h w"))
        return outputs
