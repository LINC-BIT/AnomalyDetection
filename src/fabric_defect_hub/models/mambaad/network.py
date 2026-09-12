"""Teacher-fusion-decoder wiring: `MultiScaleFusion` (upstream's `MFF_OCE`)
compresses the frozen teacher's three feature levels into one embedding at
the deepest level's resolution, `MambaADNet` ties a frozen `timm`
`features_only` teacher, that fusion block, and `MambaUPNet` (see
`decoder.py`) together into one reverse-distillation model: a frozen
teacher's multi-scale features are the training target, the fused
embedding is what the (trainable) decoder has to reconstruct them from.

Same "reverse distillation" shape as `models/dinomaly/adapter.py`'s
`ViTill` (frozen encoder, trainable bottleneck + decoder, multi-scale
cosine-loss target) -- MambaAD is a CNN-teacher, Mamba-decoder instance of
the same family, not architecturally novel outside its Mamba decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.resnet import Bottleneck

from fabric_defect_hub.models.mambaad.decoder import MambaUPNet


def _resnet_bottleneck_stage(
    in_channels: int, planes: int, blocks: int, stride: int, norm_layer: type[nn.Module]
) -> nn.Sequential:
    """One stage of `timm`'s `Bottleneck` blocks (expansion=4), matching
    `torchvision.models.resnet._make_layer`'s shape -- used both as
    `MultiScaleFusion`'s output stage and, implicitly, as what a
    ResNet/wide-ResNet teacher's own stages look like, so the fused
    embedding's channel width lines up with `net_t`'s deepest level.
    """

    downsample = None
    out_channels = planes * Bottleneck.expansion
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            norm_layer(out_channels),
        )
    layers = [Bottleneck(in_channels, planes, stride=stride, downsample=downsample, norm_layer=norm_layer)]
    layers += [Bottleneck(out_channels, planes, norm_layer=norm_layer) for _ in range(1, blocks)]
    return nn.Sequential(*layers)


class MultiScaleFusion(nn.Module):
    """Fuses a teacher's 3 feature levels (`[level1, level2, level3]`, at
    decreasing spatial resolution -- a `timm` `features_only` backbone's
    `out_indices=[1, 2, 3]`) into one embedding at level 3's resolution
    (downsampled once more), then runs it through one more Bottleneck
    stage so the decoder's input has already mixed information across
    scales before reconstruction starts.

    Reimplementation of upstream's `MFF_OCE` (omni-scale context
    extraction): downsample level1 to level2's resolution and add,
    downsample that to level3's resolution and add, project, then a
    `Bottleneck` stage. One deliberate change from upstream: `MFF_OCE`
    hardcodes its channel widths as `16/32/64 * Bottleneck.expansion`
    (64/128/256), which only equals a real teacher's own channel counts by
    a coincidence of arithmetic for the BasicBlock-ResNet family it ships
    configs for (resnet34's layer1/2/3 are exactly 64/128/256) and silently
    stops matching for anything else (e.g. a Bottleneck-based
    wide_resnet50_2 teacher, whose layers are 256/512/1024). Taking
    `teacher_channels` explicitly instead removes that assumption; it
    works for whatever backbone `presets.py` names, not only the one
    upstream's hardcoding happens to fit.
    """

    def __init__(
        self,
        teacher_channels: list[int],
        out_channels: int | None = None,
        fusion_blocks: int = 3,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        c1, c2, c3 = teacher_channels
        # Default mirrors upstream's own shape (its hardcoded `MFF_OCE`
        # widens once more via a Bottleneck stage on top of the fused
        # embedding); `out_channels` lets `MambaADNet` pin this to
        # whatever `dims_decoder[0]` the decoder actually expects instead
        # of the two having to be kept in sync by hand.
        out_channels = out_channels or c3 * 2
        if out_channels % Bottleneck.expansion:
            raise ValueError(
                f"MultiScaleFusion: out_channels={out_channels} must be a multiple of "
                f"Bottleneck.expansion ({Bottleneck.expansion})."
            )

        self.down1to2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(c2)
        )
        self.proj2 = nn.Sequential(nn.Conv2d(c2, c2, kernel_size=1), norm_layer(c2))
        self.down2to3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(c3)
        )
        self.proj3 = nn.Sequential(nn.Conv2d(c3, c3, kernel_size=1), norm_layer(c3))
        self.project = nn.Sequential(nn.Conv2d(c3, c3, kernel_size=1), norm_layer(c3))
        self.relu = nn.ReLU(inplace=True)
        self.fusion_stage = _resnet_bottleneck_stage(
            c3, out_channels // Bottleneck.expansion, fusion_blocks, stride=2, norm_layer=norm_layer
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        level1, level2, level3 = features
        fused2 = self.relu(self.proj2(self.down1to2(level1))) + level2
        fused3 = self.relu(self.proj3(self.down2to3(fused2))) + level3
        fused = self.relu(self.project(fused3))
        return self.fusion_stage(fused).contiguous()


class MambaADNet(nn.Module):
    """`teacher (frozen) -> MultiScaleFusion -> MambaUPNet (trainable)`.

    `forward` returns `(teacher_features, student_features)`, each a list
    of 3 same-shaped `(B, C_i, H_i, W_i)` tensors -- what the training
    loss and the anomaly map both compare level-by-level. Teacher features
    are always detached; freezing is enforced by `MambaADAdapter.train`
    (excluding `self.teacher.parameters()` from the optimizer) plus
    `.eval()` on the teacher submodule specifically, not by requires_grad
    alone, matching upstream's `MAMBAAD.train()` override.
    """

    def __init__(
        self,
        teacher: nn.Module,
        teacher_channels: list[int],
        dims_decoder: list[int],
        depths_decoder: list[int],
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        base_size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 4,
    ):
        super().__init__()
        self.teacher = teacher
        # `dims_decoder[0]` is the fused embedding's own channel count, not
        # an independent architecture choice -- pinning it here (rather
        # than letting a preset declare both and hoping they agree) is
        # what makes them structurally impossible to drift apart.
        self.fusion = MultiScaleFusion(teacher_channels=teacher_channels, out_channels=dims_decoder[0])
        self.decoder = MambaUPNet(
            dims_decoder=dims_decoder, depths_decoder=depths_decoder, d_state=d_state,
            drop_path_rate=drop_path_rate, base_size=base_size,
            scan_type=scan_type, num_direction=num_direction,
        )

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        teacher_features = [feature.detach() for feature in self.teacher(images)]
        student_features = self.decoder(self.fusion(teacher_features))
        return teacher_features, student_features

    def train(self, mode: bool = True) -> "MambaADNet":
        """Like upstream's `MAMBAAD.train()` override: `self.teacher`
        always stays in eval mode (frozen batch-norm running stats,
        no dropout) regardless of what the rest of the model is doing --
        a plain `model.train()` call must not accidentally un-freeze it.
        """

        self.training = mode
        self.teacher.eval()
        self.fusion.train(mode)
        self.decoder.train(mode)
        return self
