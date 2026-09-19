"""Model-variant registry, fabric-tailored training defaults, and
augmentation presets for the torchvision detection backend
(Faster R-CNN, Mask R-CNN — see the top-level README for why
`torchvision.models.detection` replaces `mmdetection` here: mmdetection's
`mmcv` dependency has no macOS arm64 wheel and no Python 3.14 support,
confirmed by a real install attempt in this environment; torchvision is
pure PyTorch, actively maintained, and installs everywhere torch does).

Mirrors `models/anomalib/presets.py` and `models/ultralytics/presets.py`:
we do not reimplement Faster/Mask R-CNN — `torchvision.models.detection`
already ships correct, maintained implementations (including COCO-pretrained
weights). What lives here is (1) resolution of friendly variant names to
the actual factory functions/weights enums, and (2) fine-tuning defaults
tuned for small fabric-defect datasets rather than COCO-scale training.

Every symbol referenced below was introspected against the installed
torchvision 0.27.1 (see `pip install -e ".[torchvision]"`), not guessed:
the pretrained-weights-plus-custom-num_classes incompatibility, the
`FastRCNNPredictor`/`MaskRCNNPredictor` head-swap recipe, and the
`tv_tensors`-based joint image/box/mask augmentation API were all verified
live before this module was written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    class nn_Module:
        pass
    nn = type("nn", (object,), {"Module": nn_Module})

# The patched module classes below are compiled by `torch.jit.script` when a
# benchmark exports them (and traced by `torch.export` when it profiles them).
# `torch.jit` walks each method's *own* code object and refuses an `import`
# statement inside it — "UnsupportedNodeError: import statements aren't
# supported", which cost DETR, Cascade R-CNN and DeepLabV3+ their resolution
# sweep — so every name those methods touch has to be a module global. Bound
# here, under the same optional guard, so this module still imports without
# torchvision/scipy. The factory functions (`build_model`, `convert_to_
# cascade`, ...) are never scripted and keep their own local imports.
try:
    import math
    import torch.nn.functional as F
    import torchvision.models as models
    from scipy.optimize import linear_sum_assignment
    from torchvision.models.detection._utils import BoxCoder, Matcher
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.roi_heads import maskrcnn_inference
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from torchvision.ops import box_convert, clip_boxes_to_image, generalized_box_iou
    from torchvision.ops import boxes as box_ops
except ImportError:  # pragma: no cover - the backends that install these patches need all of them
    math = None
    F = None
    models = None
    linear_sum_assignment = None
    BoxCoder = Matcher = None
    FastRCNNPredictor = TwoMLPHead = None
    MaskRCNNPredictor = None
    maskrcnn_inference = None
    GeneralizedRCNNTransform = None
    box_convert = clip_boxes_to_image = generalized_box_iou = None
    box_ops = None

# The patched `forward`s below run under `torch.export` / `torch.fx` when a
# benchmark profiles or resolution-sweeps them, and the tracer walks the
# function's *own* code object: an `import` statement inside a forward is a
# hard error ("UnsupportedNodeError: import statements aren't supported",
# which cost DETR, Cascade R-CNN and DeepLabV3+ their resolution slope). Every
# name a forward touches therefore has to be a module global, imported here
# under the same optional guard so this module stays importable without
# torchvision/scipy.
try:
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from torchvision.models.detection.roi_heads import maskrcnn_inference
    from torchvision.ops import box_convert, generalized_box_iou
except ImportError:  # pragma: no cover - the backends that install these patches need all of them
    F = None
    linear_sum_assignment = None
    maskrcnn_inference = None
    box_convert = None
    generalized_box_iou = None

# Friendly variant name -> torchvision factory function name + weights enum
# name (both resolved lazily in `build_model()` so importing this module
# never requires torchvision to be installed).
MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "fasterrcnn_resnet50_fpn": {
        "factory": "fasterrcnn_resnet50_fpn",
        "weights_enum": "FasterRCNN_ResNet50_FPN_Weights",
        "task": "detect",
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "factory": "fasterrcnn_resnet50_fpn_v2",
        "weights_enum": "FasterRCNN_ResNet50_FPN_V2_Weights",
        "task": "detect",
    },
    "maskrcnn_resnet50_fpn": {
        "factory": "maskrcnn_resnet50_fpn",
        "weights_enum": "MaskRCNN_ResNet50_FPN_Weights",
        "task": "instance_segmentation",
    },
    "maskrcnn_resnet50_fpn_v2": {
        "factory": "maskrcnn_resnet50_fpn_v2",
        "weights_enum": "MaskRCNN_ResNet50_FPN_V2_Weights",
        "task": "instance_segmentation",
    },
    "fasterrcnn_vgg16_fpn": {
        "factory": "fasterrcnn_vgg16_fpn",
        "weights_enum": "VGG16_Weights",
        "task": "detect",
        "custom": True,
    },
    "fasterrcnn_shufflenet_v2_x1_0_fpn": {
        "factory": "fasterrcnn_shufflenet_v2_x1_0_fpn",
        "weights_enum": "ShuffleNet_V2_X1_0_Weights",
        "task": "detect",
        "custom": True,
    },
    "cascadercnn_resnet50_fpn": {
        "factory": "cascadercnn_resnet50_fpn",
        "weights_enum": "FasterRCNN_ResNet50_FPN_Weights",
        "task": "detect",
        "cascade": True,
    },
    "cascadercnn_resnet50_fpn_v2": {
        "factory": "cascadercnn_resnet50_fpn_v2",
        "weights_enum": "FasterRCNN_ResNet50_FPN_V2_Weights",
        "task": "detect",
        "cascade": True,
    },
    "cascadercnn_vgg16_fpn": {
        "factory": "cascadercnn_vgg16_fpn",
        "weights_enum": "VGG16_Weights",
        "task": "detect",
        "cascade": True,
        "custom": True,
    },
    "cascadercnn_shufflenet_v2_x1_0_fpn": {
        "factory": "cascadercnn_shufflenet_v2_x1_0_fpn",
        "weights_enum": "ShuffleNet_V2_X1_0_Weights",
        "task": "detect",
        "cascade": True,
        "custom": True,
    },
    "detr_resnet50": {
        "factory": "detr_resnet50",
        "weights_enum": "ResNet50_Weights",
        "task": "detect",
        "custom": True,
        "detr": True,
    },
    "detr_vgg16": {
        "factory": "detr_vgg16",
        "weights_enum": "VGG16_Weights",
        "task": "detect",
        "custom": True,
        "detr": True,
    },
    "detr_shufflenet_v2_x1_0": {
        "factory": "detr_shufflenet_v2_x1_0",
        "weights_enum": "ShuffleNet_V2_X1_0_Weights",
        "task": "detect",
        "custom": True,
        "detr": True,
    },
    "unetplusplus_resnet34": {
        "factory": "unetplusplus_resnet34",
        "weights_enum": "ResNet34_Weights",
        "task": "segmentation",
        "custom": True,
        "segmentation": True,
    },
    "deeplabv3plus_resnet50": {
        "factory": "deeplabv3plus_resnet50",
        "weights_enum": "ResNet50_Weights",
        "task": "segmentation",
        "custom": True,
        "segmentation": True,
    },
}

VARIANT_ALIASES: dict[str, str] = {
    "fasterrcnn": "fasterrcnn_resnet50_fpn",
    "faster_rcnn": "fasterrcnn_resnet50_fpn",
    "faster-rcnn": "fasterrcnn_resnet50_fpn",
    "fasterrcnn_v2": "fasterrcnn_resnet50_fpn_v2",
    "faster_rcnn_v2": "fasterrcnn_resnet50_fpn_v2",
    "maskrcnn": "maskrcnn_resnet50_fpn",
    "mask_rcnn": "maskrcnn_resnet50_fpn",
    "mask-rcnn": "maskrcnn_resnet50_fpn",
    "maskrcnn_v2": "maskrcnn_resnet50_fpn_v2",
    "mask_rcnn_v2": "maskrcnn_resnet50_fpn_v2",
    "fasterrcnn_vgg16": "fasterrcnn_vgg16_fpn",
    "fasterrcnn_shufflenet_v2": "fasterrcnn_shufflenet_v2_x1_0_fpn",
    "cascadercnn": "cascadercnn_resnet50_fpn",
    "cascadercnn_v2": "cascadercnn_resnet50_fpn_v2",
    "cascadercnn_vgg16": "cascadercnn_vgg16_fpn",
    "cascadercnn_shufflenet_v2": "cascadercnn_shufflenet_v2_x1_0_fpn",
    "detr": "detr_resnet50",
    "detr_resnet": "detr_resnet50",
    "detr_shufflenet_v2": "detr_shufflenet_v2_x1_0",
    "unetplusplus": "unetplusplus_resnet34",
    "unetpp": "unetplusplus_resnet34",
    "unetxx": "unetplusplus_resnet34",
    "deeplabv3plus": "deeplabv3plus_resnet50",
    "deeplabv3p": "deeplabv3plus_resnet50",
}

# Fine-tuning defaults for small fabric datasets. torchvision's own
# reference detection training script (references/detection/train.py) uses
# SGD lr=0.005/momentum=0.9/weight_decay=0.0005 with a 3-epoch StepLR decay
# for training Faster/Mask R-CNN *from ImageNet init* on full COCO — those
# numbers are tuned for a much bigger dataset trained for much longer. Since
# we start from full COCO-pretrained detection weights (see
# `adapter.py::_build_model`) and fine-tune on a low-/few-shot fabric
# selection, we use a lower learning rate (less to relearn) and a much
# longer patience-based early stop instead of a fixed decay schedule sized
# for COCO's epoch count.
COMMON_FABRIC_TRAIN_DEFAULTS: dict[str, Any] = {
    "epochs": 30,
    "batch_size": 4,
    "optimizer": "sgd",  # 'sgd' | 'adamw'
    "lr": 0.002,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler": "cosine",  # 'cosine' | 'step' | 'none'
    "step_size": 10,
    "gamma": 0.1,
    "warmup_epochs": 1,  # linear LR warmup within the first epoch, per torchvision's reference recipe
    "grad_clip_norm": 5.0,
    "patience": 8,  # early stop on no val-mAP improvement
    # Sample-backed datasets can be staged from transient local paths; zero
    # workers is the portable default on macOS and can be overridden for a
    # prepared Linux/CUDA training host.
    "num_workers": 0,
    "trainable_backbone_layers": 3,  # of 5; freeze the earliest (most generic) ResNet stages
    "amp": False,  # mixed precision; only actually engaged on CUDA, see engine.run_training
    # --- augmentation, retuned for small texture defects (see build_transforms) ---
    "hflip_prob": 0.5,
    "vflip_prob": 0.5,
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.02},
}

# DETR is a randomly-initialised transformer sitting on an ImageNet ResNet
# backbone: torchvision ships no DETR implementation and the official COCO
# `detr_resnet50` checkpoint is not reachable from this environment (no
# github.com / huggingface.co), so only the backbone can be pretrained. It
# therefore follows the *published DETR schedule* rather than the 30-epoch SGD
# fabric default — AdamW with a split LR (transformer/heads 1e-4, backbone
# 1e-5) for 300 epochs. `patience: 0` disables mAP early stopping on purpose:
# DETR's val mAP sits at ~0 for the first tens of epochs, so any patience
# would stop the run long before the queries have differentiated.
_DETR_TRAIN_DEFAULTS: dict[str, Any] = {
    "epochs": 300,
    "batch_size": 8,
    "optimizer": "adamw",
    "lr": 0.0001,
    "backbone_lr": 0.00001,
    "patience": 0,
}

VARIANT_TRAIN_OVERRIDES: dict[str, dict[str, Any]] = {
    "fasterrcnn_resnet50_fpn": {},
    "fasterrcnn_resnet50_fpn_v2": {"lr": 0.0015},  # v2 head is heavier; slightly gentler LR
    "maskrcnn_resnet50_fpn": {"batch_size": 2},  # masks triple memory use per image
    "maskrcnn_resnet50_fpn_v2": {"batch_size": 2, "lr": 0.0015},
    "detr_resnet50": dict(_DETR_TRAIN_DEFAULTS),
    "detr_vgg16": dict(_DETR_TRAIN_DEFAULTS),
    "detr_shufflenet_v2_x1_0": dict(_DETR_TRAIN_DEFAULTS),
}


def resolve_variant(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key in MODEL_VARIANTS:
        return key
    if key in VARIANT_ALIASES:
        return VARIANT_ALIASES[key]
    known = sorted(MODEL_VARIANTS)
    raise KeyError(f"unknown torchvision detection variant {name!r}. Known variants: {known}")


def variant_task(name: str) -> str:
    """'detect' (Faster R-CNN) or 'instance_segmentation' (Mask R-CNN)."""

    return MODEL_VARIANTS[resolve_variant(name)]["task"]


def uses_masks(name: str) -> bool:
    return variant_task(name) == "instance_segmentation"


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    backbone_weights: bool = True,
    offline: bool = False,
):
    """Construct the torchvision detection model for `name`, sized for
    `num_classes` foreground classes + background (i.e. pass
    `len(class_names) + 1`).

    `pretrained=True` loads full COCO-pretrained detection weights, then
    swaps the classifier (and mask predictor, for Mask R-CNN) head to match
    `num_classes` — the standard torchvision fine-tuning recipe. Passing
    `num_classes` straight into the factory together with COCO weights is
    *not* supported by torchvision (raises `ValueError`: expects 91); the
    head-swap is the only way to combine pretrained features with a custom
    class count.

    `pretrained=False` builds from an ImageNet-pretrained backbone only
    (random-init detection head), for training from scratch — unless
    `backbone_weights=False`, which skips even that download. Use
    `backbone_weights=False` when you're about to overwrite every parameter
    with a full `state_dict` anyway (see `adapter.load_weights`):
    downloading ImageNet weights just to discard them immediately is wasted
    network I/O, and makes checkpoint reloading needlessly dependent on
    internet access.

    `min_size`/`max_size` override the model's internal `GeneralizedRCNNTransform`
    resize range (default 800/1333, tuned for COCO photos). ZJU-Leaper images
    are 512x512, so the default *upscales* them for no benefit; smaller
    values speed up both training and inference with little accuracy cost on
    fabric's uniform, non-photographic content.
    """

    import torchvision.models.detection as tv_detection

    variant = resolve_variant(name)
    spec = MODEL_VARIANTS[variant]

    if spec.get("cascade"):
        base_variant = variant.replace("cascadercnn", "fasterrcnn", 1)
        model = build_model(
            name=base_variant,
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            backbone_weights=backbone_weights,
            offline=offline,
        )
        model = convert_to_cascade(model, num_classes)
        return model

    size_kwargs: dict[str, Any] = {}
    if min_size is not None:
        size_kwargs["min_size"] = min_size
    if max_size is not None:
        size_kwargs["max_size"] = max_size

    if spec.get("detr"):
        backbone_name = variant.replace("detr_", "", 1)
        model = build_detr(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            backbone_weights=backbone_weights,
            offline=offline,
        )
        return model

    if spec.get("segmentation"):
        if variant == "unetplusplus_resnet34":
            model = build_unetplusplus(
                num_classes=num_classes,
                pretrained=pretrained,
            )
        elif variant == "deeplabv3plus_resnet50":
            model = build_deeplabv3plus(
                num_classes=num_classes,
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"Unknown segmentation variant: {variant}")
        return model

    if spec.get("custom"):
        import torchvision.models as models
        weights_enum = getattr(models, spec["weights_enum"])
        factory = globals()[spec["factory"]]

        if pretrained:
            weights = weights_enum.DEFAULT
            if offline:
                from fabric_defect_hub.core.preflight import require_cached_weight
                import torch

                checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
                cached = require_cached_weight(weights.url, "torchvision", [checkpoint_dir])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                expected_path = checkpoint_dir / Path(weights.url).name
                if cached != expected_path and not expected_path.exists():
                    import shutil

                    shutil.copy2(cached, expected_path)
            model = factory(
                num_classes=num_classes,
                pretrained_backbone=True,
                trainable_backbone_layers=trainable_backbone_layers,
                **size_kwargs,
            )
        else:
            model = factory(
                num_classes=num_classes,
                pretrained_backbone=backbone_weights,
                trainable_backbone_layers=trainable_backbone_layers,
                **size_kwargs,
            )
        return model

    factory = getattr(tv_detection, spec["factory"])
    if pretrained:
        weights_enum = getattr(tv_detection, spec["weights_enum"])
        weights = weights_enum.DEFAULT
        if offline:
            from fabric_defect_hub.core.preflight import require_cached_weight
            import torch

            checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
            cached = require_cached_weight(weights.url, "torchvision", [checkpoint_dir])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            expected_path = checkpoint_dir / Path(weights.url).name
            if cached != expected_path and not expected_path.exists():
                import shutil

                shutil.copy2(cached, expected_path)
        model = factory(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            **size_kwargs,
        )
        _replace_head(model, num_classes, with_mask=uses_masks(variant))
    else:
        no_download_kwargs: dict[str, Any] = {} if backbone_weights else {"weights_backbone": None}
        if not backbone_weights and trainable_backbone_layers is not None:
            import warnings

            warnings.warn(
                f"trainable_backbone_layers={trainable_backbone_layers} has no effect without "
                "backbone weights (pretrained=False, offline=True, or a checkpoint reload): "
                "torchvision trains all 5 backbone stages in that case.",
                RuntimeWarning,
                stacklevel=2,
            )
        model = factory(
            weights=None,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            **size_kwargs,
            **no_download_kwargs,
        )
    return model


def _replace_head(model, num_classes: int, with_mask: bool) -> None:
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if with_mask:
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)


def default_train_kwargs(name: str) -> dict[str, Any]:
    """Fabric-tailored default training kwargs for `name` (copy, safe to mutate)."""

    variant = resolve_variant(name)
    merged = dict(COMMON_FABRIC_TRAIN_DEFAULTS)
    merged.update(VARIANT_TRAIN_OVERRIDES.get(variant, {}))
    return merged


def build_transforms(train: bool, hflip_prob: float = 0.5, vflip_prob: float = 0.5, color_jitter: dict | None = None):
    """Box/mask-aware augmentation pipeline via `torchvision.transforms.v2`.

    Fabric texture has no preferred orientation and mirror symmetry is
    label-preserving, so both flips are safe free augmentation (unlike
    COCO's natural-object-photo default of horizontal-flip-only). Colour
    jitter is kept mild — some fabric defects (stains, discoloration) are
    themselves colour cues we don't want to wash out. Eval/predict gets no
    augmentation, only the dtype conversion.
    """

    import torch
    from torchvision.transforms import v2

    if not train:
        return v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    color_jitter = color_jitter or {}
    steps = [
        v2.RandomHorizontalFlip(p=hflip_prob),
        v2.RandomVerticalFlip(p=vflip_prob),
        v2.ColorJitter(**color_jitter),
        v2.ToDtype(torch.float32, scale=True),
    ]
    return v2.Compose(steps)


def list_supported_variants() -> list[str]:
    return sorted(MODEL_VARIANTS)


def fasterrcnn_vgg16_fpn(
    *,
    num_classes: int = 91,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    **kwargs
):
    import torchvision.models as models
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models.detection import FasterRCNN

    weights = models.VGG16_Weights.DEFAULT if pretrained_backbone else None
    backbone_model = models.vgg16(weights=weights)

    stages = [
        list(range(0, 5)),
        list(range(5, 10)),
        list(range(10, 17)),
        list(range(17, 24)),
        list(range(24, 31)),
    ]
    tbl = trainable_backbone_layers if trainable_backbone_layers is not None else 3
    if tbl < 0 or tbl > 5:
        raise ValueError(f"trainable_backbone_layers should be in [0, 5], got {tbl}")

    frozen_stages = 5 - tbl
    for stage_idx in range(frozen_stages):
        for idx in stages[stage_idx]:
            for param in backbone_model.features[idx].parameters():
                param.requires_grad = False

    return_layers = {"9": "0", "16": "1", "23": "2", "30": "3"}
    in_channels_list = [128, 256, 512, 512]
    out_channels = 256
    backbone = BackboneWithFPN(
        backbone_model.features,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )

    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)
    return model


def fasterrcnn_shufflenet_v2_x1_0_fpn(
    *,
    num_classes: int = 91,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    **kwargs
):
    import torchvision.models as models
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models.detection import FasterRCNN

    weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained_backbone else None
    backbone_model = models.shufflenet_v2_x1_0(weights=weights)

    stages = [
        ["conv1", "maxpool"],
        ["stage2"],
        ["stage3"],
        ["stage4"],
    ]
    tbl = trainable_backbone_layers if trainable_backbone_layers is not None else 3
    if tbl < 0 or tbl > 4:
        raise ValueError(f"trainable_backbone_layers should be in [0, 4], got {tbl}")

    frozen_stages = 4 - tbl
    for stage_idx in range(frozen_stages):
        for name in stages[stage_idx]:
            module = getattr(backbone_model, name)
            for param in module.parameters():
                param.requires_grad = False

    return_layers = {"maxpool": "0", "stage2": "1", "stage3": "2", "stage4": "3"}
    in_channels_list = [24, 116, 232, 464]
    out_channels = 256
    backbone = BackboneWithFPN(
        backbone_model,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )

    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)
    return model


_CascadeRoIHeads = None


def _get_cascade_roi_heads_class():
    global _CascadeRoIHeads
    if _CascadeRoIHeads is not None:
        return _CascadeRoIHeads

    import torch
    import torch.nn as nn
    from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

    class CascadeRoIHeads(RoIHeads):
        def __init__(
            self,
            box_roi_pool,
            box_heads,
            box_predictors,
            fg_iou_thresholds,
            bg_iou_thresholds,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights_list,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool=None,
            mask_head=None,
            mask_predictor=None,
        ):
            super().__init__(
                box_roi_pool=box_roi_pool,
                box_head=box_heads[0],
                box_predictor=box_predictors[0],
                fg_iou_thresh=fg_iou_thresholds[0],
                bg_iou_thresh=bg_iou_thresholds[0],
                batch_size_per_image=batch_size_per_image,
                positive_fraction=positive_fraction,
                bbox_reg_weights=bbox_reg_weights_list[0],
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img,
                mask_roi_pool=mask_roi_pool,
                mask_head=mask_head,
                mask_predictor=mask_predictor,
            )
            self.box_heads = nn.ModuleList(box_heads)
            self.box_predictors = nn.ModuleList(box_predictors)

            self.proposal_matchers = [
                Matcher(fg, bg, allow_low_quality_matches=False)
                for fg, bg in zip(fg_iou_thresholds, bg_iou_thresholds)
            ]
            self.box_coders = [
                BoxCoder(weights) for weights in bbox_reg_weights_list
            ]

        @torch.jit.unused
        def select_training_samples_stage(self, stage_idx, proposals, targets):
            if targets is None:
                raise ValueError("targets should not be None")
            dtype = proposals[0].dtype
            device = proposals[0].device

            gt_boxes = [t["boxes"].to(dtype) for t in targets]
            gt_labels = [t["labels"] for t in targets]

            proposals = self.add_gt_proposals(proposals, gt_boxes)

            matched_idxs = []
            labels = []
            for img_id in range(len(proposals)):
                proposals_in_image = proposals[img_id]
                gt_boxes_in_image = gt_boxes[img_id]
                gt_labels_in_image = gt_labels[img_id]

                if gt_boxes_in_image.numel() == 0:
                    labels.append(torch.zeros(proposals_in_image.shape[0], dtype=torch.int64, device=device))
                    matched_idxs.append(torch.zeros(proposals_in_image.shape[0], dtype=torch.int64, device=device))
                else:
                    match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
                    matched_idx = self.proposal_matchers[stage_idx](match_quality_matrix)

                    matched_labels = gt_labels_in_image[matched_idx.clamp(min=0)]
                    matched_labels = matched_labels.clone()

                    bg_inds = matched_idx == self.proposal_matchers[stage_idx].BELOW_LOW_THRESHOLD
                    matched_labels[bg_inds] = 0

                    ignore_inds = matched_idx == self.proposal_matchers[stage_idx].BETWEEN_THRESHOLDS
                    matched_labels[ignore_inds] = -1

                    matched_idxs.append(matched_idx)
                    labels.append(matched_labels)

            sampled_inds = self.subsample(labels)

            matched_gt_boxes = []
            sampled_proposals = []
            sampled_labels = []
            sampled_matched_idxs = []

            num_images = len(proposals)
            for img_id in range(num_images):
                img_sampled_inds = sampled_inds[img_id]

                p = proposals[img_id][img_sampled_inds]
                l = labels[img_id][img_sampled_inds]
                m = matched_idxs[img_id][img_sampled_inds]

                sampled_proposals.append(p)
                sampled_labels.append(l)
                sampled_matched_idxs.append(m)

                gt_boxes_in_image = gt_boxes[img_id]
                if gt_boxes_in_image.numel() == 0:
                    gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
                matched_gt_boxes.append(gt_boxes_in_image[m.clamp(min=0)])

            regression_targets = self.box_coders[stage_idx].encode(matched_gt_boxes, sampled_proposals)
            return sampled_proposals, sampled_matched_idxs, sampled_labels, regression_targets

        @torch.jit.unused
        def refine_proposals(self, stage_idx, proposals, box_regression, labels, image_shapes):

            pred_boxes = self.box_coders[stage_idx].decode(box_regression.detach(), proposals)
            num_classes = pred_boxes.shape[1]

            boxes_per_image = [len(p) for p in proposals]
            concat_labels = torch.cat(labels, dim=0)
            clamped_labels = concat_labels.clamp(min=0, max=num_classes - 1)

            num_proposals = pred_boxes.shape[0]
            if num_proposals > 0:
                gather_idx = clamped_labels.unsqueeze(-1).unsqueeze(-1).expand(num_proposals, 1, 4)
                refined_boxes = torch.gather(pred_boxes, 1, gather_idx).squeeze(1)
            else:
                refined_boxes = pred_boxes.reshape(0, 4)

            refined_proposals = []
            start_idx = 0
            for i, num_boxes in enumerate(boxes_per_image):
                end_idx = start_idx + num_boxes
                boxes_in_image = refined_boxes[start_idx:end_idx]
                boxes_in_image = clip_boxes_to_image(boxes_in_image, image_shapes[i])
                refined_proposals.append(boxes_in_image)
                start_idx = end_idx

            return refined_proposals

        def refine_proposals_inference(self, stage_idx, proposals, box_regression, pred_classes, image_shapes):

            pred_boxes = self.box_coders[stage_idx].decode(box_regression, proposals)
            num_classes = pred_boxes.shape[1]
            boxes_per_image = [len(p) for p in proposals]

            clamped_classes = pred_classes.clamp(min=0, max=num_classes - 1)
            num_proposals = pred_boxes.shape[0]

            if num_proposals > 0:
                gather_idx = clamped_classes.unsqueeze(-1).unsqueeze(-1).expand(num_proposals, 1, 4)
                refined_boxes = torch.gather(pred_boxes, 1, gather_idx).squeeze(1)
            else:
                refined_boxes = pred_boxes.reshape(0, 4)

            refined_proposals = []
            start_idx = 0
            for i, num_boxes in enumerate(boxes_per_image):
                end_idx = start_idx + num_boxes
                boxes_in_image = refined_boxes[start_idx:end_idx]
                boxes_in_image = clip_boxes_to_image(boxes_in_image, image_shapes[i])
                refined_proposals.append(boxes_in_image)
                start_idx = end_idx

            return refined_proposals

        def postprocess_detections_cascade(self, pred_scores, box_regression, proposals, image_shapes):

            device = pred_scores.device
            num_classes = pred_scores.shape[-1]

            boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
            pred_boxes = self.box_coders[2].decode(box_regression, proposals)

            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)

            all_boxes = []
            all_scores = []
            all_labels = []
            for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
                boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

                labels = torch.arange(num_classes, device=device)
                labels = labels.view(1, -1).expand_as(scores)

                boxes = boxes[:, 1:]
                scores = scores[:, 1:]
                labels = labels[:, 1:]

                boxes = boxes.reshape(-1, 4)
                scores = scores.reshape(-1)
                labels = labels.reshape(-1)

                inds = torch.where(scores > self.score_thresh)[0]
                boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

                keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
                keep = keep[: self.detections_per_img]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

            return all_boxes, all_scores, all_labels

        def forward(self, features, proposals, image_shapes, targets=None):

            if self.training:
                if targets is None:
                    raise ValueError("targets should not be None in training mode")

                losses = {}
                current_proposals = proposals

                for stage in range(3):
                    sampled_proposals, matched_idxs, labels, regression_targets = \
                        self.select_training_samples_stage(stage, current_proposals, targets)

                    box_features = self.box_roi_pool(features, sampled_proposals, image_shapes)
                    box_features = self.box_heads[stage](box_features)
                    class_logits, box_regression = self.box_predictors[stage](box_features)

                    loss_classifier, loss_box_reg = fastrcnn_loss(
                        class_logits, box_regression, labels, regression_targets
                    )

                    losses[f"loss_classifier_stage{stage+1}"] = loss_classifier
                    losses[f"loss_box_reg_stage{stage+1}"] = loss_box_reg

                    if stage < 2:
                        current_proposals = self.refine_proposals(
                            stage, sampled_proposals, box_regression, labels, image_shapes
                        )

                return [], losses
            else:
                sum_scores = None
                current_proposals = proposals
                final_box_regression = None

                for stage in range(3):
                    box_features = self.box_roi_pool(features, current_proposals, image_shapes)
                    box_features = self.box_heads[stage](box_features)
                    class_logits, box_regression = self.box_predictors[stage](box_features)

                    probs = F.softmax(class_logits, dim=-1)
                    if sum_scores is None:
                        sum_scores = probs
                    else:
                        sum_scores = sum_scores + probs

                    final_box_regression = box_regression

                    if stage < 2:
                        pred_classes = probs.argmax(dim=-1)
                        current_proposals = self.refine_proposals_inference(
                            stage, current_proposals, box_regression, pred_classes, image_shapes
                        )

                avg_scores = sum_scores / 3.0

                boxes, scores, labels = self.postprocess_detections_cascade(
                    avg_scores, final_box_regression, current_proposals, image_shapes
                )

                result = []
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                        }
                    )

                if self.has_mask():
                    mask_proposals = [p["boxes"] for p in result]
                    if self.mask_roi_pool is not None and self.mask_head is not None and self.mask_predictor is not None:
                        mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                        mask_features = self.mask_head(mask_features)
                        mask_logits = self.mask_predictor(mask_features)

                        labels_list = [r["labels"] for r in result]
                        masks_probs = maskrcnn_inference(mask_logits, labels_list)
                        for mask_prob, r in zip(masks_probs, result):
                            r["masks"] = mask_prob

                return result, {}

    _CascadeRoIHeads = CascadeRoIHeads
    return _CascadeRoIHeads


def convert_to_cascade(model, num_classes: int) -> nn.Module:
    import torch.nn as nn
    from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

    roi_heads = model.roi_heads
    box_roi_pool = roi_heads.box_roi_pool
    original_head = roi_heads.box_head
    original_predictor = roi_heads.box_predictor

    in_channels_head = original_head.fc6.in_features
    representation_size = original_head.fc7.out_features
    in_channels_pred = original_predictor.cls_score.in_features

    box_heads = [
        TwoMLPHead(in_channels_head, representation_size)
        for _ in range(3)
    ]
    box_predictors = [
        FastRCNNPredictor(in_channels_pred, num_classes)
        for _ in range(3)
    ]

    box_heads[0].load_state_dict(original_head.state_dict())
    box_predictors[0].load_state_dict(original_predictor.state_dict())

    CascadeRoIHeads = _get_cascade_roi_heads_class()
    cascade_roi_heads = CascadeRoIHeads(
        box_roi_pool=box_roi_pool,
        box_heads=box_heads,
        box_predictors=box_predictors,
        fg_iou_thresholds=[0.5, 0.6, 0.7],
        bg_iou_thresholds=[0.5, 0.6, 0.7],
        batch_size_per_image=roi_heads.fg_bg_sampler.batch_size_per_image,
        positive_fraction=roi_heads.fg_bg_sampler.positive_fraction,
        bbox_reg_weights_list=[
            (10.0, 10.0, 5.0, 5.0),
            (20.0, 20.0, 10.0, 10.0),
            (30.0, 30.0, 15.0, 15.0)
        ],
        score_thresh=roi_heads.score_thresh,
        nms_thresh=roi_heads.nms_thresh,
        detections_per_img=roi_heads.detections_per_img,
        mask_roi_pool=roi_heads.mask_roi_pool,
        mask_head=roi_heads.mask_head,
        mask_predictor=roi_heads.mask_predictor,
    )

    model.roi_heads = cascade_roi_heads
    return model


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):

        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        # `@torch.no_grad()` would need `torch` at class-body/module-import
        # time -- unlike every other torch use in this file, which is
        # lazily inside a method body, so importing this module never
        # requires torch to be installed. Doing it as a `with` block instead
        # keeps that guarantee while still disabling autograd for the match.

        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            indices = []
            for b in range(bs):
                out_prob = outputs["pred_logits"][b].softmax(-1)
                out_bbox = outputs["pred_boxes"][b]

                tgt_ids = targets[b]["labels"]
                tgt_bbox = targets[b]["boxes"]

                if len(tgt_ids) == 0:
                    indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                    continue

                cost_class = -out_prob[:, tgt_ids]
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                out_bbox_xyxy = box_convert(out_bbox, "cxcywh", "xyxy")
                tgt_bbox_xyxy = box_convert(tgt_bbox, "cxcywh", "xyxy")
                cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                # A degenerate box — zero width or height, which ZJU-Leaper's
                # 1-pixel-tall sliver annotations become once `cxcywh` is
                # converted to `xyxy`, and which an early DETR prediction can
                # collapse into — makes `box_iou`/`generalized_box_iou` divide
                # by a zero area and return NaN. scipy then rejects the *whole*
                # cost matrix ("matrix contains invalid numeric entries"),
                # which killed a 300-epoch retrain at epoch 57 (2026-09-18).
                # A large finite cost leaves those pairs unmatched, which is
                # what a box with no area deserves, and the epoch survives.
                C = torch.nan_to_num(C, nan=1e8, posinf=1e8, neginf=-1e8)
                C = C.cpu()

                src_ind, tgt_ind = linear_sum_assignment(C.numpy())
                indices.append((
                    torch.as_tensor(src_ind, dtype=torch.int64, device=out_bbox.device),
                    torch.as_tensor(tgt_ind, dtype=torch.int64, device=out_bbox.device)
                ))
            return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):

        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx] = targets[i]["labels"][tgt_idx]
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):

        pred_boxes = outputs["pred_boxes"]
        device = pred_boxes.device
        src_boxes = []
        tgt_boxes = []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(pred_boxes[i, src_idx])
                tgt_boxes.append(targets[i]["boxes"][tgt_idx])
        if len(src_boxes) == 0:
            return {"loss_bbox": pred_boxes.sum() * 0.0, "loss_giou": pred_boxes.sum() * 0.0}
        src_boxes = torch.cat(src_boxes, dim=0)
        tgt_boxes = torch.cat(tgt_boxes, dim=0)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")
        src_boxes_xyxy = box_convert(src_boxes, "cxcywh", "xyxy")
        tgt_boxes_xyxy = box_convert(tgt_boxes, "cxcywh", "xyxy")
        giou_matrix = generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)
        loss_giou = (1.0 - torch.diag(giou_matrix)).mean()
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)
        losses = {}
        for loss_name in self.losses:
            if loss_name == "labels":
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss_name == "boxes":
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        # `weight_dict` is DETR's loss weighting (ce 1 / bbox 5 / giou 2) and
        # has to be applied *here*: `engine.train_one_epoch` just sums whatever
        # this returns, so returning the raw components trained every term at
        # weight 1. With 100 queries and only ~1.7 objects per ZJU-Leaper image
        # that let the classification term dominate and the decoder collapse
        # onto one constant box for every query and every image (the published
        # `detr_resnet50.pt` predicts `[60, 52, 427, 462]` with score 0.005 on
        # every input) — see tests/test_detr.py::test_detr_losses_use_weight_dict.
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_queries=100,
        min_size=800,
        max_size=1333,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        backbone.eval()
        with torch.no_grad():
            dummy_out = backbone(torch.zeros(1, 3, 64, 64))
        if isinstance(dummy_out, torch.Tensor):
            in_channels = dummy_out.shape[1]
        else:
            if "3" in dummy_out:
                in_channels = dummy_out["3"].shape[1]
            else:
                last_key = list(dummy_out.keys())[-1]
                in_channels = dummy_out[last_key].shape[1]

        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.pe = PositionEmbeddingSine(num_pos_feats=d_model // 2)
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self.transform = GeneralizedRCNNTransform(
            min_size=min_size, max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            eos_coef=0.1,
            losses=["labels", "boxes"],
        )

    def forward(self, images, targets=None):

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        transformed_images, transformed_targets = self.transform(images, targets)

        features = self.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            src = features
        else:
            if "3" in features:
                src = features["3"]
            else:
                last_key = list(features.keys())[-1]
                src = features[last_key]

        src_proj = self.input_proj(src)
        pos = self.pe(src_proj)

        B, C, H, W = src_proj.shape
        src_flatten = src_proj.flatten(2).transpose(1, 2)
        pos_flatten = pos.flatten(2).transpose(1, 2)

        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.transformer(src_flatten + pos_flatten, query_embed)

        pred_logits = self.class_embed(out)
        pred_boxes = self.bbox_embed(out).sigmoid()

        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        if self.training:
            norm_targets = []
            for t, img_size in zip(transformed_targets, transformed_images.image_sizes):
                h, w = img_size
                boxes = t["boxes"]
                boxes_cxcywh = box_convert(boxes, "xyxy", "cxcywh")
                normalization_tensor = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
                boxes_norm = boxes_cxcywh / normalization_tensor

                norm_targets.append({
                    "labels": t["labels"],
                    "boxes": boxes_norm
                })

            losses = self.criterion(outputs, norm_targets)
            return losses
        else:
            results = []
            for i, img_size in enumerate(transformed_images.image_sizes):
                h, w = img_size
                scaling_tensor = torch.tensor([w, h, w, h], dtype=torch.float32, device=pred_boxes.device)
                abs_boxes_cxcywh = pred_boxes[i] * scaling_tensor
                abs_boxes = box_convert(abs_boxes_cxcywh, "cxcywh", "xyxy")

                probs = pred_logits[i].softmax(-1)
                fg_probs = probs[:, 1:]
                max_scores, max_labels = fg_probs.max(-1)
                max_labels = max_labels + 1

                results.append({
                    "boxes": abs_boxes,
                    "scores": max_scores,
                    "labels": max_labels,
                })

            results = self.transform.postprocess(results, transformed_images.image_sizes, original_image_sizes)
            return results


def build_detr(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    backbone_weights: bool = True,
    offline: bool = False,
):
    import torchvision.models.detection.backbone_utils as bu
    import torchvision.models as models

    if backbone_name == "resnet50":
        backbone = bu.resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None,
            trainable_layers=trainable_backbone_layers if trainable_backbone_layers is not None else 3,
        )
    elif backbone_name == "vgg16":
        fasterrcnn = fasterrcnn_vgg16_fpn(
            num_classes=num_classes,
            pretrained_backbone=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        backbone = fasterrcnn.backbone
    elif backbone_name == "shufflenet_v2_x1_0":
        fasterrcnn = fasterrcnn_shufflenet_v2_x1_0_fpn(
            num_classes=num_classes,
            pretrained_backbone=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        backbone = fasterrcnn.backbone
    else:
        raise ValueError(f"Unsupported DETR backbone {backbone_name}")

    model = DETR(
        backbone=backbone,
        num_classes=num_classes,
        min_size=min_size or 800,
        max_size=max_size or 1333,
    )
    return model


class SegVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class UNetPlusPlusResNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        nb_filter = [64, 64, 128, 256, 512]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = SegVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SegVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SegVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SegVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SegVGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SegVGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SegVGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SegVGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SegVGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SegVGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x0_0 = self.stem(x)  # H/2, W/2, 64
        x1_0 = self.layer1(self.maxpool(x0_0))  # H/4, W/4, 64

        # Dense Skip Pathways
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.layer2(x1_0)  # H/8, W/8, 128
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.layer3(x2_0)  # H/16, W/16, 256
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.layer4(x3_0)  # H/32, W/32, 512
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        out = self.final(x0_4)
        out = self.up_final(out)
        return out


class SegASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_channels + 2 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):

        res = []
        for conv in self.convs:
            res.append(conv(x))

        h, w = x.shape[2:]
        pool_feat = self.pooling(x)
        pool_feat_up = F.interpolate(pool_feat, size=(h, w), mode='bilinear', align_corners=True)
        res.append(pool_feat_up)

        out = torch.cat(res, dim=1)
        return self.project(out)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = SegASPP(in_channels=2048, out_channels=256, rates=[6, 12, 18])

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):

        h_img, w_img = x.shape[2:]

        x_stem = self.stem(x)
        low_level_feat = self.layer1(x_stem)
        x_enc = self.layer2(low_level_feat)
        x_enc = self.layer3(x_enc)
        x_enc = self.layer4(x_enc)

        aspp_feat = self.aspp(x_enc)
        aspp_feat_up = F.interpolate(aspp_feat, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)

        low_level_feat_proj = self.low_level_proj(low_level_feat)

        concat_feat = torch.cat([aspp_feat_up, low_level_feat_proj], dim=1)
        out = self.decoder(concat_feat)

        out = F.interpolate(out, size=(h_img, w_img), mode='bilinear', align_corners=True)
        return out


def build_unetplusplus(num_classes: int = 1, pretrained: bool = True):
    return UNetPlusPlusResNet(num_classes=num_classes, pretrained=pretrained)


def build_deeplabv3plus(num_classes: int = 1, pretrained: bool = True):
    return DeepLabV3Plus(num_classes=num_classes, pretrained=pretrained)
