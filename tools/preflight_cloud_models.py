"""Validate the four missing-model cloud prerequisites without training."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    key: str
    config: str
    modules: tuple[str, ...]
    paths: tuple[Path, ...]


def _root_from_env(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, str(ROOT / fallback)))


def _checks() -> tuple[Check, ...]:
    zju = _root_from_env("ZJU_LEAPER_ROOT", "data/ZJU-Leaper")
    mvtec = _root_from_env("MVTEC_AD_ROOT", "datasets/general/MVTec AD")
    imagenette = _root_from_env("IMAGENET_DIR", "datasets/general/imagenette")
    return (
        Check("EfficientAD", "configs/models/anomalib_efficientad.yaml", ("anomalib", "torch", "timm"), (zju, imagenette)),
        Check("WinCLIP", "configs/models/anomalib_winclip.yaml", ("anomalib", "torch", "open_clip"), (zju,)),
        Check("MoECLIP", "configs/models/moeclip_example.yaml", ("torch", "timm", "kornia", "transformers"), (mvtec, zju, ROOT / "components/moeclip/model/ViT-L-14-336px.pt")),
        Check("MambaAD", "configs/models/mambaad_example.yaml", ("torch", "timm", "einops"), (zju,)),
    )


def main() -> int:
    failures = 0
    for check in _checks():
        missing_modules = []
        for module in check.modules:
            try:
                importlib.import_module(module)
            except Exception as exc:
                missing_modules.append(f"{module} ({type(exc).__name__})")
        missing_paths = [str(path) for path in check.paths if not path.exists()]
        config = ROOT / check.config
        if not config.is_file():
            missing_paths.append(str(config))
        if missing_modules or missing_paths:
            failures += 1
            print(f"BLOCKED {check.key}")
            if missing_modules:
                print("  missing modules: " + ", ".join(missing_modules))
            if missing_paths:
                print("  missing paths: " + ", ".join(missing_paths))
        else:
            print(f"READY   {check.key}: {check.config}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
