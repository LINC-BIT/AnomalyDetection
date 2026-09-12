#!/usr/bin/env python3
"""Export a registered model artifact, optionally building a TensorRT engine
or applying post-training ONNX quantization for edge deployment."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.loader import load_model
from fabric_defect_hub.models.base import Artifact

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True, help="model variant/name")
    parser.add_argument("--artifact", required=True, help="trained checkpoint path")
    parser.add_argument("--target", required=True, help="export target, such as onnx or torchscript")
    parser.add_argument("--metadata", help="JSON file containing Artifact metadata")
    parser.add_argument("--export-config", help="JSON object passed to adapters that support config")
    parser.add_argument("--output", help="engine output path when --target tensorrt")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
    parser.add_argument(
        "--calibration-dir",
        help="directory of images used as INT8 calibration data (TensorRT --precision int8, "
        "or --quantize-level int8-static)",
    )
    parser.add_argument(
        "--quantize-level", choices=("fp16", "int8-dynamic", "int8-static"),
        help="apply post-training ONNX quantization after export (requires --target onnx)",
    )
    parser.add_argument("--quantize-output", help="output path for the quantized ONNX model")
    parser.add_argument(
        "--input-size", default="640x640",
        help="HxW used for INT8-static/TensorRT calibration preprocessing (default: 640x640)",
    )
    args = parser.parse_args(argv)

    metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
    artifact = Artifact(path=args.artifact, backend=args.backend, metadata=metadata)
    model = load_model(args.backend, args.model)
    input_size = _parse_input_size(args.input_size)

    if args.target == "tensorrt":
        if not args.output:
            parser.error("--output is required when --target tensorrt")
        from fabric_defect_hub.profiling.tensorrt import TensorRTBuildConfig, build_tensorrt_engine

        onnx = model.export(artifact, "onnx")
        calibrator = None
        if args.precision == "int8":
            if not args.calibration_dir:
                parser.error("--calibration-dir is required for --precision int8")
            from fabric_defect_hub.quantization import build_int8_calibrator

            calibrator = build_int8_calibrator(
                _samples_from_dir(args.calibration_dir), input_size=input_size
            )
        exported = build_tensorrt_engine(
            onnx, args.output, TensorRTBuildConfig(precision=args.precision), calibrator=calibrator
        )
    else:
        export_config = json.loads(args.export_config) if args.export_config else {}
        if "config" in inspect.signature(model.export).parameters:
            exported = model.export(artifact, args.target, config=export_config)
        else:
            if export_config:
                raise ValueError(f"Backend {args.backend!r} does not accept export configuration.")
            exported = model.export(artifact, args.target)

    if args.quantize_level:
        if exported.target != "onnx":
            parser.error("--quantize-level requires --target onnx")
        if not args.quantize_output:
            parser.error("--quantize-output is required with --quantize-level")
        from fabric_defect_hub.quantization import quantize_onnx

        calibration_samples = _samples_from_dir(args.calibration_dir) if args.calibration_dir else None
        exported = quantize_onnx(
            exported, args.quantize_level, args.quantize_output,
            calibration_samples=calibration_samples, input_size=input_size,
        )

    print(json.dumps(asdict(exported), indent=2, ensure_ascii=False))
    return 0


def _parse_input_size(text: str) -> tuple[int, int]:
    try:
        height_str, width_str = text.lower().split("x")
        return int(height_str), int(width_str)
    except ValueError as exc:
        raise ValueError(f"--input-size must look like 'HxW' (e.g. '640x640'), got {text!r}") from exc


def _samples_from_dir(directory: str) -> list:
    from fabric_defect_hub.core.types import Annotations, Sample

    root = Path(directory)
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"no calibration images found under {root} (expected {_IMAGE_SUFFIXES})")
    return [
        Sample(id=path.stem, image_path=str(path), task="anomaly", annotations=Annotations())
        for path in paths
    ]


if __name__ == "__main__":
    raise SystemExit(main())
