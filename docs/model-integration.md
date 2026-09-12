# Model integration

All models enter one lifecycle regardless of source. The web layer must never import Ultralytics, torchvision, Anomalib, or a component checkout.

## Supported integration types

- **Anomalib:** register the real class and validated constructor parameters, then select it in YAML. Do not copy Anomalib implementations.
- **torchvision:** register a factory, task, checkpoint loader, and common prediction conversion.
- **Ultralytics:** register the variant and translate its result objects into common predictions.
- **Component:** register the upstream repository as a Git submodule at `components/<name>`, isolate imports in `vendor.py`, use a typed configuration object, and wrap upstream train/inference functions in an adapter. Users initialize it with `git submodule update --init --recursive`; they do not clone it separately.
- **Native:** implement a normal package under `src/.../models/` and the same adapter contract.

## Publish checklist

1. Add or reuse an adapter and declare truthful capabilities.
2. Add a recipe under `configs/models/`.
3. Add exactly one row to `configs/registry/models.yaml`, including method, domain, training dataset/split, optional evaluation datasets, and weight source.
4. Put the canonical checkpoint in the declared project cache or fetch and verify it from the configured distribution.
5. Test loading, prediction shape, missing-weight behavior, CLI inventory, and UI creation.

No model-specific UI edit is permitted. If the manifest row is complete, every frontend consumer receives it through the application inventory.
