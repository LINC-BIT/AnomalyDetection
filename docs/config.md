# Configuration

`configs/registry/models.yaml` declares concrete published models and their provenance. `configs/models/*.yaml` declares executable training and inference parameters. The manifest points to a recipe by filename; runtime code resolves it under `configs/models/` unless an explicit path is supplied.

Environment variables may override dataset roots, but repository-relative categorized defaults remain available. Secrets and machine-specific absolute paths must never enter committed YAML. Domain overlays may change dataset or artifact parameters, but cannot redefine backend lifecycle behavior.

Validate changes with `adh inventory`, `adh doctor`, and the configuration tests.
