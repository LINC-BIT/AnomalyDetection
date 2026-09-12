# Contributing

Keep algorithms domain-independent and domain assets in their domain directory. A backend implements `ModelAdapter`; a dataset implements `DatasetAdapter` and declares capabilities; a publishable model has exactly one row in `configs/registry/models.yaml`.

Run `pytest -q`, `pytest -q -m architecture`, and `git diff --check` before submission. Do not commit datasets, weights, results, environments, caches, credentials, or machine-specific paths. Preserve third-party licenses and provenance.
