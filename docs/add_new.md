# Adding datasets, domains, and backends

## Dataset

Implement `DatasetAdapter.load_samples()`, register the adapter, and declare its tasks, roles, and default root in the dataset-capability registry. Return only common `Sample` objects. Add contract tests for normal, anomalous, missing, and malformed inputs.

## Domain

Create `datasets/<domain>/`, `<domain>/domain.yaml`, optional overlays and examples, and a domain-owned artifact directory. Reuse root algorithms. Publish independently trained weights with explicit `domain` and `trained_on` metadata.

## Backend

Implement the lifecycle documented in [model integration](model-integration.md). Add a recipe and one manifest row. Run `adh inventory`, `adh doctor`, default tests, and architecture tests. A new model must not require a web-layer change.
