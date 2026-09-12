# ADR-0001: Unified registry, adapters, and presentation-only UI

Status: Accepted
Date: 2026-09-12

The platform combines multiple execution frameworks, research checkouts, and application domains. `ModelAdapter` is the execution contract; `DatasetAdapter`, `Sample`, and `Prediction` are data contracts; `configs/registry/models.yaml` is the published-model source of truth; and `application/` is the only supported business boundary for CLI and UI. External research components are pinned Git submodules under root `components/`. Domain directories own data, overlays, examples, and weights, never reusable algorithms. Consequently, adding a conforming manifest entry updates all consumers without model-specific UI code.
