# Textile domain

This directory owns textile-specific assets only: `artifacts/`, configuration overlays, adapter exports, examples, and `domain.yaml`. Reusable algorithms, training, inference, evaluation, and UI services live under repository-level `src/`.

Raw textile datasets live in `datasets/textile/`; general auxiliary datasets live in `datasets/general/`. Published weights are textile-specific unless the model manifest explicitly declares external pretraining or cross-domain training. Run `adh inventory`, `adh doctor`, and `adh-ui` from the repository root.
