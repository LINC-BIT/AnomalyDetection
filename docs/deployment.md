# Deployment

## Local UI

Activate `anomalib_env`, install the root package editable, run `adh doctor`, then launch `adh-ui`. The UI is offline by default and never downloads a missing checkpoint implicitly.

## Batch inference

Use the canonical model ID, explicit input, and manifest-resolved checkpoint. Persist the inventory and provenance block with evaluation outputs so a result can be reproduced.

## Release checks

Verify dataset availability, checkpoint identity, backend importability, expected prediction fields, target-device profiling, and a rollback checkpoint. Run both default and architecture test suites. Never infer deployability merely because an adapter exists; the published weight must be present and compatible.
