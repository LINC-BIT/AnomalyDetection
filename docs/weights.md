# Weight storage and distribution

Weights are immutable runtime artifacts and are not committed to Git. The current project-local cache is `textile/artifacts/models/published/<model-id>.<extension>`. A file without a manifest row is unsupported; a row without a file is reported as `missing`.

Public releases should use a dedicated Hugging Face Hub model repository with immutable revisions. Each entry should include repository ID, revision, filename, byte size, and SHA-256; download tooling must verify the checksum before publication. A project cache is preferable to a generic Python library cache because it keeps deployment provenance explicit. Google Drive is legacy/manual import only, including the current MoECLIP source.

The UI never downloads implicitly. Existing local weights work offline, while missing files show their exact expected path. Before release, record training dataset/split/domain, export and hash the canonical checkpoint, upload it to a versioned revision, update the manifest, and test a clean download through inference, evaluation, and UI smoke tests.
