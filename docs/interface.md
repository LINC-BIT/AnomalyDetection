# Unified interface contract

`ModelAdapter` defines `train(config)`, `predict(samples, artifact, output_dir, config)`, `export(artifact, target, config)`, `load_trained_model(artifact)`, and `capabilities()`. Capabilities truthfully enumerate tasks, prediction fields, and export targets.

`DatasetAdapter` converts a source dataset into common `Sample` objects. Models return common `Prediction` objects. Evaluators consume those contracts and never import a backend.

The backend registry answers how an implementation runs. The published-model manifest answers which concrete model is offered, which method and integration it uses, where it was trained/evaluated, and where its weight resolves. Application services join these layers. CLI and UI call application services; the web package owns presentation only.

Runtime contracts run with `pytest`. Source-boundary audits run with `pytest -m architecture`.
