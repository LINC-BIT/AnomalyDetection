"""Inference services, both shapes this project needs.

* `runner.py` -- the config-driven one-shot entry points (`run_predict` /
  `run_evaluate`), the inference mirror of `training.run_train`. This is what
  `fdh predict` / `fdh evaluate` and `api.predict` / `api.evaluate` call.
* `session.py` -- the resident-model one (`InferenceSessionManager`), which
  keeps a single loaded adapter alive across the web UI's clicks.

`runner.py` was a top-level `fabric_defect_hub/predict.py` until it collided
with the facade: `fdh.predict` is a function, but importing the submodule
`fabric_defect_hub.predict` rebinds that same attribute on the package to the
*module*, so `fdh.predict(...)` stopped being callable partway through a
session. Naming the module after what it is, one level down, removes the
collision rather than papering over it -- see
`tests/test_api_facade.py::test_facade_functions_survive_importing_every_submodule`.
"""

from fabric_defect_hub.inference.session import InferenceSessionManager, ModelNotLoadedError

__all__ = ["InferenceSessionManager", "ModelNotLoadedError"]
