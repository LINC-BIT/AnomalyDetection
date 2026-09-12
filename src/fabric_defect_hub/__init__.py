"""AnomalyDetection: unified multi-domain detection and anomaly SDK.

Provides per-model config profiles, adaptive loading strategies, metric
evaluators, and reporting tools.

Quick Usage:
    import fabric_defect_hub as fdh

    fdh.list_models("anomalib")                      # what can I run?
    cfg = fdh.load_config("stfpm", dataset="zju-leaper", epochs=50)
    run = fdh.train(cfg)
    out = fdh.predict("stfpm", weights=run.result.registered_artifact.path,
                      source="sample.jpg")

    weights = fdh.from_pretrained("PatchCore")       # a published checkpoint
    fdh.evaluate("PatchCore", weights=weights, dataset="tilda-400")

That is the front door (see `api.py`). Underneath it sit the three
abstractions everything else is built from, exported here too for callers
who want to compose runs themselves rather than drive a config:

    ModelAdapter / ModelCapabilities   a model backend  (models/base.py)
    DataAdapter / BatchSpec            Samples -> batches (core/data_adapter.py)
    TrainConfig                        the shared hyperparameter vocabulary

plus `load_dataset` / `load_model` / `run_experiment` for assembling a run
by hand. See `docs/interface.md` for the contracts.
"""

from fabric_defect_hub import recipes
from fabric_defect_hub.api import (
    BenchmarkResult,
    PretrainedWeights,
    RunConfig,
    evaluate,
    from_pretrained,
    list_datasets,
    list_metric_tables,
    list_models,
    list_pretrained,
    load_config,
    load_results,
    measure,
    predict,
    train,
)
from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.data_adapter import BatchSpec, DataAdapter, Normalization
from fabric_defect_hub.core.registry import get_recipe, list_recipes
from fabric_defect_hub.core.train_config import TrainConfig
from fabric_defect_hub.evaluation.lmei_profiler import calculate_lmei
from fabric_defect_hub.evaluation.pro_calculator import compute_pro_score
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment
from fabric_defect_hub.models.base import ModelAdapter, ModelCapabilities
from fabric_defect_hub.reporting.latex_generator import generate_latex_table
try:
    from fabric_defect_hub.strategies.loader_strategies import (
        BatchNormCalibrator,
        SlidingWindowTiler,
        SparseSubsampler,
        TTAInferenceWrapper,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    # Catalog/configuration APIs remain usable with the documented core install.
    # These four strategies become available after a training backend installs torch.
    BatchNormCalibrator = None  # type: ignore[assignment]
    SlidingWindowTiler = None  # type: ignore[assignment]
    SparseSubsampler = None  # type: ignore[assignment]
    TTAInferenceWrapper = None  # type: ignore[assignment]

__version__ = "0.2.0"

__all__ = [
    # -- the front door (api.py) --------------------------------------
    "load_config",
    "load_dataset",
    "load_model",
    "from_pretrained",
    "train",
    "predict",
    "evaluate",
    "list_models",
    "list_datasets",
    "list_pretrained",
    "RunConfig",
    "PretrainedWeights",
    # -- measurement (metric_sweep + metrics_taxonomy) -----------------
    # `measure`, not `benchmark`: `fabric_defect_hub.benchmark` is a module,
    # and a same-named facade function would be shadowed on import.
    "measure",
    "load_results",
    "list_metric_tables",
    "BenchmarkResult",
    # -- composing a run by hand --------------------------------------
    "run_experiment",
    "recipes",
    "BaseModelRecipe",
    "get_recipe",
    "list_recipes",
    # The model/data contracts (see docs/EXTENDING.md)
    "ModelAdapter",
    "ModelCapabilities",
    "TrainConfig",
    "DataAdapter",
    "BatchSpec",
    "Normalization",
    "compute_pro_score",
    "calculate_lmei",
    "generate_latex_table",
    "SparseSubsampler",
    "SlidingWindowTiler",
    "TTAInferenceWrapper",
    "BatchNormCalibrator",
]
