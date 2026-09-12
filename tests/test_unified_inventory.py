import ast
from pathlib import Path

from fabric_defect_hub.application import model_inventory, platform_inventory


def test_all_canonical_models_have_classification_and_integration():
    rows = model_inventory()
    assert len(rows) == 20
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["category"] for row in rows} == {
        "supervised_defect_detection",
        "supervised_defect_segmentation",
        "anomaly_detection",
    }
    assert {row["integration"] for row in rows} == {
        "ultralytics", "torchvision", "anomalib", "component", "mambaad",
    }


def test_component_models_point_to_component_integration():
    rows = {row["id"]: row for row in model_inventory()}
    assert rows["Dinomaly"]["integration"] == "component"
    assert rows["MoECLIP"]["integration"] == "component"


def test_inventory_uses_domain_owned_textile_weights():
    for row in model_inventory():
        assert Path(row["weight"]).parts[-5:-3] == ("textile", "artifacts")


def test_platform_inventory_is_the_shared_serializable_contract():
    inventory = platform_inventory()
    assert len(inventory["models"]) == 20
    assert len(inventory["datasets"]) == 9


def test_every_published_model_declares_training_provenance():
    for row in model_inventory():
        assert row["method"]
        assert row["domain"] in {"textile", "general"}
        assert row["trained_on"]
    moeclip = next(row for row in model_inventory() if row["id"] == "MoECLIP")
    assert moeclip["trained_on"] == ["mvtec-ad"]
    assert moeclip["evaluated_on"] == ["zju-leaper"]


def test_model_manifest_is_the_single_catalog_source():
    manifest = Path(__file__).parents[1] / "configs" / "registry" / "models.yaml"
    assert manifest.is_file()
    assert "CANONICAL_MODELS: list[CanonicalModel] = _load_models()" in (
        Path(__file__).parents[1] / "src" / "fabric_defect_hub" / "catalog.py"
    ).read_text(encoding="utf-8")


def test_web_layer_does_not_import_backend_business_logic():
    """Keep Gradio as a view over the application API, never an executor."""
    web_root = Path(__file__).parents[1] / "src" / "fabric_defect_hub" / "web"
    forbidden = {
        "fabric_defect_hub.catalog",
        "fabric_defect_hub.core",
        "fabric_defect_hub.evaluation",
        "fabric_defect_hub.loader",
        "fabric_defect_hub.models",
        "fabric_defect_hub.training",
    }
    violations = []
    for source_path in web_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        for module in imports:
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{source_path.name}: {module}")
    assert violations == []
