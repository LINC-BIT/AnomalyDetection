"""The UI must be a *reader* of the backend's measurement layer, never a
second implementation of it.

The requirement in the user's words: the front end cannot be self-contained.
Running `fdh.benchmark(...)` from a script and opening the Benchmark tab must
produce the same tables, because they are the same object — otherwise a
metric can be measured and shown under a heading the report disagrees with,
or shown on the page and missing from the JSON, and nobody finds out until a
number in a paper cannot be traced.

These are AST/structural checks, so they fail on the commit that reintroduces
the coupling rather than on the next person to read the code.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import fabric_defect_hub as fdh
from fabric_defect_hub import api
from fabric_defect_hub.metrics_taxonomy import OVERHEAD_TABLES, TABLES, TECHNICAL_TABLES
from fabric_defect_hub.web import tables as web_tables

_WEB_DIR = Path(inspect.getfile(web_tables)).parent


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _string_constants(tree: ast.Module) -> set[str]:
    return {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


# --------------------------------------------------------------------------- #
# The library entry point exists and is importable without a framework
# --------------------------------------------------------------------------- #
def test_the_package_root_exposes_the_measurement_entry_points():
    """`import fabric_defect_hub` must reach measurement the same way it
    reaches training — the user drives this from code, not the CLI."""

    for name in ("measure", "load_results", "list_metric_tables", "BenchmarkResult"):
        assert hasattr(fdh, name), name
        assert name in fdh.__all__


def test_listing_tables_needs_no_measurement_and_no_backend():
    declared = fdh.list_metric_tables()

    assert declared["technical"] == list(TECHNICAL_TABLES)
    assert declared["overhead"] == list(OVERHEAD_TABLES)


def test_a_result_loaded_from_disk_and_one_just_measured_are_the_same_type():
    """`load_results` is what lets a page render without re-measuring."""

    assert inspect.signature(api.measure).return_annotation == "BenchmarkResult"
    assert inspect.signature(api.load_results).return_annotation == "BenchmarkResult"


# --------------------------------------------------------------------------- #
# The UI owns no metric knowledge
# --------------------------------------------------------------------------- #
# `web/benchmark.py` is a measurement *driver* that predates `metric_sweep`
# and still names metrics itself (headline-metric selection, profiling
# columns). It is not a renderer, so this guard does not cover it — but the
# duplication is real and is tracked, not waived: it should eventually run
# through `metric_sweep` like everything else.
_PRESENTATION_MODULES = ("tables.py", "app.py", "single_image.py")


@pytest.mark.architecture
@pytest.mark.parametrize("filename", _PRESENTATION_MODULES)
def test_no_presentation_module_hard_codes_a_metric_name(filename):
    """A metric name spelled in a rendering module is the UI deciding what to
    show, which is how a measured metric ends up in a table nobody designed.
    """

    from fabric_defect_hub.metrics_taxonomy import METRIC_SPECS

    path = _WEB_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    found = _string_constants(_parse(path)) & {spec.key for spec in METRIC_SPECS}

    assert not found, (
        f"web/{filename} names metrics directly: {sorted(found)}. Ask "
        "`metrics_taxonomy` which table a metric belongs to instead."
    )


@pytest.mark.architecture
def test_the_renderer_holds_titles_but_no_metric_mapping():
    """`web/tables.py` may title a table; it may not decide its contents.

    The distinction is checked structurally: a module-level dict whose values
    are metric keys would be a second taxonomy, whereas the title maps are
    keyed by table name and hold prose.
    """

    from fabric_defect_hub.metrics_taxonomy import METRIC_SPECS

    metric_keys = {spec.key for spec in METRIC_SPECS}
    tree = _parse(_WEB_DIR / "tables.py")
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        literals = {
            item.value for item in [*value.keys, *value.values]
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        assert not literals & metric_keys, (
            f"web/tables.py declares a metric->something map containing "
            f"{sorted(literals & metric_keys)}; that mapping belongs in metrics_taxonomy."
        )


def test_the_renderer_reads_its_table_list_from_the_taxonomy():
    """`empty_sections` must not carry its own copy of the table names."""

    sections = {section["category"]: section for section in web_tables.empty_sections()}

    assert [t["name"] for t in sections["technical"]["tables"]] == list(TECHNICAL_TABLES)
    assert [t["name"] for t in sections["overhead"]["tables"]] == list(OVERHEAD_TABLES)


# --------------------------------------------------------------------------- #
# Page and script render the same thing
# --------------------------------------------------------------------------- #
def test_the_page_and_a_script_render_identical_tables():
    """The equivalence the whole layering exists for."""

    rows = [{
        "model": "yolov8n", "status": "ok",
        "metrics": {"map_50": 0.42, "fps": 31.5, "peak_memory_mb": 900.0},
    }]
    result = api.BenchmarkResult(rows=rows)

    from_api = {table.name: table.as_matrix() for table in result.tables()}
    from_page = {
        table["name"]: table["rows"]
        for section in web_tables.render_sections(result)
        for table in section["tables"]
    }

    assert from_page == from_api


def test_the_page_surfaces_metrics_the_taxonomy_cannot_place():
    """A measured-but-unplaced metric is invisible in a page of named
    tables, so the status line has to say it out loud."""

    result = api.BenchmarkResult(rows=[{
        "model": "m", "status": "ok", "metrics": {"brand_new_metric": 1.0},
    }])

    assert result.unrecognised_metrics() == ["brand_new_metric"]
    assert "brand_new_metric" in web_tables.status_line(result)


def test_the_unimplemented_table_reaches_the_page_with_its_reason():
    rendered = {
        table["name"]: table
        for section in web_tables.render_sections(api.BenchmarkResult())
        for table in section["tables"]
    }

    assert rendered["communication"]["status"] == "not_implemented"
    assert "Bandwidth saving" in rendered["communication"]["note"]


# --------------------------------------------------------------------------- #
# Charts are gone
# --------------------------------------------------------------------------- #
def test_the_charts_module_is_gone():
    assert not (_WEB_DIR / "charts.py").exists()


@pytest.mark.architecture
def test_no_web_module_renders_a_plot():
    """Cloud display is tables only — numbers a reader can copy into a paper."""

    banned = {"BarPlot", "LinePlot", "ScatterPlot", "render_radar_svg", "matplotlib"}
    offenders: dict[str, set[str]] = {}
    for path in sorted(_WEB_DIR.glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        found = {
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in banned
        } | {
            name for name in banned
            if any(
                isinstance(node, ast.Name) and node.id == name for node in ast.walk(tree)
            )
        }
        if found:
            offenders[path.name] = found
    assert not offenders, f"web modules still plot: {offenders}"


# --------------------------------------------------------------------------- #
# JSON export
# --------------------------------------------------------------------------- #
def test_to_json_writes_the_two_part_structure(tmp_path):
    import json

    result = api.BenchmarkResult(rows=[{
        "model": "m", "status": "ok", "metrics": {"fps": 10.0, "map_50": 0.3},
    }])

    path = result.to_json(tmp_path / "report.json")
    payload = json.loads(path.read_text())

    assert set(payload["categories"]) == {"technical", "overhead"}
    names = [table["name"] for table in payload["categories"]["overhead"]]
    assert names == list(OVERHEAD_TABLES)
    compute = next(t for t in payload["categories"]["overhead"] if t["name"] == "compute")
    assert compute["header"] == ["Model", "FPS (aggregate)"]
    assert compute["rows"] == [["m", "10"]]
