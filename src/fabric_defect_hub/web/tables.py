"""Renders `api.BenchmarkResult` into the shapes Gradio's table widgets take.

Strictly a *view*. Every question about what a metric is, which table it
belongs to, which direction is better, and how a value should be formatted is
answered by `metrics_taxonomy` through `BenchmarkResult`; this module only
decides how the answer is laid out on a page.

That boundary is the whole reason the file exists. The Benchmark tab used to
build its own column list by scanning whatever keys the leaderboard rows
happened to contain, which meant the page and the report could disagree
about where a metric belonged, and a newly measured metric appeared in a
table nobody had designed. Now a metric shows up on the page because the
taxonomy says which table it is in — or it does not show up at all and
`unrecognised_metrics()` says so out loud.

No charts: the cloud view is tables only, so a reader can copy numbers
straight into a paper without reading them off a plot.
"""

from __future__ import annotations

from typing import Any

# The two top-level sections, in display order, with the headings a page
# shows above them. The names come from the taxonomy; only the wording is
# this module's business.
SECTION_TITLES: dict[str, str] = {
    "technical": "Technical",
    "overhead": "Overhead",
}

TABLE_TITLES: dict[str, str] = {
    "image_level": "Image level",
    "pixel_level": "Pixel level",
    "instance_level": "Instance level",
    "cross_domain": "Cross-domain",
    "compute": "Compute",
    "memory": "Memory",
    "communication": "Communication",
}


def table_title(name: str) -> str:
    return TABLE_TITLES.get(name, name.replace("_", " ").title())


def render_table(table: Any) -> dict[str, Any]:
    """One taxonomy table as `{title, headers, rows, status, note}`.

    `headers`/`rows` are already strings — `MetricTable.as_matrix()` applies
    each metric's own unit and precision, so a page never re-formats a number
    and never has to know that latency is milliseconds while memory is MB.
    """

    return {
        "name": table.name,
        "title": table_title(table.name),
        "headers": table.header(),
        "rows": table.as_matrix(),
        "status": table.status,
        "note": table.note,
    }


def render_sections(result: Any) -> list[dict[str, Any]]:
    """The full page: two sections, each holding its tables in taxonomy order."""

    grouped = result.by_category()
    return [
        {
            "category": category,
            "title": SECTION_TITLES.get(category, category.title()),
            "tables": [render_table(table) for table in grouped.get(category, [])],
        }
        for category in SECTION_TITLES
    ]


def status_line(result: Any) -> str:
    """A one-line markdown summary under the run button.

    Reports unrecognised metrics because a measured-but-unplaced metric is
    invisible on a page made of named tables, and silence there would hide
    exactly the case this layering was built to prevent.
    """

    summary = result.summary()
    counts = summary.get("by_status", {})
    parts = [f"{count} {status}" for status, count in sorted(counts.items())]
    line = f"**{summary.get('total', 0)} measurements** — " + ", ".join(parts) if parts else "No measurements yet."

    stray = result.unrecognised_metrics()
    if stray:
        line += (
            f"\n\n:warning: {len(stray)} measured metric(s) have no table in "
            f"`metrics_taxonomy`: `{'`, `'.join(stray)}`. They were measured but are "
            "shown nowhere — add a `MetricSpec` for each."
        )
    if result.output_path:
        line += f"\n\nJSONL: `{result.output_path}`"
    return line


def empty_sections() -> list[dict[str, Any]]:
    """The page layout before any run — table headings exist, rows do not.

    Built from the taxonomy rather than hard-coded so the placeholder cannot
    drift from what a real run produces.
    """

    from fabric_defect_hub.api import list_metric_tables
    from fabric_defect_hub.metrics_taxonomy import EMPTY_HINTS, UNIMPLEMENTED

    declared = list_metric_tables()
    return [
        {
            "category": category,
            "title": SECTION_TITLES.get(category, category.title()),
            "tables": [
                {
                    "name": name, "title": table_title(name),
                    "headers": ["Model"], "rows": [], "status": "empty",
                    # The pre-run page already explains what each table needs,
                    # so a reader knows before pressing Run which of them the
                    # models they picked can actually fill.
                    "note": UNIMPLEMENTED.get(name) or EMPTY_HINTS.get(name, ""),
                }
                for name in declared.get(category, [])
            ],
        }
        for category in SECTION_TITLES
    ]
