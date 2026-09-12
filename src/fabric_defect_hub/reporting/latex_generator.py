r"""Paper-Grade LaTeX Table Generator Engine for IEEE/CVPR Benchmark Submissions.

Renders benchmark and model evaluation results directly into publication-ready
LaTeX code with automatic best/second-best bolding (\textbf{}) and underlining (\underline{}).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def generate_latex_table(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    caption: str = "Quantitative Comparison of Fabric Defect Inspection Models under Unified Benchmark Protocols (UTAD-Framework).",
    label: str = "tab:fabric_benchmark",
) -> str:
    """Generates LaTeX table string formatted for IEEE / CVPR / ICCV templates.

    Args:
        results: List of dictionaries, each containing 'model', 'recipe', and metric scores.
        metrics: List of metric keys to display (e.g., ['I-AUROC', 'P-AUROC', 'PRO-Score', 'FPS', 'LMEI'])
        caption: Table caption for paper
        label: LaTeX cross-reference label

    Returns:
        str: LaTeX source code string
    """
    if not results:
        return "% Empty benchmark results provided."

    if metrics is None:
        metrics = ["I-AUROC", "P-AUROC", "PRO-Score", "FPS", "LMEI"]

    header_cols = ["Model", "Recipe / Acronym"] + metrics
    col_spec = "l c " + " ".join(["c"] * len(metrics))

    # Find max and second max for bolding/underlining
    best_vals: Dict[str, float] = {}
    second_best_vals: Dict[str, float] = {}

    for m in metrics:
        vals = [r.get(m, 0.0) for r in results if isinstance(r.get(m), (int, float))]
        if vals:
            sorted_vals = sorted(list(set(vals)), reverse=True)
            best_vals[m] = sorted_vals[0]
            second_best_vals[m] = sorted_vals[1] if len(sorted_vals) > 1 else -999.0

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_cols) + " \\\\",
        "\\midrule",
    ]

    for row in results:
        model_name = row.get("model", "Unknown")
        recipe_name = row.get("recipe", "Standard")

        formatted_cells = [model_name, recipe_name]
        for m in metrics:
            val = row.get(m, 0.0)
            if isinstance(val, float):
                val_str = f"{val:.4f}"
                if val == best_vals.get(m):
                    val_str = f"\\textbf{{{val_str}}}"
                elif val == second_best_vals.get(m):
                    val_str = f"\\underline{{{val_str}}}"
            else:
                val_str = str(val)
            formatted_cells.append(val_str)

        latex_lines.append(" & ".join(formatted_cells) + " \\\\")

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )

    return "\n".join(latex_lines)
