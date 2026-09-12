"""Hugging Face Spaces-compatible Gradio entry point."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fabric_defect_hub.web.app import launch


if __name__ == "__main__":
    launch()
