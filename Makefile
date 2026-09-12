.PHONY: help install install-ui doctor inventory ui test test-architecture check

help:
	@echo "make install            Install the full project in editable mode"
	@echo "make install-ui         Install UI/inference dependencies"
	@echo "make doctor             Check backends, datasets, and weights"
	@echo "make inventory          Print the unified model/dataset manifest"
	@echo "make ui                 Launch the AnomalyDetection UI"
	@echo "make test               Run runtime tests"
	@echo "make test-architecture  Run source-boundary tests"
	@echo "make check              Run all pre-commit checks"

install:
	python -m pip install -r requirements-full.txt
	python -m pip install --no-deps --no-build-isolation -e .

install-ui:
	python -m pip install -r requirements.txt
	python -m pip install --no-deps --no-build-isolation -e .

doctor:
	adh doctor

inventory:
	adh inventory

ui:
	adh-ui

test:
	python -m pytest -q

test-architecture:
	python -m pytest -q -m architecture

check: test test-architecture
	git diff --check
