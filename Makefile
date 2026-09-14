.PHONY: help install install-ui install-train install-eval doctor inventory ui test test-architecture check

help:
	@echo "make install            Install everything (detects CUDA automatically)"
	@echo "make install-ui         Install the web interface / inference dependencies"
	@echo "make install-train      Install the training backends (detects CUDA automatically)"
	@echo "make install-eval       Install the evaluation / profiling dependencies"
	@echo "make doctor             Check backends, datasets, and weights"
	@echo "make inventory          Print the unified model/dataset manifest"
	@echo "make ui                 Launch the AnomalyDetection UI"
	@echo "make test               Run runtime tests"
	@echo "make test-architecture  Run source-boundary tests"
	@echo "make check              Run all pre-commit checks"

install:
	python tools/install_deps.py all

install-ui:
	python tools/install_deps.py ui

install-train:
	python tools/install_deps.py train

install-eval:
	python tools/install_deps.py eval

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
