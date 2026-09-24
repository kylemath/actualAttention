# Makefile for paper experiments and compilation

.PHONY: all setup experiments figures paper clean help

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Default target
all: setup experiments figures paper

help:
	@echo "Available targets:"
	@echo "  setup       - Create virtual environment and install dependencies"
	@echo "  experiments - Run experiments in cycle mode"
	@echo "  experiments-fast - Run experiments in fast mode (20 epochs)"
	@echo "  experiments-full - Run experiments in full mode (100 epochs)"
	@echo "  figures     - Generate all figures"
	@echo "  paper       - Compile LaTeX paper"
	@echo "  clean       - Remove generated files"
	@echo "  clean-data  - Remove experiment data"
	@echo "  clean-figs  - Remove generated figures"

# Setup virtual environment
setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch numpy matplotlib seaborn scipy

# Run experiments
experiments: setup
	$(PYTHON) experiments/run_experiments.py --mode cycle --tasks copy --archs rnn lstm transformer

experiments-parallel: setup
	$(PYTHON) experiments/run_parallel.py --mode parallel --tasks copy --archs rnn lstm transformer

experiments-fast: setup
	$(PYTHON) experiments/run_experiments.py --mode fast --tasks copy --archs rnn lstm transformer

experiments-full: setup
	$(PYTHON) experiments/run_experiments.py --mode full --tasks copy --archs rnn lstm transformer

# Generate figures
figures: setup
	$(PYTHON) experiments/generate_figures.py

# Compile paper
paper:
	cd tex && pdflatex -interaction=nonstopmode main.tex
	cd tex && bibtex main || true
	cd tex && pdflatex -interaction=nonstopmode main.tex
	cd tex && pdflatex -interaction=nonstopmode main.tex
	@echo "Paper compiled: tex/main.pdf"

# Clean targets
clean: clean-tex

clean-tex:
	cd tex && rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot

clean-data:
	rm -f data/*.pkl data/*.json

clean-figs:
	rm -rf figures/cycle*
	rm -f figures/*.pdf figures/*.png

# Watch and rebuild (requires fswatch)
watch:
	fswatch -o tex/*.tex | xargs -n1 -I{} make paper
