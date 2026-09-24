#!/usr/bin/env bash
set -e

echo "== Bio-Attention Paper Pipeline =="

if [ -z "$VIRTUAL_ENV" ]; then
  echo "Warning: No virtual environment detected."
  echo "Please activate a venv before running."
fi

echo "Running experiments..."
python experiments/run_experiments.py --mode full

echo "Generating figures..."
python generate_manuscript.py --figures-only

echo "Compiling manuscript..."
python generate_manuscript.py --compile-pdf

echo "Done."
