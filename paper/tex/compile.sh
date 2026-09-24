#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

echo "PDF generated: tex/main.pdf"
