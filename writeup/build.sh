#!/bin/bash
# Build the PDF write-up from docs/results.md: check the pages, refresh the run table from analysis/results,
# then pandoc -> PDF, with the vector PDF figures in place of the PNGs that GitHub shows.
set -euo pipefail
cd "$(dirname "$0")"
python3 check_docs.py
python3 make_run_table.py
cd ../docs
pandoc results.md -f markdown-implicit_figures \
  --metadata-file ../writeup/pdf_meta.yaml --shift-heading-level-by=-1 \
  --lua-filter ../writeup/png2pdf.lua --pdf-engine=xelatex \
  -o ../writeup/sprott_writeup.pdf
echo "write-up -> $(cd ../writeup && pwd)/sprott_writeup.pdf"
