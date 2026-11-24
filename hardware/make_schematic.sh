#!/bin/bash
# Draw the schematic: sprott_schematic.tex -> sprott_schematic.pdf (vector) and .png (200 dpi, for GitHub).
set -euo pipefail
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error sprott_schematic.tex > /dev/null
rm -f sprott_schematic.aux sprott_schematic.log
pdftoppm -png -r 200 -singlefile sprott_schematic.pdf sprott_schematic
pdftoppm -png -r 100 -singlefile sprott_schematic.pdf sprott_schematic_preview
echo "schematic -> $(pwd)/sprott_schematic.pdf and .png"
