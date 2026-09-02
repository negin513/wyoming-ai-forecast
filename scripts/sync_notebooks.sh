#!/usr/bin/env bash
# Copies the root-level notebooks into docs/ so mkdocs-jupyter can render them.
# Copies (not symlinks) because mkdocs-jupyter rewrites the file it converts;
# a symlink would corrupt the source notebook in place.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cp 01_wyoming_ai_forecast.ipynb docs/01_wyoming_ai_forecast.ipynb
cp 02_stormscope_wyoming_demo.ipynb docs/02_stormscope_wyoming_demo.ipynb
