#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo " Mauritanian Livestock Ethno-Toxicology — Full Reproduction Script"
echo "=================================================================="

python3 -m pip install --upgrade pip
pip install -r requirements.txt

python3 src/run_full_pipeline.py --seed 42
python3 tests/test_pipeline.py

echo "Done! All tables (outputs/tables/) and 12 300 DPI figures (outputs/figures/) reproduced."
