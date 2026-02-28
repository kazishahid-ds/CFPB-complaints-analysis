#!/usr/bin/env python
"""
Standalone script to execute 03b and 03c segmented notebooks
directly in the conda environment to validate the workflows.
"""
import sys
import subprocess

notebooks = [
    "notebooks/03b_cfpb_time_series_xgboost.ipynb",
    "notebooks/03c_cfpb_time_series_neuralprophet.ipynb"
]

for nb in notebooks:
    print(f"\n{'='*60}")
    print(f"Running: {nb}")
    print('='*60)
    
    # Use nbconvert with explicit python interpreter
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", nb,
        "--output-dir=notebooks",
        "--no-input"
    ]
    
    result = subprocess.run(cmd, cwd=".")
    if result.returncode != 0:
        print(f"ERROR: {nb} failed with exit code {result.returncode}")
    else:
        print(f"SUCCESS: {nb} completed")

print("\n" + "="*60)
print("All notebooks executed")
print("="*60)
