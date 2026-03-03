"""
Run all notebooks in sequence using papermill.

Usage: python scripts/run_papermill.py
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

try:
    import papermill as pm
except ImportError:
    print("papermill not installed. Run:")
    print("  pip install papermill")
    sys.exit(1)


NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocess_feature.ipynb",
    "notebooks/03_mining_or_clustering.ipynb",
    "notebooks/04_modeling.ipynb",
    "notebooks/04b_semi_supervised.ipynb",
    "notebooks/05_evaluation_report.ipynb",
]

OUTPUT_DIR = "outputs/reports"


def main():
    """Execute all notebooks in order."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_start = time.time()

    print("=" * 60)
    print("  RUNNING NOTEBOOKS WITH PAPERMILL")
    print("=" * 60)

    for nb_path in NOTEBOOKS:
        if not os.path.exists(nb_path):
            print(f"\n[SKIP] {nb_path} (not found)")
            continue

        nb_name = os.path.basename(nb_path)
        out_path = os.path.join(OUTPUT_DIR, nb_name)

        print(f"\n[RUN] {nb_path}")
        start = time.time()

        try:
            pm.execute_notebook(
                nb_path,
                out_path,
                kernel_name="python3",
            )
            elapsed = time.time() - start
            print(f"  [OK] {elapsed:.1f}s -> {out_path}")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  [FAIL] {elapsed:.1f}s: {e}")

    total_elapsed = time.time() - total_start
    print(f"\n[DONE] All notebooks in {total_elapsed:.1f}s")
    print(f"  Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
