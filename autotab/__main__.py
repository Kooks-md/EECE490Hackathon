"""`python -m autotab DATASET_DIR OUTPUT_DIR` - same as `python main.py ...`."""

import os
import runpy
import sys

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.argv[0] = os.path.join(root, "main.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
