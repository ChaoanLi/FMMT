"""
Entry point: reproduce the shear layer case study (Section 6, Figure 5).

Usage:
    python scripts/run_realdata.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from realdata.shear_layer import run_shear_layer


if __name__ == "__main__":
    run_shear_layer()
