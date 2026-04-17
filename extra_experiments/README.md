# Extra Experiments

This directory contains experimental code **not included in the paper**.
These are provided for reference only and are not needed to reproduce the
published results.

## Contents

| File | Description |
|------|-------------|
| `pjg2015.py` | Pardo, Juan-Verdejo & Garcia-Escudero (2015) characteristic-function test |
| `comparison_pjg2015.py` | Comparison experiment with PJG2015 baseline |

## Why excluded from paper

PJG2015 is a two-sample test that requires bootstrap calibration and does
not target the one-sample model validation setting studied in the paper.
We include it here for reproducibility of exploratory analyses.

## Usage

These scripts import from `code/` and can be run from the repository root:

```bash
python extra_experiments/comparison_pjg2015.py --quick
```
