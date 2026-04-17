# FMMT: Fourier Maximum Modulus Test for Computer Model Validation

Supplementary code for:

> **Statistical Validation of Computer Models: Global and Subdomain Hypothesis Testing**  
> Chaoan Li, Xianyang Zhang, Rui Tuo  
> (submitted)

---

## Overview

This repository implements the Fourier Maximum Modulus Test (FMMT) for
validating computer simulation models against physical data.  The method
leverages kernel ridge regression (KRR) and generalized Fourier coefficients
to test for functional discrepancies globally or over user-specified
subdomains.

---

## Repository Structure

```
ModelValidator/
├── code/               # Core methodology (Section 3)
│   ├── kernel_ridge.py      Matern KRR with 5-fold CV (Section 3.1)
│   ├── fourier_basis.py     Fourier basis functions and decay weights (Section 3.3)
│   ├── kde.py               Epanechnikov KDE with boundary correction (Section 3.3)
│   ├── statistical_tests.py FMMT test statistics and p-values (Section 3.2)
│   ├── functions.py         Simulation scenario functions
│   ├── data_generation.py   Sample generation
│   ├── utils.py             Reproducibility utilities
│   ├── nd2003.py            Neumeyer & Dette (2003) baseline
│   └── eubank_hart1992.py   Eubank & Hart (1992) baseline
├── simulation/         # Simulation studies (Section 5)
│   ├── comparison.py        Table 1: FMMT vs ND vs EH (Section 5.1)
│   ├── sim_1d_global.py     Figure 1 + Table 2: 1D global (Section 5.2)
│   ├── sim_1d_subdomain.py  Figure 2: 1D subdomain (Section 5.2)
│   ├── sim_2d_global.py     Figure 3 + Table 3: 2D global (Section 5.2)
│   ├── sim_2d_subdomain.py  Figure 4: 2D subdomain (Section 5.2)
│   └── visualize.py         Publication figure generator
├── realdata/           # Case study (Section 6)
│   └── shear_layer.py       Figure 5: shear layer experiment
├── scripts/            # Convenience entry points
│   ├── run_simulations.py   Run all or individual simulation steps
│   └── run_realdata.py      Run the shear layer analysis
├── results/            # Outputs (reproducible)
│   ├── figures/             Figures 1-5 (PNG)
│   └── tables/              Tables 1-3 (CSV)
├── data/               # Data (embedded in realdata/shear_layer.py)
├── extra_experiments/  # Code not in paper (PJG2015 comparison)
├── run_all.py          # Single command to reproduce all results
└── requirements.txt
```

---

## Paper-to-Code Mapping

| Paper section | Code file |
|---|---|
| §3.1 KRR (Eq. 1–2, Matérn kernel) | `code/kernel_ridge.py` |
| §3.2 FMMT test statistics (Eq. 3–4) | `code/statistical_tests.py` |
| §3.3 Fourier basis + decay weights (Eq. 5–6) | `code/fourier_basis.py` |
| §3.3 KDE with boundary correction | `code/kde.py` |
| §5.1 Comparison (Table 1) | `simulation/comparison.py` |
| §5.2 1D global (Figure 1, Table 2) | `simulation/sim_1d_global.py` |
| §5.2 1D subdomain (Figure 2) | `simulation/sim_1d_subdomain.py` |
| §5.2 2D global (Figure 3, Table 3) | `simulation/sim_2d_global.py` |
| §5.2 2D subdomain (Figure 4) | `simulation/sim_2d_subdomain.py` |
| §6 Shear layer (Figure 5) | `realdata/shear_layer.py` |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Reproducing All Paper Results

### Full reproduction (1000 Monte Carlo replications per scenario)

```bash
python run_all.py
```

Expected runtime: several hours on a modern multi-core workstation.

### Quick validation (50 replications — for checking the pipeline)

```bash
python run_all.py --quick
```

Expected runtime: ~10–30 minutes.

### Individual components

```bash
# Table 1 only (Section 5.1)
python -m simulation.comparison

# Figure 1 + Table 2 (1D global, Section 5.2)
python -m simulation.sim_1d_global
python -m simulation.visualize --step 1

# Figure 2 (1D subdomain, Section 5.2)
python -m simulation.sim_1d_subdomain
python -m simulation.visualize --step 2

# Figure 3 + Table 3 (2D global, Section 5.2)
python -m simulation.sim_2d_global
python -m simulation.visualize --step 3

# Figure 4 (2D subdomain, Section 5.2)
python -m simulation.sim_2d_subdomain
python -m simulation.visualize --step 4

# Figure 5 (shear layer, Section 6)
python -m realdata.shear_layer
```

---

## Outputs

| File | Content | Paper reference |
|------|---------|----------------|
| `results/tables/table1_comparison.csv` | Type I error and power for FMMT, ND, EH | Table 1 |
| `results/figures/step1_power_curves.png` | 1D global power curves | Figure 1 |
| `results/tables/step1_type_i_error.csv` | 1D global Type I error | Table 2 |
| `results/figures/step2_power_curves.png` | 1D subdomain power curves | Figure 2 |
| `results/figures/step3_power_curves.png` | 2D global power curves | Figure 3 |
| `results/tables/step3_type_i_error.csv` | 2D global Type I error | Table 3 |
| `results/figures/step4_power_curves.png` | 2D subdomain power curves | Figure 4 |
| `results/figures/shear_layer_results.png` | Shear layer analysis | Figure 5 |
| `results/tables/shear_layer_pvalues.csv` | Shear layer p-values | Section 6 |

---

## Reproducibility Notes

- All random seeds default to 42.  Pass `--seed N` to change.
- All simulations use 1000 Monte Carlo replications by default.
- KRR uses `nu=3.5`, `theta=1.0`, `lambda_n = C_n / n` with `C_n` chosen
  by 5-fold CV over a log-spaced grid from `10^{-9}` to `1`.
- Noise variance is estimated using effective degrees of freedom (Eq. after Eq. 2 in paper).
- KDE uses Epanechnikov kernel with boundary correction and GCV bandwidth.
- Decay parameter `ell=0.7` (`weight_decay_n=0.7`) for Sections 5.2 and 6;
  `ell=1.1` for Table 1 comparison.

---

## Data Availability

The shear layer dataset is embedded in `realdata/shear_layer.py` and was
originally published by Wang & Ding (2009),
doi:[10.1198/TECH.2009.07011](https://doi.org/10.1198/TECH.2009.07011).

---

## License

MIT License — see `LICENSE`.
