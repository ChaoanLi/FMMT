"""
Section 5.1 - Comparison with Existing Methods (Table 1).

Reproduces Table 1: Type I error and power for FMMT, ND (K1, K2), and EH
across six 1D scenarios, n in {25, 50}, alpha in {0.025, 0.05, 0.10},
1000 Monte Carlo replications each.

Paper settings:
  - Design: truncated normal (mu=0.5, sigma_x=0.2) on [0,1]
  - Noise: Gaussian, sigma=0.5
  - FMMT: ell=0.7 weight_decay_n, Matern nu=3.5 theta=1.0, 5-fold CV
  - ND: two equal-size-n samples; first from f1 sigma1=0, second sigma2=0.5
  - EH: half-cosine basis, difference-based variance
  - Modulation strength c=1.0 for H1

Usage:
    python -m simulation.comparison            # full run (1000 reps)
    python -m simulation.comparison --quick    # quick test (50 reps)
"""
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.functions import get_scenario_functions
from code.statistical_tests import compute_1d_pvalue_kde
from code.nd2003 import nd2003_hypothesis_test
from code.eubank_hart1992 import eubank_hart_test, generate_eh_comparison_data
from code.utils import set_random_seed

SCENARIOS = [
    "constant_vs_linear",
    "exp_vs_exp_linear",
    "sine_vs_sine_linear",
    "constant_vs_sine",
    "exp_vs_exp_sine",
    "sine_vs_double_sine",
]

SCENARIO_LABELS = {
    "constant_vs_linear":  "Const-Linear",
    "exp_vs_exp_linear":   "Exp-Linear",
    "sine_vs_sine_linear": "Sin-Linear",
    "constant_vs_sine":    "Const-Sin",
    "exp_vs_exp_sine":     "Exp-Sin",
    "sine_vs_double_sine": "Sin-Scale",
}

SAMPLE_SIZES   = [25, 50]
ALPHA_LEVELS   = [0.025, 0.05, 0.1]
NOISE_SIGMA    = 0.5
MODULATION_C   = 1.0
WEIGHT_DECAY_N = 1.1
N_BOOTSTRAPS   = 200


def _run_fmmt(scenario, hypothesis, n, sigma, c):
    modulation = 0.0 if hypothesis == "H0" else c
    try:
        _, _, _, global_pval = compute_1d_pvalue_kde(
            n_samples=n,
            domain_bounds=np.array([[0, 1]]),
            sigma=sigma,
            noise_type="gaussian",
            function_type=scenario,
            max_frequency=int(np.sqrt(n)),
            weight_decay_type="log",
            distribution_type="truncated_normal",
            use_mse=True,
            modulation_strength=modulation,
            subdomain_flags=(False, False, False),
            weight_decay_n=WEIGHT_DECAY_N,
        )
        return float(global_pval)
    except Exception:
        return np.nan


def _run_nd2003(scenario, hypothesis, n, sigma, c):
    try:
        ref = get_scenario_functions(scenario, "H0", "1D", c=0)
        f1 = ref["f1"]
        f2 = ref["f1"] if hypothesis == "H0" else \
             get_scenario_functions(scenario, "H1", "1D", c=c)["f2"]
        result = nd2003_hypothesis_test(
            f1=f1, f2=f2, n1=n, n2=n,
            sigma1=0, sigma2=sigma,
            num_bootstraps=N_BOOTSTRAPS, alpha=0.05,
            dist_type="truncated_normal",
        )
        return float(result["p_values"]["K1"]), float(result["p_values"]["K2"])
    except Exception:
        return np.nan, np.nan


def _run_eh(scenario, hypothesis, n, sigma, c):
    try:
        ref = get_scenario_functions(scenario, "H0", "1D", c=0)
        f1 = ref["f1"]
        f2 = ref["f1"] if hypothesis == "H0" else \
             get_scenario_functions(scenario, "H1", "1D", c=c)["f2"]
        X, Y, cm = generate_eh_comparison_data(
            f1=f1, f2=f2, n=n, sigma=sigma, dist_type="truncated_normal"
        )
        return {alpha: bool(eubank_hart_test(X, Y, cm, alpha=alpha, null_type="p0"))
                for alpha in ALPHA_LEVELS}
    except Exception:
        return {alpha: np.nan for alpha in ALPHA_LEVELS}


def run_comparison(n_reps=1000, seed=42, output_dir="results/tables"):
    """Run Table 1 comparison and save results to output_dir."""
    set_random_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    records = []
    for scenario in SCENARIOS:
        for n in SAMPLE_SIZES:
            for hypothesis in ["H0", "H1"]:
                c_val = MODULATION_C if hypothesis == "H1" else 0.0
                label = f"{SCENARIO_LABELS[scenario]} n={n} {hypothesis}"
                fmmt_pvals, k1_pvals, k2_pvals = [], [], []
                eh_rejects = {alpha: [] for alpha in ALPHA_LEVELS}
                for _ in tqdm(range(n_reps), desc=label, leave=False):
                    fmmt_pvals.append(_run_fmmt(scenario, hypothesis, n, NOISE_SIGMA, c_val))
                    k1, k2 = _run_nd2003(scenario, hypothesis, n, NOISE_SIGMA, c_val)
                    k1_pvals.append(k1)
                    k2_pvals.append(k2)
                    eh_rej = _run_eh(scenario, hypothesis, n, NOISE_SIGMA, c_val)
                    for alpha in ALPHA_LEVELS:
                        eh_rejects[alpha].append(eh_rej[alpha])
                fmmt_arr = np.array(fmmt_pvals)
                k1_arr   = np.array(k1_pvals)
                k2_arr   = np.array(k2_pvals)
                for alpha in ALPHA_LEVELS:
                    records.append({
                        "Scenario":   SCENARIO_LABELS[scenario],
                        "n":          n,
                        "Hypothesis": hypothesis,
                        "alpha":      alpha,
                        "K1":         float(np.nanmean(k1_arr < alpha)),
                        "K2":         float(np.nanmean(k2_arr < alpha)),
                        "EH":         float(np.nanmean(np.array(eh_rejects[alpha], dtype=float))),
                        "FMMT":       float(np.nanmean(fmmt_arr < alpha)),
                    })
    df = pd.DataFrame(records)
    out_path = Path(output_dir) / "table1_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Table 1 results saved to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 1 (Section 5.1)")
    parser.add_argument("--quick", action="store_true", help="50 reps quick test")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_comparison(n_reps=50 if args.quick else args.n_reps, seed=args.seed)
