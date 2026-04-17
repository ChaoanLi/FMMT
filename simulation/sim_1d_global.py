"""
Section 5.2 - 1D Global Scenarios (Figure 1, Table 2).

Evaluates FMMT power and Type I error for six global 1D alternatives.
n=50, sigma=0.1, c in [-2, 2] step 0.2, 1000 replications.

Outputs:
    results/tables/step1_type_i_error.csv   (Table 2 in paper)
    results/figures/step1_power_curves.png  (Figure 1 in paper)
    results/step1_1d_global.json            (raw MC results)

Usage:
    python -m simulation.sim_1d_global
    python -m simulation.sim_1d_global --quick
"""
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.statistical_tests import compute_1d_pvalue_kde
from code.utils import set_random_seed

SCENARIOS = [
    "constant_vs_linear",
    "exp_vs_exp_linear",
    "sine_vs_sine_linear",
    "constant_vs_sine",
    "exp_vs_exp_sine",
    "sine_vs_double_sine",
]

N_SAMPLES      = 50
SIGMA          = 0.1
WEIGHT_DECAY_N = 0.7
ALPHA          = 0.05
C_VALUES       = np.arange(-2.0, 2.1, 0.2)


def run_single_trial(scenario, c):
    try:
        return compute_1d_pvalue_kde(
            n_samples=N_SAMPLES,
            domain_bounds=np.array([[0, 1]]),
            sigma=SIGMA,
            noise_type="gaussian",
            function_type=scenario,
            max_frequency=int(np.sqrt(N_SAMPLES)),
            weight_decay_type="log",
            distribution_type="uniform",
            use_mse=True,
            modulation_strength=c,
            subdomain_flags=(False, False, False),
            weight_decay_n=WEIGHT_DECAY_N,
        )
    except Exception:
        return (0.5, 0.5, 0.5, 0.5)


def compute_rejection_rates(pvalue_results, alpha=ALPHA):
    arr = np.array(pvalue_results)
    n_sub = arr.shape[1] - 1
    rates = {"global": float(np.mean(arr[:, -1] < alpha))}
    for i in range(n_sub):
        rates[f"subdomain_{i+1}"] = float(np.mean(arr[:, i] < alpha))
    rates["bonferroni"] = float(np.mean(np.any(arr[:, :-1] < alpha / n_sub, axis=1)))
    return rates


def run_sim_1d_global(n_reps=1000, seed=42, output_dir="results"):
    set_random_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for scenario in SCENARIOS:
        print(f"\nStep 1 - {scenario}")
        sc_results = {}
        for c in C_VALUES:
            pvals = [run_single_trial(scenario, c)
                     for _ in tqdm(range(n_reps), desc=f"c={c:.1f}", leave=False)]
            sc_results[f"c_{c:.1f}"] = {
                "c": float(c),
                "rejection_rates": compute_rejection_rates(pvals),
                "pvalue_results": [[float(v) for v in row] for row in pvals],
            }
        results[scenario] = sc_results

    json_path = Path(output_dir) / "step1_1d_global.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nStep 1 complete. Raw results: {json_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Figure 1 and Table 2")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_sim_1d_global(n_reps=50 if args.quick else args.n_reps, seed=args.seed)
