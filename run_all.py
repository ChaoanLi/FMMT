"""
Reproduce ALL results in the paper with a single command.

Executes:
  1. Table 1 (Section 5.1) - method comparison
  2. Figure 1 + Table 2 (Section 5.2) - 1D global
  3. Figure 2 (Section 5.2) - 1D subdomain
  4. Figure 3 + Table 3 (Section 5.2) - 2D global
  5. Figure 4 (Section 5.2) - 2D subdomain
  6. Figure 5 (Section 6) - shear layer case study

Usage:
    python run_all.py            # full reproduction (~hours)
    python run_all.py --quick    # quick validation (~minutes, 50 reps)
    python run_all.py --seed 42

All outputs are written to results/figures/ and results/tables/.
"""
import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from simulation.comparison       import run_comparison
from simulation.sim_1d_global    import run_sim_1d_global
from simulation.sim_1d_subdomain import run_sim_1d_subdomain
from simulation.sim_2d_global    import run_sim_2d_global
from simulation.sim_2d_subdomain import run_sim_2d_subdomain
from simulation.visualize        import visualize_step
from realdata.shear_layer        import run_shear_layer


def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper results")
    parser.add_argument("--quick", action="store_true",
                        help="50 reps per scenario (for validation only)")
    parser.add_argument("--n-reps", type=int, default=1000,
                        help="Monte Carlo replications per scenario (default: 1000)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    n_reps = 50 if args.quick else args.n_reps
    seed   = args.seed

    print("=" * 60)
    print("REPRODUCING ALL PAPER RESULTS")
    print(f"  Replications: {n_reps}  |  Seed: {seed}")
    print("=" * 60)

    print("\n[1/6] Table 1 - Method comparison (Section 5.1)")
    run_comparison(n_reps=n_reps, seed=seed)

    print("\n[2/6] Figure 1 + Table 2 - 1D global (Section 5.2)")
    run_sim_1d_global(n_reps=n_reps, seed=seed)
    visualize_step(1)

    print("\n[3/6] Figure 2 - 1D subdomain (Section 5.2)")
    run_sim_1d_subdomain(n_reps=n_reps, seed=seed)
    visualize_step(2)

    print("\n[4/6] Figure 3 + Table 3 - 2D global (Section 5.2)")
    run_sim_2d_global(n_reps=n_reps, seed=seed)
    visualize_step(3)

    print("\n[5/6] Figure 4 - 2D subdomain (Section 5.2)")
    run_sim_2d_subdomain(n_reps=n_reps, seed=seed)
    visualize_step(4)

    print("\n[6/6] Figure 5 - Shear layer case study (Section 6)")
    run_shear_layer(seed=seed)

    print("\n" + "=" * 60)
    print("DONE. All outputs in results/figures/ and results/tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()
