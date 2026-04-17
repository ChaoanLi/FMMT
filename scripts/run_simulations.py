"""
Entry point: reproduce all simulation studies (Section 5).

Runs Steps 1-4 sequentially (or individually), then generates all figures.

Usage:
    python scripts/run_simulations.py              # all four steps
    python scripts/run_simulations.py --step 1     # step 1 only
    python scripts/run_simulations.py --quick      # quick (50 reps)
    python scripts/run_simulations.py --skip-comparison  # skip Table 1
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.sim_1d_global    import run_sim_1d_global
from simulation.sim_1d_subdomain import run_sim_1d_subdomain
from simulation.sim_2d_global    import run_sim_2d_global
from simulation.sim_2d_subdomain import run_sim_2d_subdomain
from simulation.visualize        import visualize_step
from simulation.comparison       import run_comparison


def main():
    parser = argparse.ArgumentParser(description="Reproduce Section 5 simulations")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                        help="Run a specific step only (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (50 reps per scenario)")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip Table 1 comparison (slow)")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    n_reps = 50 if args.quick else args.n_reps

    if not args.skip_comparison and (args.step is None):
        print("\n=== Table 1: Method Comparison (Section 5.1) ===")
        run_comparison(n_reps=n_reps, seed=args.seed)

    steps = [args.step] if args.step else [1, 2, 3, 4]
    step_fns = {
        1: run_sim_1d_global,
        2: run_sim_1d_subdomain,
        3: run_sim_2d_global,
        4: run_sim_2d_subdomain,
    }

    for s in steps:
        print(f"\n=== Step {s} ===")
        step_fns[s](n_reps=n_reps, seed=args.seed)
        print(f"  Generating Figure {s}...")
        visualize_step(s)

    print("\nAll simulation results written to results/figures/ and results/tables/")


if __name__ == "__main__":
    main()
