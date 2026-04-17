"""
Comprehensive Performance Comparison: Global Test Method vs Petersen, Jørgensen & Gordon (2015)

This experiment evaluates statistical performance of our global test method against
the established PJG2015 characteristic function-based ANOVA method across
multiple regression scenarios with optimized parameters based on empirical findings.

Key Features:
- Parallel processing with progress tracking
- Individual file saving for multi-terminal execution
- Comprehensive statistical analysis with detailed rejection rate summary
- Quick validation test for development

Configuration Details:
- Quick config: 2 scenarios × 1 sample size × 2 hypotheses = 4 configurations
- Full config: 6 scenarios × 2 sample sizes × 2 hypotheses = 24 configurations
- Each configuration tests 3 alpha levels [0.025, 0.05, 0.1]

Usage Examples:
    # Quick validation test (2 scenarios, 100 simulations each)
    python comparison_pjg2015_experiment.py --config quick
    
    # Full experiment (6 scenarios, 1000 simulations each, 24 configurations total)
    python comparison_pjg2015_experiment.py --config full --n-jobs 8
    
    # Parallel execution across multiple terminals
    python comparison_pjg2015_experiment.py --config full --scenario 1 --n-jobs 4
    python comparison_pjg2015_experiment.py --config full --scenario 2 --n-jobs 4
    # (run in separate terminals)
    
Performance Notes:
    - Quick config: ~2-5 minutes on modern hardware
    - Full config: ~30-60 minutes on modern hardware
    - Parallel processing provides 3-4x speedup
    - Detailed rejection rate summary shows all sample size and alpha combinations
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import sys
import json
import pickle
import hashlib
import multiprocessing as mp
from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
import time

# Add source directory to path

from code.pjg2015 import RegressionCurveTester
from code.data_generation import generate_samples_comparison

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

# Parallel processing configuration
N_JOBS = max(1, mp.cpu_count() - 1)
BATCH_SIZE = 10

# Test scenarios (same as ND2003 experiment)
TEST_SCENARIOS = [
    'constant_vs_linear',
    'exp_vs_exp_linear', 
    'sine_vs_sine_linear',
    'constant_vs_sine',
    'exp_vs_exp_sine',
    'sine_vs_double_sine'
]

# Experimental configurations
QUICK_CONFIG = {
    'sample_sizes': [50],
    'num_simulations': 100,
    'scenarios': ['exp_vs_exp_linear', 'sine_vs_sine_linear'],
    'description': 'Quick validation test for development'
}

FULL_CONFIG = {
    'sample_sizes': [25, 50],
    'num_simulations': 1000,
    'scenarios': TEST_SCENARIOS,
    'description': 'Complete experiment for publication results'
}

# Standard parameters
NOISE_LEVEL = 0.5  # Standard noise level
MODULATION_STRENGTHS = {'H0': 0.0, 'H1': 1.0}
ALPHA_LEVELS = [0.025, 0.05, 0.1]

# Weight decay parameter (based on optimization results from ablation study)
DEFAULT_WEIGHT_DECAY_N = 1.1  # Recommended from weight_decay_optimization_debug.py

# PJG2015 bandwidth parameter (h = C * n^(-0.375) as suggested in the paper)
PJG2015_BANDWIDTH_C = 1.5

# =============================================================================
# PROGRESS MANAGEMENT SYSTEM
# =============================================================================

class ExperimentProgressManager:
    """Manages experiment progress with checkpoint/resume functionality"""
    
    def __init__(self, experiment_dir, config_name, scenario_id=None):
        self.experiment_dir = Path(experiment_dir)
        self.config_name = config_name
        self.scenario_id = scenario_id
        
        # Scenario-specific file naming
        if scenario_id is not None:
            self.progress_file = self.experiment_dir / f"progress_pjg2015_{config_name}_scenario{scenario_id}.json"
            self.results_file = self.experiment_dir / f"results_pjg2015_{config_name}_scenario{scenario_id}.pkl"
        else:
            self.progress_file = self.experiment_dir / f"progress_pjg2015_{config_name}.json"
            self.results_file = self.experiment_dir / f"results_pjg2015_{config_name}.pkl"
            
        self.completed_configs = set()
        self.all_results = []
        
        self.experiment_dir.mkdir(exist_ok=True)
        self._load_progress()
    
    def _generate_config_id(self, scenario_name, hypothesis, n_sample):
        """Generate unique identifier for a configuration"""
        config_str = f"{scenario_name}_{hypothesis}_{n_sample}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _load_progress(self):
        """Load existing progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.completed_configs = set(progress_data.get('completed_configs', []))
                print(f"Loaded progress: {len(self.completed_configs)} configurations completed")
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
                self.completed_configs = set()
        
        if self.results_file.exists():
            try:
                with open(self.results_file, 'rb') as f:
                    self.all_results = pickle.load(f)
                print(f"Loaded existing results: {len(self.all_results)} simulation records")
            except Exception as e:
                print(f"Warning: Could not load results file: {e}")
                self.all_results = []
    
    def is_config_completed(self, scenario_name, hypothesis, n_sample):
        """Check if configuration is already completed"""
        config_id = self._generate_config_id(scenario_name, hypothesis, n_sample)
        return config_id in self.completed_configs
    
    def mark_config_completed(self, scenario_name, hypothesis, n_sample, results):
        """Mark configuration as completed and save results"""
        config_id = self._generate_config_id(scenario_name, hypothesis, n_sample)
        self.completed_configs.add(config_id)
        self.all_results.extend(results)
        
        self._save_progress()
        self._save_results()
        
        scenario_info = f"_scenario{self.scenario_id}" if self.scenario_id else ""
        print(f"Checkpoint saved{scenario_info}: {scenario_name}_{hypothesis}_n{n_sample} ({len(results)} simulations)")
    
    def _save_progress(self):
        """Save progress to file"""
        progress_data = {
            'completed_configs': list(self.completed_configs),
            'last_updated': datetime.now().isoformat(),
            'total_results': len(self.all_results),
            'scenario_id': self.scenario_id
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def _save_results(self):
        """Save results to file"""
        with open(self.results_file, 'wb') as f:
            pickle.dump(self.all_results, f)
    
    def get_all_results(self):
        """Get all accumulated results as DataFrame"""
        if self.all_results:
            return pd.DataFrame(self.all_results)
        else:
            return pd.DataFrame()

# =============================================================================
# CORE TESTING FUNCTIONS
# =============================================================================

def run_pjg2015_test(scenario_name, hypothesis, n_samples, sigma, c_value):
    """Execute PJG2015 method using data_generation.py and RegressionCurveTester"""
    try:
        # Generate data using our data generation system
        domain_bounds = [[0, 1]]  # Standard domain for 1D scenarios
        
        # Generate separate datasets for each "group" in PJG2015 framework
        # Both groups should have the same noise level for fair comparison
        
        # Group 1: H0 scenario with consistent noise level
        X_sample_1, y_sample_1 = generate_samples_comparison(
            omega=domain_bounds,
            n_samples=n_samples,
            scenario_name=scenario_name,
            dimension='1D',
            sigma=sigma,  # Same noise level as Group 2
            noise_type="gaussian",
            dist_type="truncated_normal",
            c=c_value
        )['H0']
        
        # Group 2: Current hypothesis scenario with same noise level
        X_sample_2, y_sample_2 = generate_samples_comparison(
            omega=domain_bounds,
            n_samples=n_samples,
            scenario_name=scenario_name,
            dimension='1D',
            sigma=sigma,  # Same noise level as Group 1
            noise_type="gaussian",
            dist_type="truncated_normal", 
            c=c_value
        )[hypothesis]
        
        # Prepare datasets for RegressionCurveTester
        # PJG2015 expects list of (X, Y) tuples
        datasets = [
            (X_sample_1.flatten(), y_sample_1),  # Group 1: reference (clean)
            (X_sample_2.flatten(), y_sample_2)   # Group 2: test (noisy)
        ]
        
        # Calculate bandwidth using paper's recommended formula
        total_n = 2 * n_samples
        h = PJG2015_BANDWIDTH_C * (total_n ** (-0.375))
        
        # Initialize and run PJG2015 test (verbose=False to suppress detailed output)
        tester = RegressionCurveTester(datasets, h=h, verbose=False)
        test_statistic, eigenvalues, p_value = tester.run_test(alpha=0.05)
        
        return {
            'method': 'pjg2015',
            'test_statistic': test_statistic,
            'p_value': p_value,
            'eigenvalues': eigenvalues.tolist() if eigenvalues is not None else None,
            'bandwidth': h,
            'total_n': total_n,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'method': 'pjg2015',
            'test_statistic': np.nan,
            'p_value': np.nan,
            'eigenvalues': None,
            'bandwidth': np.nan,
            'total_n': 2 * n_samples,
            'success': False,
            'error': str(e)
        }

def run_single_comparison(scenario_name, hypothesis, n_sample, sigma, c_value, simulation_id):
    """Execute single Monte Carlo comparison - PJG2015 only"""
    # PJG2015 method
    pjg_result = run_pjg2015_test(scenario_name, hypothesis, n_sample, sigma, c_value)
    
    # Combine results - p-values only, rejection rates computed later
    combined_result = {
        'simulation_id': simulation_id,
        'scenario': scenario_name,
        'hypothesis': hypothesis,
        'n_sample': n_sample,
        'total_n_pjg2015': 2 * n_sample,
        'sigma': sigma,
        'c_value': c_value,
        
        # PJG2015 statistical measures
        'pjg_pval': pjg_result['p_value'],
        'pjg_test_statistic': pjg_result['test_statistic'],
        'pjg_bandwidth': pjg_result['bandwidth'],
        
        # Success flags
        'pjg_success': pjg_result['success'],
        'pjg_error': pjg_result['error']
    }
    
    return combined_result

# =============================================================================
# PARALLEL EXECUTION SYSTEM
# =============================================================================

def run_single_comparison_wrapper(args):
    """Wrapper for parallel execution"""
    scenario_name, hypothesis, n_sample, sigma, c_value, simulation_id = args
    return run_single_comparison(scenario_name, hypothesis, n_sample, sigma, c_value, simulation_id)

def run_config_simulations_parallel(scenario_name, hypothesis, n_sample, sigma, c_value, num_simulations):
    """Run all simulations for a configuration in parallel batches"""
    simulation_args = [
        (scenario_name, hypothesis, n_sample, sigma, c_value, sim_id)
        for sim_id in range(num_simulations)
    ]
    
    all_results = []
    
    for batch_start in range(0, num_simulations, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_simulations)
        batch_args = simulation_args[batch_start:batch_end]
        
        batch_results = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(run_single_comparison_wrapper)(args) for args in batch_args
        )
        
        all_results.extend(batch_results)
        time.sleep(0.1)  # Brief pause to prevent system overload
    
    return all_results

# =============================================================================
# MAIN EXPERIMENT EXECUTION
# =============================================================================

def run_comparison_experiment(config_name='quick', experiment_dir=None, scenario_id=None):
    """Execute comparison experiment with parallel processing and scenario-specific saving"""
    if experiment_dir is None:
        experiment_dir = Path(__file__).parent / "experiment_checkpoints"
    
    progress_manager = ExperimentProgressManager(experiment_dir, config_name, scenario_id)
    
    # Select configuration
    config = QUICK_CONFIG if config_name == 'quick' else FULL_CONFIG
    
    # Display configuration info
    scenario_info = f" (scenario {scenario_id})" if scenario_id else ""
    print(f"Running {config_name} configuration{scenario_info}...")
    print(f"Parallel processing: {N_JOBS} cores, batch size: {BATCH_SIZE}")
    print(f"PJG2015 bandwidth constant: {PJG2015_BANDWIDTH_C}")
    print(f"Checkpoint directory: {experiment_dir}")
    
    # Generate all configurations (without alpha levels - computed later)
    all_configs = []
    for scenario_name in config['scenarios']:
        for hypothesis in ['H0', 'H1']:
            for n_sample in config['sample_sizes']:
                all_configs.append((scenario_name, hypothesis, n_sample))
    
    # Apply scenario filtering if specified
    if scenario_id is not None:
        scenario_configs = [cfg for cfg in all_configs if TEST_SCENARIOS.index(cfg[0]) + 1 == scenario_id]
    else:
        scenario_configs = all_configs
    
    total_configs = len(scenario_configs)
    completed_configs = sum(1 for cfg in scenario_configs 
                          if progress_manager.is_config_completed(*cfg))
    remaining_configs = total_configs - completed_configs
    
    print(f"Scenario configurations: {total_configs}")
    print(f"Already completed: {completed_configs}")
    print(f"Remaining: {remaining_configs}")
    print(f"Note: Each configuration runs {config['num_simulations']} simulations")
    print(f"Rejection rates will be computed for alpha levels: {ALPHA_LEVELS}")
    
    if remaining_configs == 0:
        print("All configurations completed! Loading results...")
        df_all_results = progress_manager.get_all_results()
        
        if len(df_all_results) > 0:
            print(f"\nLoaded {len(df_all_results)} total simulations")
            print(f"PJG2015 bandwidth constant: {PJG2015_BANDWIDTH_C}")
            
            # Show rejection rate summary for ALL results
            print_rejection_rate_summary(df_all_results, ALPHA_LEVELS)
        
        return df_all_results
    
    # Track current run results separately from cumulative results
    current_run_results = []
    
    # Main execution loop with progress tracking
    config_counter = completed_configs
    
    with tqdm(total=total_configs, initial=completed_configs, 
              desc=f"Experiment Progress{scenario_info}", unit="config") as pbar:
        
        for scenario_name, hypothesis, n_sample in scenario_configs:
            if progress_manager.is_config_completed(scenario_name, hypothesis, n_sample):
                continue
            
            config_counter += 1
            c_value = MODULATION_STRENGTHS[hypothesis]
            
            start_time = time.time()
            
            try:
                config_results = run_config_simulations_parallel(
                    scenario_name, hypothesis, n_sample, NOISE_LEVEL, c_value, config['num_simulations']
                )
                
                # Add to current run results
                current_run_results.extend(config_results)
                
                progress_manager.mark_config_completed(scenario_name, hypothesis, n_sample, config_results)
                
                elapsed_time = time.time() - start_time
                simulations_per_second = config['num_simulations'] / elapsed_time
                
                pbar.set_postfix({
                    'current': f"{scenario_name}_{hypothesis}_n{n_sample}",
                    'rate': f"{simulations_per_second:.1f}sim/s",
                    'success': f"PJG:{np.mean([r['pjg_success'] for r in config_results]):.1%}"
                })
                
            except Exception as e:
                print(f"Error in {scenario_name}_{hypothesis}_n{n_sample}: {e}")
                continue
                
            pbar.update(1)
    
    # Use ONLY current run results for analysis (NOT cumulative results)
    if len(current_run_results) > 0:
        df_current_run = pd.DataFrame(current_run_results)
        
        print(f"\nExperiment completed successfully{scenario_info}!")
        print(f"Current run simulations: {len(df_current_run)}")
        print(f"PJG2015 bandwidth constant: {PJG2015_BANDWIDTH_C}")
        
        # Show concise rejection rate summary for CURRENT RUN ONLY
        print_rejection_rate_summary(df_current_run, ALPHA_LEVELS)
        
        return df_current_run
        
    else:
        print("No results generated!")
        return pd.DataFrame()

# =============================================================================
# QUICK VALIDATION TEST
# =============================================================================

def quick_validation_test():
    """Ultra-fast validation test for immediate feedback"""
    print(f"Quick Validation Test: 2 scenarios, 10 simulations each")
    print(f"PJG2015 bandwidth constant: {PJG2015_BANDWIDTH_C}")
    
    scenarios = ['exp_vs_exp_linear', 'sine_vs_sine_linear']
    n_samples = 25
    sigma = 0.25
    n_trials = 10
    alpha_levels = ALPHA_LEVELS  # Use global alpha levels
    
    results = {}
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario}")
        
        # H0 test
        h0_results = []
        for i in tqdm(range(n_trials), desc=f"H0 {scenario}", leave=False):
            try:
                result = run_single_comparison(scenario, 'H0', n_samples, sigma, 0, i)
                h0_results.append(result)
            except Exception as e:
                print(f"H0 trial {i+1} failed: {e}")
        
        # H1 test
        h1_results = []
        for i in tqdm(range(n_trials), desc=f"H1 {scenario}", leave=False):
            try:
                result = run_single_comparison(scenario, 'H1', n_samples, sigma, 1.0, i)
                h1_results.append(result)
            except Exception as e:
                print(f"H1 trial {i+1} failed: {e}")
        
        # Analyze results
        h0_df = pd.DataFrame(h0_results)
        h1_df = pd.DataFrame(h1_results)
        
        # Calculate rejection rates for all metrics across all alpha levels
        h0_rejection_rates = {}
        h1_rejection_rates = {}
        
        metrics = ['pjg_pval']
        
        for metric in metrics:
            h0_pvals = h0_df[metric].dropna()
            h1_pvals = h1_df[metric].dropna()
            
            if len(h0_pvals) > 0:
                h0_rejection_rates[metric] = {f'alpha_{alpha}': (h0_pvals < alpha).mean() for alpha in alpha_levels}
            
            if len(h1_pvals) > 0:
                h1_rejection_rates[metric] = {f'alpha_{alpha}': (h1_pvals < alpha).mean() for alpha in alpha_levels}
        
        results[scenario] = {
            'h0_rejection_rates': h0_rejection_rates,
            'h1_rejection_rates': h1_rejection_rates,
            'h0_results': h0_results,
            'h1_results': h1_results
        }
        
        # Show only PJG2015 metrics
        key_metrics = {
            'PJG2015': 'pjg_pval'
        }
        
        print(f"Rejection Rates:")
        print(f"  H0 (Type I Error):")
        for metric_name, metric_key in key_metrics.items():
            if metric_key in h0_rejection_rates:
                rate_05 = h0_rejection_rates[metric_key]['alpha_0.05']
                print(f"    {metric_name:12}: {rate_05:.3f}")
        
        print(f"  H1 (Power):")
        for metric_name, metric_key in key_metrics.items():
            if metric_key in h1_rejection_rates:
                rate_05 = h1_rejection_rates[metric_key]['alpha_0.05']
                print(f"    {metric_name:12}: {rate_05:.3f}")
    
    return results

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Global Test vs PJG2015 Performance Comparison')
    parser.add_argument('--config', choices=['quick', 'full'], default='quick',
                       help='Experiment configuration')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS,
                       help=f'Number of parallel jobs (default: {N_JOBS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size for parallel processing (default: {BATCH_SIZE})')
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Directory for checkpoint files')
    parser.add_argument('--clean-start', action='store_true',
                       help='Start fresh, ignore existing checkpoints')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick validation test and exit')
    
    parser.add_argument('--pjg-bandwidth-c', type=float, default=PJG2015_BANDWIDTH_C,
                       help=f'PJG2015 bandwidth constant (default: {PJG2015_BANDWIDTH_C})')
    
    # Scenario-specific execution for parallel terminal execution
    parser.add_argument('--scenario', type=int, default=None,
                       help='Specific scenario ID for parallel execution (1-6)')
    
    return parser.parse_args()

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def print_rejection_rate_summary(df_results, alpha_levels):
    """Print comprehensive rejection rate summary for all configurations"""
    
    print(f"\nREJECTION RATE SUMMARY")
    print("=" * 80)
    
    # Focus on key statistics
    key_metrics = {
        'PJG2015': 'pjg_pval'
    }
    
    # Get all unique combinations
    scenarios = sorted(df_results['scenario'].unique())
    sample_sizes = sorted(df_results['n_sample'].unique())
    hypotheses = ['H0', 'H1']
    
    total_configs = len(scenarios) * len(sample_sizes) * len(hypotheses)
    print(f"Total configurations: {total_configs}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Alpha levels: {alpha_levels}")
    print(f"Scenarios: {len(scenarios)}, Hypotheses: {len(hypotheses)}")
    print("=" * 80)
    
    config_count = 0
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        print("-" * 60)
        
        df_scenario = df_results[df_results['scenario'] == scenario]
        
        for n_sample in sample_sizes:
            df_n = df_scenario[df_scenario['n_sample'] == n_sample]
            
            if len(df_n) == 0:
                continue
                
            print(f"\n  Sample Size n={n_sample}:")
            
            for hypothesis in hypotheses:
                df_hyp = df_n[df_n['hypothesis'] == hypothesis]
                
                if len(df_hyp) == 0:
                    continue
                    
                config_count += 1
                print(f"\n    {hypothesis} (Config {config_count}/{total_configs}):")
                
                for metric_name, metric_col in key_metrics.items():
                    pvals = df_hyp[metric_col].dropna()
                    
                    if len(pvals) > 0:
                        # Calculate rejection rates for all alpha levels
                        rates_str = []
                        for alpha in alpha_levels:
                            rate = (pvals < alpha).mean()
                            rates_str.append(f"α={alpha}: {rate:.3f}")
                        
                        n_sims = len(pvals)
                        print(f"      {metric_name:12}: [{', '.join(rates_str)}] (n_sim={n_sims})")
                    else:
                        print(f"      {metric_name:12}: No data")
    
    print(f"\n" + "=" * 80)
    print(f"Summary: Displayed {config_count} configurations")
    
    # Additional summary statistics
    if len(df_results) > 0:
        print(f"Total simulations: {len(df_results)}")
        success_rate = df_results['pjg_success'].mean()
        print(f"PJG2015 success rate: {success_rate:.1%}")
    
    print("=" * 80)

def analyze_and_save_results(df_results, config_name, output_dir):
    """Analyze results and save in structured format - simplified for key metrics only"""
    alpha_levels = ALPHA_LEVELS  # Use global alpha levels
    
    # Get PJG2015 bandwidth constant from results
    pjg_bandwidth_c = df_results['pjg_bandwidth'].iloc[0] / (df_results['total_n_pjg2015'].iloc[0] ** (-0.375)) if len(df_results) > 0 else PJG2015_BANDWIDTH_C
    
    # Prepare detailed p-values for inclusion in main results file (PJG2015 only)
    detailed_pvalues = df_results[['scenario', 'hypothesis', 'n_sample', 'simulation_id', 
                                  'pjg_pval', 'pjg_test_statistic', 'pjg_bandwidth']].to_dict('records')
    
    # Simplified analysis structure - only key metrics
    analysis = {}
    
    # Focus only on PJG2015 statistics
    key_metrics = {
        'PJG2015': 'pjg_pval'
    }
    
    for hypothesis in ['H0', 'H1']:
        df_hyp = df_results[df_results['hypothesis'] == hypothesis]
        hyp_analysis = {}
        
        for scenario in sorted(df_hyp['scenario'].unique()):
            df_scenario = df_hyp[df_hyp['scenario'] == scenario]
            scenario_analysis = {}
            
            # Analyze by sample size
            for n_sample in sorted(df_scenario['n_sample'].unique()):
                df_config = df_scenario[df_scenario['n_sample'] == n_sample]
                config_analysis = {}
                
                # Only analyze key metrics
                for metric_name, metric_col in key_metrics.items():
                    pvals = df_config[metric_col].dropna()
                    if len(pvals) > 0:
                        # Rejection rates for all alpha levels
                        rejection_rates = {
                            f'alpha_{alpha}': float((pvals < alpha).mean()) for alpha in alpha_levels
                        }
                        config_analysis[metric_name] = rejection_rates
                
                config_analysis['n_simulations'] = int(len(df_config))
                scenario_analysis[f'n_{n_sample}'] = config_analysis
            
            hyp_analysis[scenario] = scenario_analysis
        
        analysis[hypothesis] = hyp_analysis
    
    # Save results  
    output_file = output_dir / f'comparison_pjg2015_results_{config_name}.json'
    
    results_summary = {
        "experiment_name": f"Global Test vs PJG2015 Comparison ({config_name.capitalize()})",
                 "description": "Complete results: rejection rates and detailed p-values for PJG2015 statistics",
        "parameters": {
            "noise_level": NOISE_LEVEL,
            "alpha_levels": alpha_levels,
            "sample_sizes": sorted(df_results['n_sample'].unique().tolist()),
            "scenarios": sorted(df_results['scenario'].unique().tolist()),
                         "key_statistics": ["PJG2015"],
             "pjg2015_bandwidth_c": float(pjg_bandwidth_c)
        },
        "rejection_rates": analysis,
        "detailed_pvalues": detailed_pvalues,
        "summary": {
                         "total_simulations": int(len(df_results)),
             "configurations": int(len(df_results.groupby(['scenario', 'hypothesis', 'n_sample']))),
             "pjg2015_bandwidth_constant": float(pjg_bandwidth_c)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    file_size = output_file.stat().st_size
    size_str = f"{file_size/(1024*1024):.1f} MB" if file_size > 1024*1024 else f"{file_size/1024:.1f} KB"
    
    print(f"Results saved: {output_file}")
    print(f"File size: {size_str}")
    
    # Print experiment summary (rejection rates already shown during execution)
    print(f"\nExperiment Summary:")
    print(f"Configurations: {len(df_results.groupby(['scenario', 'hypothesis', 'n_sample']))}")
    print(f"Simulations per config: {df_results.groupby(['scenario', 'hypothesis', 'n_sample']).size().iloc[0]}")
    print(f"Key statistics: PJG2015")
    print(f"Alpha levels: {alpha_levels}")
    print(f"PJG2015 bandwidth constant: {pjg_bandwidth_c}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    
    # Quick validation test mode
    if args.quick_test:
        print("=== QUICK VALIDATION TEST ===")
        quick_validation_test()
        exit(0)
    
    # Update global settings
    N_JOBS = args.n_jobs
    BATCH_SIZE = args.batch_size
    PJG2015_BANDWIDTH_C = args.pjg_bandwidth_c
    
    # Display execution info
    print("=" * 60)
    print("GLOBAL TEST vs PETERSEN, JØRGENSEN & GORDON (2015) COMPARISON")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Parallel jobs: {N_JOBS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Noise level: {NOISE_LEVEL}")
    print(f"PJG2015 bandwidth constant: {args.pjg_bandwidth_c}")
    
    if args.scenario:
        print(f"Scenario: {args.scenario}")
    
    # Clean start if requested
    if args.clean_start:
        experiment_dir = Path(args.experiment_dir) if args.experiment_dir else Path(__file__).parent / "experiment_checkpoints"
        if experiment_dir.exists():
            import shutil
            # Only clean PJG2015-related files
            for file_path in experiment_dir.glob("*pjg2015*"):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            print(f"Cleaned PJG2015 checkpoint files in: {experiment_dir}")
    
    # Run experiment
    try:
        results_df = run_comparison_experiment(
            args.config, 
            args.experiment_dir, 
            args.scenario
        )
        
        if len(results_df) == 0:
            print("No results generated!")
            exit(1)
            
        # Analyze and save results
        output_dir = Path(args.experiment_dir) if args.experiment_dir else Path(__file__).parent
        analyze_and_save_results(results_df, args.config, output_dir)
        
        print("\nExperiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved for resumption.")
        exit(0)
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
