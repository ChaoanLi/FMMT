"""
Neumeyer & Dette (2003) Method Implementation

This module implements the nonparametric comparison method from:
Neumeyer, N., & Dette, H. (2003). Nonparametric comparison of regression functions: 
An empirical process approach. The Annals of Statistics, 31(3), 880-920.

Key features:
- Local linear regression estimation with Epanechnikov kernel
- Plug-in bandwidth selection and KDE from kde.py module
- Wild bootstrap procedure with golden ratio weights
- Data generation compatible with local validation method comparison
"""

import numpy as np
from .kde import KDE_plugin
from .data_generation import generate_samples_from_distribution, sample_noise


# =============================================================================
# KERNEL FUNCTIONS
# =============================================================================

def epanechnikov_kernel(u):
    """
    Epanechnikov kernel function: K(u) = 0.75(1 - u²)I(|u| ≤ 1).
    
    Parameters:
        u (array): Standardized input values
        
    Returns:
        array: Kernel weights
    """
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)


# =============================================================================
# LOCAL LINEAR ESTIMATION
# =============================================================================

def local_linear_estimator(x_eval, X, Y, h):
    """
    Local linear regression estimator using Epanechnikov kernel.
    
    
    Parameters:
        x_eval (array): Evaluation points
        X (array): Sample covariates
        Y (array): Sample responses  
        h (float): Bandwidth parameter
        
    Returns:
        array: Local linear estimates at evaluation points
    """
    y_hat = np.zeros_like(x_eval, dtype=float)
    
    for i, x0 in enumerate(x_eval):
        # Kernel weights
        W = epanechnikov_kernel((X - x0) / h)
        
        # Local linear design matrix
        D = np.vstack([np.ones_like(X), X - x0]).T
        
        # Weighted least squares solution
        DtW = D.T * W
        DtWD = DtW @ D
        DtWY = DtW @ Y
        
        try:
            beta = np.linalg.solve(DtWD, DtWY)
            y_hat[i] = beta[0]  # Intercept term
        except np.linalg.LinAlgError:
            y_hat[i] = np.nan
    
    # Handle numerical issues via interpolation
    valid_mask = ~np.isnan(y_hat)
    if np.any(~valid_mask) and np.any(valid_mask):
        y_hat = np.interp(x_eval, x_eval[valid_mask], y_hat[valid_mask])
    
    return y_hat


# =============================================================================
# VARIANCE ESTIMATION
# =============================================================================

def rice_variance_estimator(Y, X):
    """
    Rice (1984) variance estimator: σ̂² = 0.5(n-1)⁻¹Σᵢ(Y₍ᵢ₊₁₎ - Y₍ᵢ₎)².
    
    Parameters:
        Y (array): Response values
        X (array): Covariate values
        
    Returns:
        float: Estimated error variance
    """
    sorted_indices = np.argsort(X)
    Y_sorted = Y[sorted_indices]
    return 0.5 * np.mean((Y_sorted[1:] - Y_sorted[:-1])**2)


# =============================================================================
# BANDWIDTH SELECTION
# =============================================================================

def get_rule_of_thumb_bandwidths(Y1, X1, Y2, X2):
    """
    Calculate rule-of-thumb bandwidths for ND2003 test procedure.
    
    Parameters:
        Y1, X1 (array): First sample observations
        Y2, X2 (array): Second sample observations
        
    Returns:
        tuple: (h, h1, h2, g) bandwidth parameters
    """
    n1, n2 = len(Y1), len(Y2)
    N = n1 + n2
    
    # Variance estimates
    sigma1_sq = rice_variance_estimator(Y1, X1)
    sigma2_sq = rice_variance_estimator(Y2, X2)
    
    # Optimal bandwidths
    h1 = (sigma1_sq / n1)**(1/5) if n1 > 0 else 0.1
    h2 = (sigma2_sq / n2)**(1/5) if n2 > 0 else 0.1
    
    # Pooled bandwidth
    h_num = n1 * sigma2_sq + n2 * sigma1_sq
    h = (h_num / N**2)**(1/5) if N > 0 else 0.1
    
    # Bootstrap bandwidth (undersmoothed)
    g = h**(5/6)
    
    return h, h1, h2, g


# =============================================================================
# TEST STATISTICS COMPUTATION
# =============================================================================

def calculate_nd2003_statistics(X1, Y1, X2, Y2, h, grid_size=150):
    """
    Calculate ND2003 test statistics K₁ and K₂ using plug-in density estimation.
    
    Parameters:
        X1, Y1 (array): First sample data
        X2, Y2 (array): Second sample data
        h (float): Bandwidth for regression estimation
        grid_size (int): Evaluation grid size
        
    Returns:
        tuple: (K1, K2) test statistics
    """
    n1, n2 = len(X1), len(Y2)
    N = n1 + n2
    
    # Evaluation grid
    T_GRID = np.linspace(0, 1, grid_size)
    
    # Pooled regression estimation
    X_pool = np.concatenate([X1, X2])
    Y_pool = np.concatenate([Y1, Y2])
    f_hat_pool_grid = local_linear_estimator(T_GRID, X_pool, Y_pool, h)
    
    # Density estimation using KDE_plugin
    r1_hat = KDE_plugin(X1.reshape(-1, 1))
    r2_hat = KDE_plugin(X2.reshape(-1, 1))
    
    # Evaluate densities on grid
    r1_hat_grid = r1_hat(T_GRID)
    r2_hat_grid = r2_hat(T_GRID)
    r_hat_grid = (n1/N) * r1_hat_grid + (n2/N) * r2_hat_grid
    
    # Interpolate to sample points
    f_hat_at_X1 = np.interp(X1, T_GRID, f_hat_pool_grid)
    f_hat_at_X2 = np.interp(X2, T_GRID, f_hat_pool_grid)
    
    r1_hat_at_X1 = r1_hat(X1)
    r2_hat_at_X1 = r2_hat(X1)
    r_hat_at_X1 = (n1/N) * r1_hat_at_X1 + (n2/N) * r2_hat_at_X1
    
    r1_hat_at_X2 = r1_hat(X2)
    r2_hat_at_X2 = r2_hat(X2)
    r_hat_at_X2 = (n1/N) * r1_hat_at_X2 + (n2/N) * r2_hat_at_X2
    
    # Empirical process components
    e1j = (n2/N) * (Y1 - f_hat_at_X1) * r_hat_at_X1 * r2_hat_at_X1
    e2j = (n1/N) * (Y2 - f_hat_at_X2) * r_hat_at_X2 * r1_hat_at_X2
    
    # Avoid division by zero
    r1_safe = np.maximum(r1_hat_at_X1, 1e-6)
    r2_safe = np.maximum(r2_hat_at_X2, 1e-6)
    
    f1j = (N/n1) * (Y1 - f_hat_at_X1) / r1_safe
    f2j = (N/n2) * (Y2 - f_hat_at_X2) / r2_safe
    
    # Empirical processes
    R_N1 = np.array([
        np.sum(e1j[X1 <= t]) - np.sum(e2j[X2 <= t]) 
        for t in T_GRID
    ]) / N
    
    R_N2 = np.array([
        np.sum(f1j[X1 <= t]) - np.sum(f2j[X2 <= t]) 
        for t in T_GRID
    ]) / N
    
    # Test statistics
    K1 = np.max(np.abs(R_N1))
    K2 = np.max(np.abs(R_N2))
    
    return K1, K2


# =============================================================================
# WILD BOOTSTRAP
# =============================================================================

def golden_ratio_weights(size):
    """
    Generate golden ratio distribution weights: V ∈ {(1-√5)/2, (1+√5)/2}.
    
    Parameters:
        size (int): Number of weights
        
    Returns:
        array: Bootstrap weights with E[V] = 0, Var[V] = 1
    """
    sqrt5 = np.sqrt(5)
    p = (sqrt5 + 1) / (2 * sqrt5)
    a = (1 - sqrt5) / 2
    b = (1 + sqrt5) / 2
    
    return np.random.choice([a, b], size=size, p=[p, 1 - p])


def wild_bootstrap_nd2003(X1, Y1, X2, Y2, h, g, num_bootstraps=200):
    """
    Wild bootstrap procedure for ND2003 hypothesis test.
    
    Parameters:
        X1, Y1 (array): First sample data
        X2, Y2 (array): Second sample data
        h (float): Test statistic bandwidth
        g (float): Residual estimation bandwidth
        num_bootstraps (int): Bootstrap replications
        
    Returns:
        dict: Bootstrap test statistic distributions
    """
    n1, n2 = len(Y1), len(Y2)
    
    # Pooled null hypothesis estimation
    X_pool = np.concatenate([X1, X2])
    Y_pool = np.concatenate([Y1, Y2])
    
    f_hat_g_at_X1 = local_linear_estimator(X1, X_pool, Y_pool, g)
    f_hat_g_at_X2 = local_linear_estimator(X2, X_pool, Y_pool, g)
    
    # Residuals under null hypothesis
    res1 = Y1 - f_hat_g_at_X1
    res2 = Y2 - f_hat_g_at_X2
    
    bootstrap_stats = {'K1': [], 'K2': []}
    
    for _ in range(num_bootstraps):
        # Wild bootstrap resampling
        V1 = golden_ratio_weights(n1)
        V2 = golden_ratio_weights(n2)
        
        Y1_star = f_hat_g_at_X1 + res1 * V1
        Y2_star = f_hat_g_at_X2 + res2 * V2
        
        # Bootstrap test statistics
        K1_star, K2_star = calculate_nd2003_statistics(X1, Y1_star, X2, Y2_star, h)
        
        bootstrap_stats['K1'].append(K1_star)
        bootstrap_stats['K2'].append(K2_star)
    
    return bootstrap_stats


# =============================================================================
# DATA GENERATION FOR COMPARISON
# =============================================================================

def generate_nd2003_comparison_data(f1, f2, n1, n2, sigma1, sigma2, domain=(0, 1), dist_type="truncated_normal"):
    """
    Generate data for ND2003 vs local validation comparison.
    
    Data generation logic:
    - H₀: Both samples from f₁ with respective noise levels
    - H₁: Sample 1 from f₁ (noise=0), Sample 2 from f₂ (noise=sigma)
    
    Parameters:
        f1, f2 (callable): Regression functions
        n1, n2 (int): Sample sizes
        sigma1, sigma2 (float): Error standard deviations
        domain (tuple): Covariate domain
        dist_type (str): Distribution type ('uniform', 'truncated_normal')
        
    Returns:
        tuple: (X1, Y1, X2, Y2) for comparison testing
    """
    omega = np.array([domain])
    
    # Generate covariates using specified distribution
    X1 = generate_samples_from_distribution(omega, n1, dist_type, mu=0.5, sigma=0.25)[:, 0]
    X2 = generate_samples_from_distribution(omega, n2, dist_type, mu=0.5, sigma=0.25)[:, 0]
    
    # Generate responses based on comparison logic
    # For fair comparison: H₁ has one clean sample (f₁) and one noisy sample (f₂)
    if sigma2 == 0:  # H₀ case: both from same function
        Y1 = f1(X1) + sample_noise(n1, sigma1, "gaussian")
        Y2 = f1(X2)  # No noise for second sample
    else:  # H₁ case: different functions
        Y1 = f1(X1)  # No noise for first sample  
        Y2 = f2(X2) + sample_noise(n2, sigma2, "gaussian")
    
    return X1, Y1, X2, Y2


# Create alias for backward compatibility
generate_nd2003_data = generate_nd2003_comparison_data


# =============================================================================
# MAIN HYPOTHESIS TEST
# =============================================================================

def nd2003_hypothesis_test(f1, f2, n1, n2, sigma1, sigma2, num_bootstraps=200, alpha=0.05, dist_type="truncated_normal"):
    """
    Complete ND2003 hypothesis test for regression function comparison.
    
    Tests H₀: f₁(x) = f₂(x) vs H₁: f₁(x) ≠ f₂(x).
    
    Parameters:
        f1, f2 (callable): Regression functions to compare
        n1, n2 (int): Sample sizes
        sigma1, sigma2 (float): Error standard deviations
        num_bootstraps (int): Bootstrap replications
        alpha (float): Significance level
        dist_type (str): Distribution type for covariate generation ('uniform', 'truncated_normal')
        
    Returns:
        dict: Test results with p-values and decisions
    """
    # Generate comparison data
    X1, Y1, X2, Y2 = generate_nd2003_comparison_data(f1, f2, n1, n2, sigma1, sigma2, dist_type=dist_type)
    
    # Calculate bandwidths
    h, h1, h2, g = get_rule_of_thumb_bandwidths(Y1, X1, Y2, X2)
    
    # Observed test statistics
    K1_obs, K2_obs = calculate_nd2003_statistics(X1, Y1, X2, Y2, h)
    
    # Wild bootstrap
    bootstrap_stats = wild_bootstrap_nd2003(X1, Y1, X2, Y2, h, g, num_bootstraps)
    
    # Statistical inference
    p_value_K1 = np.mean(np.array(bootstrap_stats['K1']) >= K1_obs)
    p_value_K2 = np.mean(np.array(bootstrap_stats['K2']) >= K2_obs)
    
    reject_K1 = p_value_K1 < alpha
    reject_K2 = p_value_K2 < alpha
    
    return {
        'test_statistics': {'K1': K1_obs, 'K2': K2_obs},
        'p_values': {'K1': p_value_K1, 'K2': p_value_K2},
        'reject_null': {'K1': reject_K1, 'K2': reject_K2},
        'alpha': alpha,
        'bootstrap_replications': num_bootstraps,
        'sample_sizes': {'n1': n1, 'n2': n2}
    } 