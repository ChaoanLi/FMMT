"""
Kernel Density Estimation (KDE) module with bandwidth selection and boundary correction.

This module provides unified KDE functionality using Epanechnikov kernel with normalization
and bandwidth selection. The implementation uses a single framework that handles both 1D and 2D cases,
with boundary correction support for improved accuracy on bounded domains.
"""

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import minimize_scalar, minimize
from scipy import stats

# ============================================================================
# Core Kernel Functions (Unified)
# ============================================================================

def epanechnikov_kernel(u, dimension=1):
    """
    Unified Epanechnikov kernel function for any dimension.
    
    Parameters:
        u (array): Standardized distances, can be 1D or 2D norm
        dimension (int): Dimension of the data (1 or 2)
        
    Returns:
        array: Kernel values
    """
    if dimension == 1:
        # 1D Epanechnikov: K(u) = 0.75 * (1 - u²) * I(|u| ≤ 1)
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)
    elif dimension == 2:
        # 2D Epanechnikov: K(u) = (2/π) * (1 - ||u||²) * I(||u|| ≤ 1)
        u_norm_sq = u
        return (2/np.pi) * (1 - u_norm_sq) * (u_norm_sq <= 1)
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")

# ============================================================================
# Boundary Correction Functions
# ============================================================================

def apply_boundary_correction_1d(x, density, bandwidth, domain=(0, 1)):
    """
    Apply boundary correction for 1D density estimation on bounded domain.
    
    For Epanechnikov kernel, applies a simple multiplicative correction factor
    near boundaries to compensate for the truncation of kernel support.
    
    Parameters:
        x (array): Evaluation points
        density (array): Uncorrected density values
        bandwidth (float): Bandwidth used in density estimation
        domain (tuple): Domain bounds (a, b)
        
    Returns:
        array: Boundary-corrected density values
    """
    x = np.asarray(x)
    density = np.asarray(density)
    a, b = domain
    h = bandwidth
    
    # Initialize correction factors
    correction = np.ones_like(density)
    
    # For Epanechnikov kernel: K(u) = 0.75 * (1 - u²) * I(|u| ≤ 1)
    # Support is [-1, 1], so kernel extends ±h from each point
    
    for i, xi in enumerate(x):
        correction_factor = 1.0
        
        # Left boundary correction
        if xi <= a + h:  # Within one bandwidth of left boundary
            distance_to_boundary = xi - a
            if distance_to_boundary >= 0:  # Only correct points inside domain
                # Fraction of kernel support that's cut off by boundary
                cutoff_fraction = max(0, (h - distance_to_boundary) / h)
                if cutoff_fraction > 0:
                    # Apply correction based on how much support is lost
                    # Use a conservative correction to avoid overcorrection
                    correction_factor = 1.0 + 0.5 * cutoff_fraction
        
        # Right boundary correction  
        if xi >= b - h:  # Within one bandwidth of right boundary
            distance_to_boundary = b - xi
            if distance_to_boundary >= 0:  # Only correct points inside domain
                # Fraction of kernel support that's cut off by boundary
                cutoff_fraction = max(0, (h - distance_to_boundary) / h)
                if cutoff_fraction > 0:
                    # Apply correction based on how much support is lost
                    # Use a conservative correction to avoid overcorrection
                    correction_factor = 1.0 + 0.5 * cutoff_fraction
        
        # Limit correction to avoid numerical instability
        correction[i] = min(correction_factor, 2.0)
    
    return density * correction


def apply_boundary_correction_2d(x, density, bandwidth, domain=[(0, 1), (0, 1)]):
    """
    Apply boundary correction for 2D density estimation on bounded domain.
    
    Parameters:
        x (array): Evaluation points with shape (n, 2)
        density (array): Uncorrected density values
        bandwidth (tuple): Bandwidths (h1, h2)
        domain (list): Domain bounds [(a1, b1), (a2, b2)]
        
    Returns:
        array: Boundary-corrected density values
    """
    x = np.asarray(x)
    density = np.asarray(density)
    (a1, b1), (a2, b2) = domain
    h1, h2 = bandwidth
    
    # Initialize correction factors
    correction = np.ones_like(density)
    
    # For 2D Epanechnikov kernel: K(u) = (2/π) * (1 - ||u||²) * I(||u|| ≤ 1)
    # Support is unit circle, so kernel extends ±h_i from each point in dimension i
    
    for i, (x1i, x2i) in enumerate(x):
        correction_factor = 1.0
        
        # Check boundaries in each dimension
        # Note: For 2D case, boundary correction is more complex
        # We use a simplified approach focusing on the most affected regions
        
        # X1 dimension boundaries
        if x1i - h1 < a1 or x1i + h1 > b1:
            # Apply 1D-like correction in x1 direction
            edge_factor = 1.0
            if x1i - h1 < a1:
                u1_left = (a1 - x1i) / h1
                if -1 < u1_left < 1:
                    edge_factor += 0.2  # Conservative correction
            if x1i + h1 > b1:
                u1_right = (x1i - b1) / h1
                if -1 < u1_right < 1:
                    edge_factor += 0.2  # Conservative correction
            correction_factor *= edge_factor
        
        # X2 dimension boundaries
        if x2i - h2 < a2 or x2i + h2 > b2:
            # Apply 1D-like correction in x2 direction
            edge_factor = 1.0
            if x2i - h2 < a2:
                u2_left = (a2 - x2i) / h2
                if -1 < u2_left < 1:
                    edge_factor += 0.2  # Conservative correction
            if x2i + h2 > b2:
                u2_right = (x2i - b2) / h2
                if -1 < u2_right < 1:
                    edge_factor += 0.2  # Conservative correction
            correction_factor *= edge_factor
        
        # Limit correction to avoid numerical instability
        correction[i] = min(correction_factor, 2.0)
    
    return density * correction

# ============================================================================
# Bandwidth Selection (Unified)
# ============================================================================

def get_rule_of_thumb_bandwidth(X):
    """
    Calculate bandwidth using rule of thumb for any dimension.
    
    Parameters:
        X (array): Data with shape (n, d)
        
    Returns:
        float or tuple: Bandwidth(s) for each dimension
    """
    n, d = X.shape
    
    if d == 1:
        # 1D rule of thumb: h = n^(-1/5) * σ
        sigma = np.std(X[:, 0])
        h = n**(-1/5) * (sigma)**(2/5)
        return max(h, 0.01)  # Minimum bandwidth
    
    elif d == 2:
        # 2D rule of thumb: h_j = n^(-1/6) * σ_j
        sigma1 = np.std(X[:, 0])
        sigma2 = np.std(X[:, 1])
        h1 = n**(-1/6) * (sigma1)**(1/3)
        h2 = n**(-1/6) * (sigma2)**(1/3)
        
        # For small samples, ensure minimum bandwidth to avoid zero density regions
        min_h1 = max(0.01, 0.6 * sigma1)  # At least 60% of std  
        min_h2 = max(0.01, 0.6 * sigma2)  # At least 60% of std
        
        return (max(h1, min_h1), max(h2, min_h2))
    
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")


def get_plugin_bandwidth(X, c):
    """
    Calculate bandwidth using plug-in method with learnable constant c.
    
    Parameters:
        X (array): Data with shape (n, d)
        c (float or tuple): Constant(s) to replace standard deviation
        
    Returns:
        float or tuple: Bandwidth(s) for each dimension
    """
    n, d = X.shape
    
    if d == 1:
        # 1D plug-in: h = n^(-1/5) * c
        h = n**(-1/5) * c
        return max(h, 0.01)  # Minimum bandwidth
    
    elif d == 2:
        # 2D plug-in: h_j = n^(-1/6) * c_j
        if isinstance(c, (tuple, list, np.ndarray)):
            c1, c2 = c[0], c[1]
        else:
            # If single constant provided, use same for both dimensions
            c1 = c2 = c
            
        h1 = n**(-1/6) * c1
        h2 = n**(-1/6) * c2
        
        return (max(h1, 0.01), max(h2, 0.01))
    
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")


def cross_validation_log_likelihood(X, c, cv_folds=5):
    """
    Calculate cross-validation log-likelihood to evaluate the quality of constant c.
    
    Parameters:
        X (array): Data points with shape (n, d)
        c (float or tuple): Constant c
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        float: Average cross-validation log-likelihood (negative value for minimization)
    """
    X = np.array(X)
    n, d = X.shape
    
    # Return large loss if sample size is too small
    if n < cv_folds:
        return 1e6
    
    # Create cross-validation indices
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // cv_folds
    
    log_likelihoods = []
    
    for fold in range(cv_folds):
        # Split training and test sets
        start_idx = fold * fold_size
        if fold == cv_folds - 1:  # Last fold contains all remaining points
            end_idx = n
        else:
            end_idx = (fold + 1) * fold_size
            
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
            
        X_train = X[train_indices]
        X_test = X[test_indices]
        
        # Calculate bandwidth
        try:
            bandwidth = get_plugin_bandwidth(X_train, c)
        except:
            return 1e6
            
        # Calculate density on test set
        try:
            densities = kernel_density_estimator(X_test, X_train, bandwidth)
            
            # Avoid numerical issues in log calculation
            densities = np.maximum(densities, 1e-10)
            
            # Calculate log-likelihood
            fold_log_likelihood = np.mean(np.log(densities))
            
            if np.isfinite(fold_log_likelihood):
                log_likelihoods.append(fold_log_likelihood)
        except:
            # If calculation fails, give a large penalty
            return 1e6
    
    if len(log_likelihoods) == 0:
        return 1e6
        
    # Return negative average log-likelihood (for minimization)
    return -np.mean(log_likelihoods)


def optimize_plugin_constant(X, c_bounds=None):
    """
    Optimize constant c in plug-in method using cross-validation.
    
    Parameters:
        X (array): Data points with shape (n, d)
        c_bounds (tuple or dict): Search bounds for constant c
        
    Returns:
        float or tuple: Optimal constant c
    """
    X = np.array(X)
    n, d = X.shape
    
    # Get data standard deviation as reference
    if d == 1:
        sigma_ref = np.std(X[:, 0])
        
        # Set search bounds
        if c_bounds is None:
            c_bounds = (0.1 * sigma_ref, 3.0 * sigma_ref)
        
        # Define objective function
        def objective(c):
            return cross_validation_log_likelihood(X, c)
        
        # Optimize using golden section search
        try:
            result = minimize_scalar(objective, bounds=c_bounds, method='bounded')
            optimal_c = result.x if result.success else sigma_ref
        except:
            # If optimization fails, use standard deviation as default
            optimal_c = sigma_ref
            
        return optimal_c
        
    elif d == 2:
        sigma1_ref = np.std(X[:, 0])
        sigma2_ref = np.std(X[:, 1])
        
        # Set search bounds
        if c_bounds is None:
            c_bounds = {
                'c1': (0.1 * sigma1_ref, 3.0 * sigma1_ref),
                'c2': (0.1 * sigma2_ref, 3.0 * sigma2_ref)
            }
        
        # Define objective function
        def objective(params):
            c1, c2 = params
            return cross_validation_log_likelihood(X, (c1, c2))
        
        # Initial guess
        x0 = [sigma1_ref, sigma2_ref]
        
        # Bounds
        bounds = [c_bounds['c1'], c_bounds['c2']]
        
        try:
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            optimal_c = tuple(result.x) if result.success else (sigma1_ref, sigma2_ref)
        except:
            # If optimization fails, use standard deviation as default
            optimal_c = (sigma1_ref, sigma2_ref)
            
        return optimal_c
    
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")

# ============================================================================
# Core Density Estimation (Unified Framework)
# ============================================================================

def kernel_density_estimator(X_eval, X_data, bandwidth, enable_boundary_correction=None, domain_bounds=None):
    """
    Unified kernel density estimator for 1D and 2D data with optional boundary correction.
    
    Parameters:
        X_eval (array): Evaluation points, shape (..., d)
        X_data (array): Data points, shape (n, d)
        bandwidth (float or tuple): Bandwidth(s)
        enable_boundary_correction (bool or None): Whether to apply boundary correction
                                                 - None: default True for 1D, False for 2D
                                                 - True/False: explicitly enable/disable
        domain_bounds (tuple or list): Domain bounds for boundary correction
                                     - 1D: (a, b) 
                                     - 2D: [(a1, b1), (a2, b2)]
                                     - None: default to [(0, 1)] for 1D or [(0, 1), (0, 1)] for 2D
        
    Returns:
        array: Density values at evaluation points
    """
    n, d = X_data.shape
    X_eval = np.array(X_eval)
    
    # Set default boundary correction based on dimension
    if enable_boundary_correction is None:
        enable_boundary_correction = (d == 1)  # True for 1D, False for 2D
    
    # Set default domain bounds if not provided
    if domain_bounds is None:
        if d == 1:
            domain_bounds = (0, 1)
        elif d == 2:
            domain_bounds = [(0, 1), (0, 1)]
    
    # Ensure X_eval has correct shape
    if X_eval.ndim == 1 and d > 1:
        X_eval = X_eval.reshape(1, -1)
    elif X_eval.ndim == 1 and d == 1:
        X_eval = X_eval.reshape(-1, 1)
    
    if d == 1:
        # 1D case
        h = bandwidth
        n_eval = X_eval.shape[0]
        density = np.zeros(n_eval)
        
        for i in range(n_eval):
            # Standardized distances
            u = (X_eval[i, 0] - X_data[:, 0]) / h
            # Kernel values
            kernel_vals = epanechnikov_kernel(u, dimension=1)
            # Density estimate
            density[i] = np.sum(kernel_vals) / (n * h)
        
        # Apply boundary correction if enabled
        if enable_boundary_correction:
            density = apply_boundary_correction_1d(X_eval[:, 0], density, h, domain_bounds)
            
        return density
    
    elif d == 2:
        # 2D case
        h1, h2 = bandwidth
        
        if X_eval.ndim == 1:
            # Single evaluation point
            X_eval = X_eval.reshape(1, -1)
        
        n_eval = X_eval.shape[0]
        density = np.zeros(n_eval)
        
        for i in range(n_eval):
            # Standardized distances for each dimension
            u1 = (X_eval[i, 0] - X_data[:, 0]) / h1
            u2 = (X_eval[i, 1] - X_data[:, 1]) / h2
            
            # Combined standardized distance squared
            u_norm_sq = u1**2 + u2**2
            
            # Kernel values
            kernel_vals = epanechnikov_kernel(u_norm_sq, dimension=2)
            
            # Density estimate
            density[i] = np.sum(kernel_vals) / (n * h1 * h2)
        
        # Apply boundary correction if enabled
        if enable_boundary_correction:
            density = apply_boundary_correction_2d(X_eval, density, (h1, h2), domain_bounds)
            
        return density
    
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")

# ============================================================================
# Main KDE Interface (Unified)
# ============================================================================

def kde_density_estimation(X, bandwidth=None, enable_boundary_correction=None, domain_bounds=None):
    """
    Unified density estimation using Epanechnikov kernel with optional boundary correction.
    
    Parameters:
        X (array): Input data with shape (n, d) where d ∈ {1, 2}
        bandwidth (float, tuple, or None): Bandwidth(s), if None use plug-in method
        enable_boundary_correction (bool or None): Whether to apply boundary correction
                                                 - None: default True for 1D, False for 2D
                                                 - True/False: explicitly enable/disable
        domain_bounds (tuple or list): Domain bounds for boundary correction
    
    Returns:
        function: Normalized density function
    """
    X = np.array(X)
    n, d = X.shape
    
    # Get bandwidth using plug-in method if not provided
    if bandwidth is None:
        optimal_c = optimize_plugin_constant(X)
        bandwidth = get_plugin_bandwidth(X, optimal_c)
    
    # Create evaluation grid based on dimension
    if d == 1:
        # 1D evaluation grid
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        x_range = x_max - x_min
        margin = 0.1 * x_range
        x_eval = np.linspace(x_min - margin, x_max + margin, 100).reshape(-1, 1)
        
        # Compute density
        density_values = kernel_density_estimator(x_eval, X, bandwidth, enable_boundary_correction, domain_bounds)
        
        # Normalize with numerical stability check
        dx = x_eval[1, 0] - x_eval[0, 0]
        integral = np.sum(density_values) * dx
        
        # Check for numerical issues
        if integral <= 0 or np.isnan(integral) or np.isinf(integral):
            # Fallback to uniform distribution
            density_values_normalized = np.ones_like(density_values) / (dx * density_values.size)
        else:
            density_values_normalized = density_values / integral
        
        # Create interpolation function
        density_func = interp1d(x_eval[:, 0], density_values_normalized, 
                               bounds_error=False, fill_value=0.0)
        
        def kde_function(x):
            x = np.array(x)
            if x.ndim == 2:
                x = x[:, 0] if x.shape[1] == 1 else x.ravel()
            return density_func(x)
            
    elif d == 2:
        # 2D evaluation grid
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min
        margin1 = 0.1 * x1_range
        margin2 = 0.1 * x2_range
        
        x1_grid = np.linspace(x1_min - margin1, x1_max + margin1, 50)
        x2_grid = np.linspace(x2_min - margin2, x2_max + margin2, 50)
        
        # Create meshgrid and flatten for evaluation
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        eval_points = np.column_stack([X1.ravel(), X2.ravel()])
        
        # Compute density
        density_flat = kernel_density_estimator(eval_points, X, bandwidth, enable_boundary_correction, domain_bounds)
        density_values = density_flat.reshape(X1.shape)
        
        # Normalize with numerical stability check
        dx1 = x1_grid[1] - x1_grid[0]
        dx2 = x2_grid[1] - x2_grid[0]
        integral = np.sum(density_values) * dx1 * dx2
        
        # Check for numerical issues
        if integral <= 0 or np.isnan(integral) or np.isinf(integral):
            # Fallback to uniform distribution
            density_values_normalized = np.ones_like(density_values) / (dx1 * dx2 * density_values.size)
        else:
            density_values_normalized = density_values / integral
        
        # Create 2D interpolation function
        density_func = RegularGridInterpolator(
            (x1_grid, x2_grid), density_values_normalized,
            bounds_error=False, fill_value=0.0
        )
        
        def kde_function(x):
            x = np.array(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return density_func(x)
    
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")
    
    return kde_function


# ============================================================================
# Main Interface Functions (Unified)
# ============================================================================

def KDE_plugin(X_sample, enable_boundary_correction=None, domain_bounds=None):
    """
    Kernel density estimation with plug-in bandwidth selection and optional boundary correction.
    
    Parameters:
        X_sample (array): Sample data with shape (n, d) where d ∈ {1, 2}
        enable_boundary_correction (bool or None): Whether to apply boundary correction
                                                 - None: default True for 1D, False for 2D
                                                 - True/False: explicitly enable/disable
        domain_bounds (tuple or list): Domain bounds for boundary correction
        
    Returns:
        function: Density function with automatically selected bandwidth using plug-in method
    """
    # Use plug-in method for bandwidth selection
    kde_function = kde_density_estimation(X_sample, bandwidth=None, 
                                         enable_boundary_correction=enable_boundary_correction,
                                         domain_bounds=domain_bounds)
    
    return kde_function


# ============================================================================
# GCV-based Bandwidth Selection
# ============================================================================

def gcv_constant_selection(X, c_candidates):
    """
    GCV-based constant selection for bandwidth calculation.
    
    Parameters:
        X (array): Sample data with shape (n, d)
        c_candidates (array or tuple): Candidate values for constant c
        
    Returns:
        float or tuple: Best constant(s) c based on GCV score
    """
    X = np.array(X)
    n, d = X.shape
    best_c = None
    best_score = np.inf
    
    if d == 1:
        # 1D case: h = n**(-1/5) * c
        for c in c_candidates:
            try:
                # Calculate bandwidth using the formula
                h = n**(-1/5) * c
                h = max(h, 0.01)  # Minimum bandwidth
                
                # Get density estimation with this bandwidth
                kde_func = kde_density_estimation(X, bandwidth=h, enable_boundary_correction=False)
                
                # Compute density values at sample points
                density = kde_func(X)
                
                # Ensure density values are positive
                density = np.maximum(density, 1e-10)
                
                # Calculate GCV score
                residual = (density - np.mean(density)) ** 2
                trace = np.sum((density / np.mean(density)) ** 2)  # Trace of S_lambda
                gcv_score = np.mean(residual) / (1 - trace / n) ** 2
                
                # Check for numerical stability
                if np.isfinite(gcv_score) and gcv_score < best_score:
                    best_score = gcv_score
                    best_c = c
                    
            except Exception:
                # Skip this candidate if calculation fails
                continue
                
        # Return default if no valid candidate found
        if best_c is None:
            sigma = np.std(X[:, 0])
            best_c = sigma
            
    elif d == 2:
        # 2D case: h1 = n**(-1/6) * c1, h2 = n**(-1/6) * c2
        for c_pair in c_candidates:
            try:
                if isinstance(c_pair, (tuple, list, np.ndarray)) and len(c_pair) == 2:
                    c1, c2 = c_pair
                else:
                    # If single value provided, use for both dimensions
                    c1 = c2 = c_pair
                
                # Calculate bandwidths using the formula
                h1 = n**(-1/6) * c1
                h2 = n**(-1/6) * c2
                h1 = max(h1, 0.01)  # Minimum bandwidth
                h2 = max(h2, 0.01)  # Minimum bandwidth
                
                # Get density estimation with these bandwidths
                kde_func = kde_density_estimation(X, bandwidth=(h1, h2), enable_boundary_correction=False)
                
                # Compute density values at sample points
                density = kde_func(X)
                
                # Ensure density values are positive
                density = np.maximum(density, 1e-10)
                
                # Calculate GCV score
                residual = (density - np.mean(density)) ** 2
                trace = np.sum((density / np.mean(density)) ** 2)  # Trace of S_lambda
                gcv_score = np.mean(residual) / (1 - trace / n) ** 2
                
                # Check for numerical stability
                if np.isfinite(gcv_score) and gcv_score < best_score:
                    best_score = gcv_score
                    best_c = (c1, c2)
                    
            except Exception:
                # Skip this candidate if calculation fails
                continue
                
        # Return default if no valid candidate found
        if best_c is None:
            sigma1 = np.std(X[:, 0])
            sigma2 = np.std(X[:, 1])
            best_c = (sigma1, sigma2)
    
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")
        
    return best_c


def KDE_GCV(X_sample, enable_boundary_correction=None, domain_bounds=None):
    """
    Kernel density estimation with GCV-based constant selection and optional boundary correction.
    
    Parameters:
        X_sample (array): Sample data with shape (n, d) where d ∈ {1, 2}
        enable_boundary_correction (bool or None): Whether to apply boundary correction
                                                 - None: default True for 1D, False for 2D
                                                 - True/False: explicitly enable/disable
        domain_bounds (tuple or list): Domain bounds for boundary correction
        
    Returns:
        function: Density function with GCV-optimized bandwidth
    """
    X_sample = np.array(X_sample)
    n, d = X_sample.shape
    
    if d == 1:
        # Generate candidate constants for 1D case
        sigma = np.std(X_sample[:, 0])
        # Use log-spaced candidates around the standard deviation
        c_candidates = np.logspace(-2, 2, 30) * sigma  # From 0.01*sigma to 100*sigma
        
        # Find best constant using GCV
        best_c = gcv_constant_selection(X_sample, c_candidates)
        
        # Calculate final bandwidth
        h = n**(-1/5) * best_c
        h = max(h, 0.01)
        
        # Create density function
        kde_func = kde_density_estimation(X_sample, bandwidth=h,
                                         enable_boundary_correction=enable_boundary_correction,
                                         domain_bounds=domain_bounds)
        
    elif d == 2:
        # Generate candidate constants for 2D case
        sigma1 = np.std(X_sample[:, 0])
        sigma2 = np.std(X_sample[:, 1])
        
        # Create grid of candidate constants
        c1_candidates = np.logspace(-1, 1, 10) * sigma1  # From 0.1*sigma1 to 10*sigma1
        c2_candidates = np.logspace(-1, 1, 10) * sigma2  # From 0.1*sigma2 to 10*sigma2
        
        # Create all combinations
        c_candidates = [(c1, c2) for c1 in c1_candidates for c2 in c2_candidates]
        
        # Find best constants using GCV
        best_c1, best_c2 = gcv_constant_selection(X_sample, c_candidates)
        
        # Calculate final bandwidths
        h1 = n**(-1/6) * best_c1
        h2 = n**(-1/6) * best_c2
        h1 = max(h1, 0.01)
        h2 = max(h2, 0.01)
        
        # Create density function
        kde_func = kde_density_estimation(X_sample, bandwidth=(h1, h2),
                                         enable_boundary_correction=enable_boundary_correction,
                                         domain_bounds=domain_bounds)
        
    else:
        raise ValueError("Only 1D and 2D dimensions are supported")
    
    return kde_func


def KDE(X_sample, enable_boundary_correction=None, domain_bounds=None):
    """
    Default kernel density estimation interface with boundary correction support. Uses GCV-based method.
    
    Parameters:
        X_sample (array): Sample data with shape (n, d) where d ∈ {1, 2}
        enable_boundary_correction (bool or None): Whether to apply boundary correction
                                                 - None: default True for 1D, False for 2D
                                                 - True/False: explicitly enable/disable
        domain_bounds (tuple or list): Domain bounds for boundary correction
                                     - 1D: (a, b) default: (0, 1)
                                     - 2D: [(a1, b1), (a2, b2)] default: [(0, 1), (0, 1)]
        
    Returns:
        function: Density function with automatically selected bandwidth and boundary correction
    """
    return KDE_GCV(X_sample, enable_boundary_correction=enable_boundary_correction, 
                   domain_bounds=domain_bounds)




