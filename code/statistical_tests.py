"""
Statistical testing module for local validation framework.

This module provides functions for computing test statistics, p-values,
and conducting statistical tests for both 1D and 2D cases using KDE density estimation.
"""

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

# Relative imports for package modules
from .fourier_basis import (get_basis_frequency, create_1d_basis_function, 
                           create_decay_weight_function, compute_orthonormal_fourier_coefficients)
from .kernel_ridge import kernel_ridge_regression
from .kde import KDE
from .data_generation import generate_samples
from .functions import get_scenario_functions


# =============================================================================
# CUMULATIVE DISTRIBUTION FUNCTIONS (CDFs)
# =============================================================================

def compute_1d_max_statistic_cdf(t, max_frequency, weight_function):
    """
    Compute the product CDF for max |T_i| in 1D case.
    
    For the 1D case, F(t) = ∏_{j=0}^{g(k)-1}(2Φ(t/ρᵢ)-1) where ρᵢ is the decay 
    coefficient for the frequency of the j-th basis function.
    
    Parameters:
        t (float): Test statistic threshold value
        max_frequency (int): Maximum frequency index to consider
        weight_function (callable): Weight function ρ_i(basis_index)
        
    Returns:
        float: Cumulative probability P(max |T_i| ≤ t)
    """
    from .fourier_basis import get_basis_frequency
    
    product = 1.0
    basis_index = 0
    
    # Frequency 0: 1 basis function (constant)
    if max_frequency >= 0:
        freq_0_decay = 1.0 / np.log(0 + 2)  # ρ₀ = 1/log(2)
        product *= 2 * stats.norm.cdf(t / freq_0_decay) - 1
        basis_index += 1
    
    # Frequencies k ≥ 1: each has 2 basis functions (cos, sin)
    for k in range(1, max_frequency + 1):
        freq_k_decay = 1.0 / np.log(k + 2)  # ρₖ = 1/log(k+2)
        base_term = 2 * stats.norm.cdf(t / freq_k_decay) - 1
        # Each frequency k has 2 basis functions
        product *= base_term ** 2
        basis_index += 2
    
    return product


def compute_2d_max_statistic_cdf(t, weight_function, max_frequency=10):
    """
    Compute the product CDF for max |T_ijk| in 2D case.
    
    For the 2D case:
    - Frequency 0: 1 basis function
    - Frequency k ≥ 1: 4k basis functions 
    - F(t) = (2Φ(t/ρ₀)-1) × ∏_{k=1}^{max_freq} (2Φ(t/ρₖ)-1)^{4k}
    where ρₖ is determined by the weight function
    
    Parameters:
        t (float): Test statistic threshold value
        weight_function (callable): Weight function ρ(frequency_sum) that computes decay coefficients
        max_frequency (int): Maximum frequency index to consider
        
    Returns:
        float: Cumulative probability P(max |T_ijk| ≤ t)
    """
    product = 1.0
    
    # Use default decay if weight_function is None (for backward compatibility)
    if weight_function is None:
        def default_weight_function(k):
            return 1.0 / np.log(k + 2)
        weight_function = default_weight_function
    
    # Frequency 0: 1 basis function (constant)
    if max_frequency >= 0:
        freq_0_decay = weight_function(0)
        product *= 2 * stats.norm.cdf(t / freq_0_decay) - 1
    
    # Frequencies k ≥ 1: each frequency k has 4k basis functions
    for k in range(1, max_frequency + 1):
        freq_k_decay = weight_function(k)
        base_term = 2 * stats.norm.cdf(t / freq_k_decay) - 1
        # Frequency k has 4k basis functions
        product *= base_term ** (4 * k)
    
    return product


# =============================================================================
# MAXIMUM STATISTIC COMPUTATION
# =============================================================================

def compute_1d_max_statistic(max_frequency, statistic_function):
    """
    Compute the maximum absolute test statistic in 1D case.
    
    For 1D Fourier basis:
    - Index 0: constant term (1 basis function)
    - Index 1,2: frequency 1 (cos, sin - 2 basis functions)  
    - Index 3,4: frequency 2 (cos, sin - 2 basis functions)
    - ...
    - Index 2k-1, 2k: frequency k (cos, sin - 2 basis functions)
    
    Total basis functions for max_frequency k: 1 + 2*k = 2*k + 1
    So indices range from 0 to 2*max_frequency (inclusive)
    
    Parameters:
        max_frequency (int): Maximum frequency index to consider
        statistic_function (callable): Function T_i(i) that computes individual statistics
        
    Returns:
        float: max_i |T_i(i)|
    """
    # Include index 0 (constant term) and all basis functions up to max_frequency
    # Total indices: 0, 1, 2, ..., 2*max_frequency
    return np.max([abs(statistic_function(i)) for i in range(2 * max_frequency + 1)])


def compute_2d_max_statistic(statistic_function, max_frequency=10):
    """
    Compute the maximum absolute test statistic in 2D case.
    
    For the 2D case:
    - Frequency 0: 1 basis function
    - Frequency k ≥ 1: 4k basis functions
    
    We organize by frequency levels, not by (i,j) pairs.
    
    Parameters:
        statistic_function (callable): Function T(i,j,kx,ky) that computes individual statistics
        max_frequency (int): Maximum frequency index to consider
        
    Returns:
        float: max_{all basis functions} |T|
    """
    max_values = []
    
    # Frequency 0: only (0,0) with cos-cos (constant term)
    if max_frequency >= 0:
        max_values.append(abs(statistic_function(0, 0, 0, 0)))
    
    # Frequencies k ≥ 1: each frequency k has 4k basis functions
    for k in range(1, max_frequency + 1):
        # For frequency k, we need to find all (i,j) pairs where i+j = k
        # and count 4k basis functions total
        
        # Edge cases: (k,0) and (0,k) - each contributes 2 functions
        # (k,0): cos and sin in x-direction only
        for kx in (0, 1):  # cos, sin
            max_values.append(abs(statistic_function(k, 0, kx, 0)))
        
        # (0,k): cos and sin in y-direction only  
        for ky in (0, 1):  # cos, sin
            max_values.append(abs(statistic_function(0, k, 0, ky)))
        
        # Mixed terms: (i,j) where i,j > 0 and i+j = k
        # Each contributes 4 functions (all cos/sin combinations)
        for i in range(1, k):
            j = k - i
            if j > 0:  # Both i,j > 0
                for kx in (0, 1):  # cos, sin in x
                    for ky in (0, 1):  # cos, sin in y
                        max_values.append(abs(statistic_function(i, j, kx, ky)))
    
    return max(max_values) if max_values else 0.0


# =============================================================================
# 1D TEST STATISTICS WITH KDE (PRIMARY FUNCTIONS)
# =============================================================================

def compute_1d_test_statistic_kde_quadrature(n_samples, weight_function, sigma_hat, 
                                           domain_bounds, fitted_function, true_function,
                                           density_estimate):
    """
    Compute 1D test statistic using numerical quadrature integration with KDE density weighting.
    
    This function computes the test statistic T_i via exact numerical integration with
    KDE density estimation. It's more accurate but slower than the FFT-based approach.
    
    Parameters:
        n_samples (int): Number of samples in the dataset
        weight_function (callable): Weight function ρ_i(i)
        sigma_hat (float): Estimated noise standard deviation
        domain_bounds (array): Integration domain boundaries [(a,b)]
        fitted_function (callable): Fitted regression function f̂(x)
        true_function (callable): True function y(x)
        density_estimate (callable): Estimated density function p̂(x)
        
    Returns:
        function: Test statistic function T_i(i) that computes the i-th statistic
    """
    def test_statistic(i):
        # Scaling factor for the test statistic
        scaling_factor = np.sqrt(n_samples) * weight_function(i) / sigma_hat
        
        # Get the i-th basis function
        basis_function = create_1d_basis_function(i)
        
        # Define integrand: (f̂(x) - y(x)) * √p̂(x) * h_i(x)
        def integrand(*x):
            x_array = np.array(x).reshape(1, -1)
            function_diff = fitted_function(x_array) - true_function(x_array)
            density_weight = np.sqrt(density_estimate(x_array))
            basis_value = basis_function(x_array)
            return function_diff * density_weight * basis_value
        
        # Compute the integral over the domain
        integral_value, _ = integrate.nquad(integrand, domain_bounds)
        
        return scaling_factor * integral_value
    
    return test_statistic


def compute_1d_test_statistic_kde_fft(n_samples, weight_function, sigma_hat, 
                                     domain_bounds, fitted_function, true_function, 
                                     density_estimate, grid_points=1024):
    """
    Compute 1D test statistic using FFT with KDE density weighting.
    
    This is the primary 1D test statistic function that incorporates 
    density estimation for general (non-uniform) distributions.
    
    Parameters:
        n_samples (int): Number of samples in the dataset
        weight_function (callable): Weight function ρ_i(i)
        sigma_hat (float): Estimated noise standard deviation
        domain_bounds (array): Integration domain boundaries [(a,b)]
        fitted_function (callable): Fitted regression function f̂(x)
        true_function (callable): True function y(x)
        density_estimate (callable): Estimated density function p̂(x)
        grid_points (int): Number of grid points for FFT computation
        
    Returns:
        function: Test statistic function T_i(i) that computes the i-th statistic
    """
    # Create uniform grid over the domain
    a, b = domain_bounds[0]
    grid_coords = np.linspace(a, b, grid_points, endpoint=False).reshape(-1, 1)
    
    # Compute function values - ensure consistent 1D arrays
    fitted_vals = fitted_function(grid_coords)
    true_vals = true_function(grid_coords)
    density_vals = density_estimate(grid_coords)
    
    # Force conversion to 1D arrays with proper shapes
    if fitted_vals.ndim > 1:
        fitted_vals = fitted_vals.ravel()
    if true_vals.ndim > 1:
        true_vals = true_vals.ravel()
    if density_vals.ndim > 1:
        density_vals = density_vals.ravel()
        
    # Ensure all arrays have exactly the expected length
    expected_length = grid_points
    if len(fitted_vals) != expected_length:
        fitted_vals = fitted_vals[:expected_length] if len(fitted_vals) > expected_length else np.pad(fitted_vals, (0, expected_length - len(fitted_vals)))
    if len(true_vals) != expected_length:
        true_vals = true_vals[:expected_length] if len(true_vals) > expected_length else np.pad(true_vals, (0, expected_length - len(true_vals)))
    if len(density_vals) != expected_length:
        density_vals = density_vals[:expected_length] if len(density_vals) > expected_length else np.pad(density_vals, (0, expected_length - len(density_vals)))
    
    # Compute weighted function difference: (f̂ - y) * √p̂
    function_diff = fitted_vals - true_vals
    density_weights = np.sqrt(density_vals)
    weighted_diff = function_diff * density_weights
    
    # Apply FFT
    fourier_coeffs = np.fft.fft(weighted_diff) / grid_points
    
    def test_statistic(i):
        # Scaling factor
        scaling_factor = np.sqrt(n_samples) * weight_function(i) / sigma_hat
        
        # Get frequency and component type
        k, component_type = get_basis_frequency(i)
        
        # Extract the appropriate Fourier coefficient with proper normalization
        domain_length = b - a
        if component_type == "constant":
            coefficient = fourier_coeffs[0].real / np.sqrt(1 / domain_length)
        elif component_type == "cos":
            coefficient = (fourier_coeffs[k].real + fourier_coeffs[-k].real) / np.sqrt(2 / domain_length)
        else:  # sin
            coefficient = -(fourier_coeffs[k].imag - fourier_coeffs[-k].imag) / np.sqrt(2 / domain_length)
        
        return scaling_factor * coefficient
    
    return test_statistic


def compute_1d_test_statistic_kde_subdomain(n_samples, weight_function, sigma_hat,
                                           domain_bounds, fitted_function, true_function,
                                           density_estimate, grid_points=1024, n_subdomains=3):
    """
    Compute 1D subdomain test statistics using FFT with KDE density weighting.
    
    This function splits the domain into equal subintervals and computes
    test statistics on each subdomain for local analysis.
      
    Parameters:
        n_samples (int): Number of samples in the dataset
        weight_function (callable): Weight function ρ_i(i)
        sigma_hat (float): Estimated noise standard deviation
        domain_bounds (array): Integration domain boundaries [(a,b)]
        fitted_function (callable): Fitted regression function f̂(x)
        true_function (callable): True function y(x)
        density_estimate (callable): Estimated density function p̂(x)
        grid_points (int): Number of grid points for FFT computation
        n_subdomains (int): Number of subdomains to create
        
    Returns:
        function: Subdomain test statistic function T_ij(i, j) where j is subdomain index
    """
    # Split domain into equal subintervals
    a, b = domain_bounds[0]
    subdomain_points = np.linspace(a, b, n_subdomains + 1)
    
    # Create grids for each subdomain
    subdomain_grids = []
    for k in range(n_subdomains):
        grid = np.linspace(subdomain_points[k], subdomain_points[k+1], 
                          grid_points, endpoint=False).reshape(-1, 1)
        subdomain_grids.append(grid)
    
    # Compute weighted residuals for each subdomain
    def compute_weighted_residual(grid):
        fitted_vals = fitted_function(grid)
        true_vals = true_function(grid)
        density_vals = density_estimate(grid)
        
        # Force conversion to 1D arrays
        if fitted_vals.ndim > 1:
            fitted_vals = fitted_vals.ravel()
        if true_vals.ndim > 1:
            true_vals = true_vals.ravel()
        if density_vals.ndim > 1:
            density_vals = density_vals.ravel()
            
        # Ensure consistent lengths
        expected_length = grid_points
        if len(fitted_vals) != expected_length:
            fitted_vals = fitted_vals[:expected_length] if len(fitted_vals) > expected_length else np.pad(fitted_vals, (0, expected_length - len(fitted_vals)))
        if len(true_vals) != expected_length:
            true_vals = true_vals[:expected_length] if len(true_vals) > expected_length else np.pad(true_vals, (0, expected_length - len(true_vals)))
        if len(density_vals) != expected_length:
            density_vals = density_vals[:expected_length] if len(density_vals) > expected_length else np.pad(density_vals, (0, expected_length - len(density_vals)))
        
        function_diff = fitted_vals - true_vals
        density_weights = np.sqrt(density_vals)
        return function_diff * density_weights
    
    # Apply FFT to each subdomain
    fft_results = []
    for grid in subdomain_grids:
        weighted_residual = compute_weighted_residual(grid)
        fft_result = np.fft.fft(weighted_residual) / grid_points
        fft_results.append(fft_result)
    
    def subdomain_test_statistic(i, subdomain_idx):
        """
        Compute test statistic for frequency i on subdomain subdomain_idx.
        
        Parameters:
            i (int): Frequency index
            subdomain_idx (int): Subdomain index (1-based)
        """
        if subdomain_idx < 1 or subdomain_idx > n_subdomains:
            raise ValueError(f"Subdomain index must be between 1 and {n_subdomains}")
        
        # Get FFT result for this subdomain
        F_subdomain = fft_results[subdomain_idx - 1]
        
        # Scaling factor
        scaling_factor = np.sqrt(n_samples) * weight_function(i) / sigma_hat
        
        # Get frequency and component type
        k, component_type = get_basis_frequency(i)
        
        # Compute coefficient with subdomain-specific normalization
        subdomain_length = (b - a) / n_subdomains
        if component_type == "constant":
            coefficient = F_subdomain[0].real / np.sqrt(n_subdomains)
        elif component_type == "cos":
            coefficient = 2 * F_subdomain[k].real / np.sqrt(2 * n_subdomains)
        else:  # sin
            coefficient = -2 * F_subdomain[k].imag / np.sqrt(2 * n_subdomains)
        
        return scaling_factor * coefficient
    
    return subdomain_test_statistic


# =============================================================================
# 2D TEST STATISTICS WITH KDE
# =============================================================================

def compute_2d_test_statistic_global_quadrature(sigma_hat, fitted_function, true_function, 
                                              density_estimate, domain_bounds, n_samples, weight_function=None):
    """
    Compute 2D global test statistics using numerical quadrature integration.
    
    This function provides a reference implementation using exact numerical integration
    for validating the FFT-based approach. It's more accurate but much slower.
    
    Parameters:
        sigma_hat (float): Estimated noise standard deviation
        fitted_function (callable): Fitted regression function f̂(x,y)
        true_function (callable): True function y(x,y) 
        density_estimate (callable): Estimated density function p̂(x,y)
        domain_bounds (array): Domain boundaries [(x0,x1), (y0,y1)]
        n_samples (int): Number of samples in dataset
        weight_function (callable, optional): Weight function rho(i,j). If None, uses default decay
        
    Returns:
        function: Test statistic function T(i, j, type_x, type_y) where:
            - i, j: frequency indices
            - type_x, type_y: 0=cosine, 1=sine
    """
    from .fourier_basis import create_2d_basis_function
    
    def test_statistic(freq_i, freq_j, type_x, type_y):
        """
        Compute test statistic for given frequency and type combination using numerical integration.
        
        Parameters:
            freq_i, freq_j (int): Frequency indices
            type_x, type_y (int): Component types (0=cos, 1=sin)
        """
        # Compute decay coefficient using weight function or default
        if weight_function is not None:
            decay_coefficient = weight_function(freq_i + freq_j)
        else:
            frequency_layer = freq_i + freq_j
            decay_coefficient = 1.0 / np.log(frequency_layer + 2)  # ρₖ = 1/log(k+2)
        
        # Scaling factor using decay coefficient
        scaling_factor = np.sqrt(n_samples) * decay_coefficient / sigma_hat
        
        # Define integrand: (f̂(x,y) - y(x,y)) * √p̂(x,y) * h_{ij}(x,y)
        def integrand(x, y):
            points = np.array([[x, y]])
            function_diff = fitted_function(points) - true_function(points)
            density_weight = np.sqrt(density_estimate(points))
            
            # Compute basis function value directly
            basis_value = create_2d_basis_function(domain_bounds, freq_i, freq_j, type_x, type_y, x, y)
            
            return function_diff[0] * density_weight[0] * basis_value
        
        # Compute the integral over the 2D domain
        (x0, x1), (y0, y1) = domain_bounds
        integral_value, _ = integrate.dblquad(
            integrand, x0, x1, lambda x: y0, lambda x: y1
        )
        
        return scaling_factor * integral_value
    
    return test_statistic

def compute_2d_test_statistic_global(sigma_hat, fitted_function, true_function, 
                                   density_estimate, domain_bounds, n_samples, 
                                   grid_points=100, max_frequency=None, weight_function=None):
    """
    Compute 2D global test statistics using FFT-based Fourier analysis.
    
    This function computes test statistics for global goodness-of-fit testing
    in 2D using all four combinations of cosine/sine basis functions.
    
    Parameters:
        sigma_hat (float): Estimated noise standard deviation
        fitted_function (callable): Fitted regression function f̂(x,y)
        true_function (callable): True function y(x,y) 
        density_estimate (callable): Estimated density function p̂(x,y)
        domain_bounds (array): Domain boundaries [(x0,x1), (y0,y1)]
        n_samples (int): Number of samples in dataset
        grid_points (int): Grid resolution for FFT
        max_frequency (int, optional): Maximum frequency to consider
        weight_function (callable, optional): Weight function rho(i,j). If None, uses default decay
        
    Returns:
        function: Test statistic function T(i, j, type_x, type_y) where:
            - i, j: frequency indices
            - type_x, type_y: 0=cosine, 1=sine
    """
    # Create weighted function difference for analysis
    def weighted_function_difference(x, y):
        """Compute (f̂ - y) * √p̂ for statistical analysis."""
        # Ensure x, y are proper arrays
        points = np.stack([x.ravel(), y.ravel()], axis=-1)
        
        # Compute weighted difference
        function_diff = (fitted_function(points) - true_function(points))
        weight = np.sqrt(density_estimate(points))
        weighted_diff = function_diff * weight
        
        return weighted_diff.reshape(x.shape)
    
    # Compute Fourier coefficients
    C_cc, C_cs, C_sc, C_ss = compute_orthonormal_fourier_coefficients(
        weighted_function_difference, domain_bounds, grid_points, max_frequency
    )

    def test_statistic(freq_i, freq_j, type_x, type_y):
        """
        Compute test statistic for given frequency and type combination.
        
        Parameters:
            freq_i, freq_j (int): Frequency indices
            type_x, type_y (int): Component types (0=cos, 1=sin)
        """
        # Compute decay coefficient using weight function or default
        if weight_function is not None:
            decay_coefficient = weight_function(freq_i + freq_j)
        else:
            frequency_layer = freq_i + freq_j
            decay_coefficient = 1.0 / np.log(frequency_layer + 2)  # ρₖ = 1/log(k+2)
        
        # Compute scaling factor using decay coefficient
        scaling_factor = np.sqrt(n_samples) * decay_coefficient / sigma_hat
        
        # Select appropriate coefficient based on component types
        if type_x == 0 and type_y == 0:      # cos-cos
            coefficient = C_cc[freq_i, freq_j]
        elif type_x == 0 and type_y == 1:    # cos-sin
            coefficient = C_cs[freq_i, freq_j]
        elif type_x == 1 and type_y == 0:    # sin-cos
            coefficient = C_sc[freq_i, freq_j]
        else:                                # sin-sin
            coefficient = C_ss[freq_i, freq_j]
        
        return scaling_factor * coefficient
    
    return test_statistic


def compute_2d_test_statistic_subdomain(sigma_hat, fitted_function, true_function, 
                                       density_estimate, domain_bounds, n_samples, 
                                       grid_points=100, max_frequency=None, weight_function=None):
    """
    Compute 2D subdomain test statistics using subdomain-specific orthonormal bases.
    
    This function implements the theoretical framework where each subdomain has its own
    orthonormal basis {h_ij}, and the test statistics are computed using these 
    subdomain-specific bases with zero extension to the full domain.
    
    Parameters:
        sigma_hat (float): Estimated noise standard deviation  
        fitted_function (callable): Fitted regression function f̂(x,y)
        true_function (callable): True function y(x,y)
        density_estimate (callable): Estimated density function p̂(x,y)
        domain_bounds (array): Domain boundaries [(x0,x1), (y0,y1)]
        n_samples (int): Number of samples in dataset
        grid_points (int): Grid resolution for FFT
        max_frequency (int, optional): Maximum frequency to consider
        weight_function (callable, optional): Weight function rho(i,j). If None, uses default decay
        
    Returns:
        function: Subdomain test statistic T(i, j, type_x, type_y, subdomain_index)
    """
    from .fourier_basis import compute_subdomain_orthonormal_fourier_coefficients
    
    # Subdivide domain into four quadrants
    (x0, x1), (y0, y1) = domain_bounds
    x_mid, y_mid = (x0 + x1) / 2, (y0 + y1) / 2
    
    subdomain_bounds = [
        ((x_mid, x1), (y_mid, y1)),   # Upper right quadrant
        ((x0, x_mid), (y_mid, y1)),   # Upper left quadrant  
        ((x0, x_mid), (y0, y_mid)),   # Lower left quadrant
        ((x_mid, x1), (y0, y_mid)),   # Lower right quadrant
    ]
    
    # Compute weighted function difference for each subdomain
    subdomain_coefficients = []
    
    for subdomain in subdomain_bounds:
        def subdomain_weighted_difference(x, y):
            """Compute (f̂ - y) * √p̂ restricted to the current subdomain."""
            # Create points array
            points = np.stack([x.ravel(), y.ravel()], axis=-1)
            
            # Compute function difference over the entire domain
            function_diff = (fitted_function(points) - true_function(points))
            
            # Apply density weighting directly to the original difference
            weight = np.sqrt(density_estimate(points))
            weighted_diff = function_diff * weight 
            
            # Restrict to subdomain: zero outside subdomain bounds
            (sub_x0, sub_x1), (sub_y0, sub_y1) = subdomain
            mask = (x >= sub_x0) & (x <= sub_x1) & (y >= sub_y0) & (y <= sub_y1)
            
            result = np.zeros_like(x)
            result[mask] = weighted_diff.reshape(x.shape)[mask]
            
            return result
        
        # Compute Fourier coefficients using subdomain's own orthonormal basis
        C_cc, C_cs, C_sc, C_ss = compute_subdomain_orthonormal_fourier_coefficients(
            subdomain_weighted_difference, subdomain, grid_points, max_frequency
        )
        

        subdomain_coefficients.append((C_cc, C_cs, C_sc, C_ss))

    def subdomain_test_statistic(freq_i, freq_j, type_x, type_y, subdomain_index):
        """
        Compute subdomain test statistic using subdomain-specific orthonormal basis.
        
        Parameters:
            freq_i, freq_j (int): Frequency indices in subdomain's own basis
            type_x, type_y (int): Component types (0=cos, 1=sin)  
            subdomain_index (int): Index of subdomain (0-3)
        """
        # Get coefficients for this subdomain
        C_cc, C_cs, C_sc, C_ss = subdomain_coefficients[subdomain_index]
        
        # Compute decay coefficient using weight function or default
        if weight_function is not None:
            decay_coefficient = weight_function(freq_i + freq_j)
        else:
            frequency_layer = freq_i + freq_j
            decay_coefficient = 1.0 / np.log(frequency_layer + 2)  # ρₖ = 1/log(k+2)
        
        # Compute scaling factor using decay coefficient
        scaling_factor = np.sqrt(n_samples) * decay_coefficient / sigma_hat
        
        # Select appropriate coefficient based on component types
        if type_x == 0 and type_y == 0:      # cos-cos
            coefficient = C_cc[freq_i, freq_j]
        elif type_x == 0 and type_y == 1:    # cos-sin
            coefficient = C_cs[freq_i, freq_j]
        elif type_x == 1 and type_y == 0:    # sin-cos
            coefficient = C_sc[freq_i, freq_j]
        else:                                # sin-sin
            coefficient = C_ss[freq_i, freq_j]
        
        return scaling_factor * coefficient
    
    return subdomain_test_statistic


# =============================================================================
# P-VALUE COMPUTATION FUNCTIONS WITH KDE
# =============================================================================

def compute_1d_pvalue_kde(n_samples, domain_bounds, sigma, noise_type, function_type,
                         max_frequency, weight_decay_type, distribution_type,
                         use_mse=True, modulation_strength=0, 
                         subdomain_flags=(False, False, False), weight_decay_n=1.0):
    """
    Compute p-values for 1D goodness-of-fit testing with KDE density estimation.
    
    This is the main function for 1D statistical testing that handles general
    distributions using KDE for density estimation.
    
    Parameters:
        n_samples (int): Number of samples in the dataset
        domain_bounds (array): Domain boundaries [(a,b)]
        sigma (float): Noise standard deviation
        noise_type (str): Type of noise distribution
        function_type (str): Type of test function
        max_frequency (int): Maximum frequency for basis functions
        weight_decay_type (str): Type of weight decay ("log" or "poly")
        distribution_type (str): Type of input distribution
        use_mse (bool): Whether to use MSE for validation
        modulation_strength (float): Strength of function modulation
        subdomain_flags (tuple): Flags for subdomain-specific modulation
        weight_decay_n (float): Weight decay parameter (default 1.0)
        
    Returns:
        tuple: (subdomain1_pval, subdomain2_pval, subdomain3_pval, global_pval)
    """
    # Determine hypothesis based on modulation_strength
    hypothesis = 'H0' if modulation_strength == 0 else 'H1'
    
    # H1: generate modulated data but compare against unmodulated function
    if hypothesis == 'H0':
        # H0: generate unmodulated data and compare against unmodulated function
        data_hypothesis = 'H0'
        data_modulation = 0
        comparison_hypothesis = 'H0'
        comparison_modulation = 0
    else:
        # H1: generate modulated data but compare against unmodulated function
        data_hypothesis = 'H1'      # Generate H1 data (with modulation)
        data_modulation = modulation_strength  # Use modulation for data generation
        comparison_hypothesis = 'H0' # But compare against H0 scenario (unmodulated)
        comparison_modulation = 0    # No modulation for comparison function
    
    # Generate samples using the logic
    X_sample, y_sample = generate_samples(
        omega=domain_bounds, 
        n_samples=n_samples, 
        scenario_name=function_type,
        hypothesis=data_hypothesis,
        dimension='1D',
        sigma=sigma,
        noise_type=noise_type,
        dist_type=distribution_type,
        c=data_modulation
    )
    
    # Fit kernel ridge regression
    final_krr, sigma_hat = kernel_ridge_regression(X_sample, y_sample)
    fitted_function = final_krr.predict
    
    # Get the comparison functions using the logic
    scenario = get_scenario_functions(function_type, comparison_hypothesis, '1D', c=comparison_modulation)
    
    # Select appropriate reference function based on hypothesis logic
    if hypothesis == 'H0':
        true_function = scenario['f1']  # H0: compare against f1 (unmodulated)
    else:
        true_function = scenario['f1']  # H1: compare against f1 (unmodulated function, zero hypothesis)
    
    # Estimate density using KDE
    density_estimate = KDE(X_sample)
    
    # Create weight function with custom decay parameter
    weight_function = create_decay_weight_function(weight_decay_n, weight_decay_type)
    
    # Compute global test statistic using FFT
    T_global = compute_1d_test_statistic_kde_fft(
        n_samples, weight_function, sigma_hat, domain_bounds,
        fitted_function, true_function, density_estimate
    )
    
    # Compute subdomain test statistics using FFT
    T_subdomain = compute_1d_test_statistic_kde_subdomain(
        n_samples, weight_function, sigma_hat, domain_bounds,
        fitted_function, true_function, density_estimate
    )
    
    # Compute maximum statistics
    frequency_limit = int(np.sqrt(n_samples))
    global_max_stat = compute_1d_max_statistic(frequency_limit, T_global)
    
    subdomain_max_stats = {}
    for j in range(1, 4):  # 3 subdomains
        subdomain_func = lambda i, j=j: T_subdomain(i, j)
        subdomain_max_stats[f'subdomain_{j}'] = compute_1d_max_statistic(frequency_limit, subdomain_func)
    
    # Compute p-values using CDF
    # Use consistent frequency_limit for statistic and CDF
    global_pval = 1 - compute_1d_max_statistic_cdf(global_max_stat, frequency_limit, weight_function)
    
    subdomain_pvals = []
    for j in range(1, 4):
        stat = subdomain_max_stats[f'subdomain_{j}']
        pval = 1 - compute_1d_max_statistic_cdf(stat, frequency_limit, weight_function)
        subdomain_pvals.append(pval)
    
    return tuple(subdomain_pvals + [global_pval])


def compute_2d_pvalue_kde(
    n_samples,
    subdomain_modulations=(False, False, False, False),
    modulation_strength=0,
    use_fft=True,
    scenario_name=None,
    sigma=0.1,
    noise_type='gaussian',
    dist_type='uniform',
    weight_decay_n=1.0,
    weight_decay_type='log',
):
    """
    Compute p-values for 2D goodness-of-fit testing with KDE density estimation.
    This function supports flexible scenario selection for global and multi-quad modulation.

    Parameters:
        n_samples (int): Number of samples in the dataset
        subdomain_modulations (tuple): Boolean flags indicating which subdomains have modulation applied 
                                       (upper-right, upper-left, lower-left, lower-right)
        modulation_strength (float): Overall modulation strength
        use_fft (bool): Whether to use FFT (faster) or numerical integration (more reliable)
        scenario_name (str, optional): Explicit scenario name for data generation. If None, use auto-detection.
        sigma (float): Noise standard deviation (default 0.1)
        noise_type (str): Type of noise distribution (default 'gaussian')
        dist_type (str): Type of input distribution (default 'uniform')
        weight_decay_n (float): Weight decay parameter (default 1.0)
        weight_decay_type (str): Type of weight decay ("log" or "poly", default 'log')
    Returns:
        tuple: (subdomain1_pval, subdomain2_pval, subdomain3_pval, subdomain4_pval, global_pval)
    """
    # Domain for 2D case
    domain_bounds = np.array([[0, 1], [0, 1]])

    # Determine hypothesis based on modulation_strength
    hypothesis = 'H0' if modulation_strength == 0 else 'H1'

    # Scenario selection logic
    if hypothesis == 'H0':
        # H0: Always use unmodulated scenario
        use_scenario = scenario_name if scenario_name is not None else 'exp_2d_vs_quad1_modulation'
        X_sample, y_sample = generate_samples(
            omega=domain_bounds,
            n_samples=n_samples,
            scenario_name=use_scenario,
            hypothesis='H0',
            dimension='2D',
            sigma=sigma,
            noise_type=noise_type,
            dist_type=dist_type,
            c=0
        )
        scenario = get_scenario_functions(use_scenario, 'H0', '2D', c=0)
        true_function = scenario['f1']
    else:
        # H1: Use explicit scenario_name if provided, otherwise auto-detect
        if scenario_name is not None:
            use_scenario = scenario_name
        else:
            has_mod_quadrant1, has_mod_quadrant2, has_mod_quadrant3, has_mod_quadrant4 = subdomain_modulations
            if has_mod_quadrant1 and not any([has_mod_quadrant2, has_mod_quadrant3, has_mod_quadrant4]):
                use_scenario = 'exp_2d_vs_quad1_modulation'
            elif has_mod_quadrant2 and not any([has_mod_quadrant1, has_mod_quadrant3, has_mod_quadrant4]):
                use_scenario = 'exp_2d_vs_quad2_modulation'
            elif has_mod_quadrant3 and not any([has_mod_quadrant1, has_mod_quadrant2, has_mod_quadrant4]):
                use_scenario = 'exp_2d_vs_quad3_modulation'
            elif has_mod_quadrant4 and not any([has_mod_quadrant1, has_mod_quadrant2, has_mod_quadrant3]):
                use_scenario = 'exp_2d_vs_quad4_modulation'
            elif has_mod_quadrant1 and has_mod_quadrant2 and not any([has_mod_quadrant3, has_mod_quadrant4]):
                use_scenario = 'exp_2d_vs_multi_quad_modulation'
            elif all([has_mod_quadrant1, has_mod_quadrant2, has_mod_quadrant3, has_mod_quadrant4]):
                use_scenario = 'exp_2d_vs_all_quad_modulation'
            else:
                # Fallback to quad1 if ambiguous
                use_scenario = 'exp_2d_vs_quad1_modulation'
        X_sample, y_sample = generate_samples(
            omega=domain_bounds,
            n_samples=n_samples,
            scenario_name=use_scenario,
            hypothesis='H1',
            dimension='2D',
            sigma=sigma,
            noise_type=noise_type,
            dist_type=dist_type,
            c=modulation_strength
        )
        # Always compare against unmodulated function for H1
        scenario = get_scenario_functions(use_scenario if hypothesis == 'H0' else 'exp_2d_vs_quad1_modulation', 'H0', '2D', c=0)
        true_function = scenario['f1']

    # Fit kernel ridge regression
    final_krr, sigma_hat = kernel_ridge_regression(X_sample, y_sample)
    fitted_function = final_krr.predict

    # Density estimation
    if n_samples < 200:
        density_estimate = lambda pts: np.ones(pts.shape[0])
    else:
        try:
            from .kde import KDE
            density_estimate = KDE(X_sample, bandwidth_factor=1.5)
        except:
            density_estimate = lambda pts: np.ones(pts.shape[0])

    # Create weight function with custom decay parameter
    weight_function = create_decay_weight_function(weight_decay_n, weight_decay_type)

    # Compute test statistics
    if use_fft:
        T_global = compute_2d_test_statistic_global(
            sigma_hat, fitted_function, true_function, density_estimate, 
            domain_bounds, n_samples, weight_function=weight_function
        )
        T_subdomain = compute_2d_test_statistic_subdomain(
            sigma_hat, fitted_function, true_function, density_estimate,
            domain_bounds, n_samples, weight_function=weight_function
        )
    else:
        T_global = compute_2d_test_statistic_global_quadrature(
            sigma_hat, fitted_function, true_function, density_estimate,
            domain_bounds, n_samples, weight_function=weight_function
        )
        T_subdomain = lambda i, j, kx, ky, z: T_global(i, j, kx, ky)

    frequency_limit = int(np.sqrt(n_samples))
    global_max_stat = compute_2d_max_statistic(T_global, frequency_limit)

    subdomain_pvals = []
    subdomain_max_stats = []
    for z in range(4):
        T_z = lambda i, j, kx, ky, _z=z: T_subdomain(i, j, kx, ky, _z)
        subdomain_max_stat = compute_2d_max_statistic(T_z, frequency_limit)
        subdomain_max_stats.append(subdomain_max_stat)
        subdomain_pval = 1 - compute_2d_max_statistic_cdf(subdomain_max_stat, weight_function, frequency_limit)
        subdomain_pvals.append(subdomain_pval)

    global_pval = 1 - compute_2d_max_statistic_cdf(global_max_stat, weight_function, frequency_limit)
    return tuple(subdomain_pvals + [global_pval])

