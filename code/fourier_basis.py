"""
Fourier basis functions and FFT-based computation module.

This module provides a comprehensive framework for Fourier analysis and statistical testing,
including 1D and 2D Fourier basis functions, frequency mapping, and FFT-based test statistics.
The module is organized into four main sections:
1. 1D Fourier basis functions and utilities
2. 1D test statistics computation
3. 2D Fourier basis functions and utilities  
4. 2D test statistics computation
"""

import numpy as np


# =============================================================================
# 1D FOURIER BASIS FUNCTIONS AND UTILITIES
# =============================================================================

def get_basis_frequency(i):
    """
    Map 1D basis index to frequency and component type.
    
    This function converts a linear basis index to the corresponding
    frequency and trigonometric component (constant, cosine, or sine).
    The mapping follows the standard ordering:
    - i=0: frequency 0, constant term (1)
    - i=1: frequency 1, cos term (√2cos2πx)
    - i=2: frequency 1, sin term (√2sin2πx) 
    - i=3: frequency 2, cos term (√2cos4πx)
    - i=4: frequency 2, sin term (√2sin4πx)
    
    Parameters:
        i (int): Basis index (0-based)
        
    Returns:
        tuple: (frequency, component_type) where:
            - frequency (int): The frequency k
            - component_type (str): "constant", "cos", or "sin"
    """
    if i == 0:
        return 0, "constant"
    else:
        m = i - 1  # Adjust for 0-based indexing after constant term
        frequency = (m // 2) + 1
        component_type = "cos" if m % 2 == 0 else "sin"
        return frequency, component_type


def create_1d_basis_function(i):
    """
    Create the i-th real-valued sine-cosine basis function h_i(x).
    
    This generates orthonormal basis functions for the 1D case,
    where x represents the norm of input vectors.
    The basis functions follow the standard ordering:
    - i=0: 1 (constant term, frequency 0)
    - i=1: √2cos(2πx) (frequency 1, cos)
    - i=2: √2sin(2πx) (frequency 1, sin)
    - i=3: √2cos(4πx) (frequency 2, cos)
    - i=4: √2sin(4πx) (frequency 2, sin)
    
    Parameters:
        i (int): Basis index
        
    Returns:
        function: Basis function h_i(x) that takes array input and returns array output
    """
    def basis_function(x):
        # Compute the norm of input vectors
        r = np.linalg.norm(x, axis=1)
        k, component_type = get_basis_frequency(i)
        
        if component_type == "constant":
            return np.ones_like(r)
        elif component_type == "cos":
            return np.sqrt(2) * np.cos(2 * np.pi * k * r)
        else:  # sin
            return np.sqrt(2) * np.sin(2 * np.pi * k * r)
    
    return basis_function


def create_decay_weight_function(n, decay_type):
    """
    Create decay weight function ρ_i for regularization.
    
    This function generates weight functions that control the relative
    importance of different frequency components in the test statistics.
    The decay coefficient follows ρᵢ:=1/log(i+2)
    
    Parameters:
        n (int): Decay parameter (higher values = faster decay)
        decay_type (str): Type of decay function
            - "log": logarithmic decay 1/log(k+2)^n
            - "poly": polynomial decay 1/k^n
        
    Returns:
        function: Weight function rho_i(i) that maps basis index to weight
        
    Raises:
        ValueError: If decay_type is not recognized
    """
    def weight_function(i):
        k, _ = get_basis_frequency(i)
        
        if decay_type == "log":
            # Standard decay: ρᵢ:=1/log(i+2)
            return 1 / (np.log(k + 2) ** n)
        elif decay_type == "poly":
            return 1 / (k ** n)
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}. Use 'log' or 'poly'.")
    
    return weight_function


# =============================================================================
# 2D FOURIER BASIS FUNCTIONS AND UTILITIES
# =============================================================================

def create_1d_orthonormal_basis(interval, frequency, component_type, x):
    """
    Create 1D orthonormal Fourier basis function on interval [a,b].
    
    Parameters:
        interval (tuple): (a, b) interval bounds
        frequency (int): Frequency index m
        component_type (int): 0 for cosine, 1 for sine
        x (array): Input points
        
    Returns:
        array: Orthonormal basis function values
    """
    a, b = interval
    length = b - a
    
    # Normalize coordinates to [0,1]
    normalized_coords = (x - a) / length
    
    # Compute normalization factor
    # For m=0 (constant): 1/√L
    # For m≥1: √(2/L)
    normalization = np.sqrt((2 - (frequency == 0)) / length)
    
    # Compute basis function
    argument = 2 * np.pi * frequency * normalized_coords
    
    if component_type == 0:  # cosine
        return normalization * np.cos(argument)
    else:  # sine
        return normalization * np.sin(argument)


def create_2d_basis_function(domain_bounds, freq_x, freq_y, type_x, type_y, x, y):
    """
    Create 2D orthonormal Fourier basis function on rectangular domain.
    
    This function creates tensor product basis functions for 2D analysis
    by combining 1D basis functions in x and y directions.
    
    Parameters:
        domain_bounds (array): Domain boundaries [(x0,x1), (y0,y1)]
        freq_x, freq_y (int): Frequency indices in x and y directions
        type_x, type_y (int): Component types (0=cosine, 1=sine)
        x, y (array): Input coordinate arrays
        
    Returns:
        array: 2D orthonormal basis function values
    """
    x_interval, y_interval = domain_bounds
    
    # Compute 1D basis functions
    basis_x = create_1d_orthonormal_basis(x_interval, freq_x, type_x, x)
    basis_y = create_1d_orthonormal_basis(y_interval, freq_y, type_y, y)
    
    # Return tensor product
    return basis_x * basis_y


def create_2d_decay_weight_function(i, j):
    """
    Create 2D decay weight function for frequency components.
    
    The decay coefficient follows ρᵢⱼ:=1/log(i+j+2) for the 2D case.
    
    Parameters:
        i, j (int): Frequency indices in x and y directions
        
    Returns:
        float: Decay weight value
    """
    return 1 / np.log(i + j + 2)


def compute_subdomain_orthonormal_fourier_coefficients(function, subdomain_bounds, grid_size, max_frequency=None):
    """
    Compute orthonormal Fourier coefficients for a function on a specific subdomain.
    
    This function computes Fourier coefficients using the subdomain's own orthonormal basis,
    as required by the theoretical framework for subdomain testing.
    
    Parameters:
        function (callable): Function f(x,y) to analyze (should be zero outside subdomain)
        subdomain_bounds (array): Subdomain boundaries [(x0,x1), (y0,y1)]
        grid_size (int): Number of grid points in each dimension
        max_frequency (int, optional): Maximum frequency to compute
        
    Returns:
        tuple: (C_cc, C_cs, C_sc, C_ss) - coefficient arrays for subdomain basis
    """
    (x0, x1), (y0, y1) = subdomain_bounds
    length_x, length_y = x1 - x0, y1 - y0
    
    if max_frequency is None:
        max_frequency = grid_size // 2
    
    # Create normalized coordinate grids for the subdomain
    u_coords = np.linspace(0, 1, grid_size, endpoint=False)
    v_coords = np.linspace(0, 1, grid_size, endpoint=False)
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    
    # Map to physical subdomain coordinates
    X_physical = x0 + length_x * U
    Y_physical = y0 + length_y * V
    function_values = function(X_physical, Y_physical)
    
    # Compute 2D FFT
    fourier_transform = np.fft.fft2(function_values) / (grid_size * grid_size)
    
    # Initialize coefficient arrays
    shape = (max_frequency + 1, max_frequency + 1)
    C_cc = np.zeros(shape)  # cos-cos coefficients
    C_cs = np.zeros(shape)  # cos-sin coefficients  
    C_sc = np.zeros(shape)  # sin-cos coefficients
    C_ss = np.zeros(shape)  # sin-sin coefficients
    
    # Extract coefficients for each frequency combination
    for m in range(max_frequency + 1):
        for n in range(max_frequency + 1):
            # Get FFT indices (with proper wrapping)
            p, q = m % grid_size, n % grid_size
            pm, qn = (-m) % grid_size, (-n) % grid_size
            
            # Extract relevant FFT coefficients
            c1, c2 = fourier_transform[p, q], fourier_transform[pm, q]
            c3, c4 = fourier_transform[p, qn], fourier_transform[pm, qn]
            
            # Compute orthonormal coefficients using symmetry relations
            A_cc = (c1 + c2 + c3 + c4).real / 4
            A_cs = (c4 - c2 + c3 - c1).imag / 4
            A_sc = (c4 + c2 - c3 - c1).imag / 4
            A_ss = -(c4 - c2 - c3 + c1).real / 4
            
            # Apply subdomain orthonormal normalization  

            normalization = np.sqrt((2 - (m == 0)) * (2 - (n == 0)))
            area_scaling = length_x * length_y  
            basis_scaling = 1.0 / np.sqrt(length_x * length_y)  
            
            total_scaling = normalization * area_scaling * basis_scaling
            
            C_cc[m, n] = A_cc * total_scaling
            C_cs[m, n] = A_cs * total_scaling  
            C_sc[m, n] = A_sc * total_scaling
            C_ss[m, n] = A_ss * total_scaling
    
    return C_cc, C_cs, C_sc, C_ss


# =============================================================================
# 2D FOURIER COEFFICIENT COMPUTATION
# =============================================================================

def compute_orthonormal_fourier_coefficients(function, domain_bounds, grid_size, max_frequency=None):
    """
    Compute orthonormal real Fourier coefficients using FFT.
    
    This function efficiently computes all four types of 2D Fourier coefficients:
    cos-cos, cos-sin, sin-cos, and sin-sin combinations.
    
    Parameters:
        function (callable): Function f(x,y) to analyze
        domain_bounds (array): Domain boundaries [(x0,x1), (y0,y1)]
        grid_size (int): Number of grid points in each dimension
        max_frequency (int, optional): Maximum frequency to compute
        
    Returns:
        tuple: (C_cc, C_cs, C_sc, C_ss) - coefficient arrays for each combination
    """
    (x0, x1), (y0, y1) = domain_bounds
    length_x, length_y = x1 - x0, y1 - y0
    
    if max_frequency is None:
        max_frequency = grid_size // 2
    
    # Create normalized coordinate grids
    u_coords = np.linspace(0, 1, grid_size, endpoint=False)
    v_coords = np.linspace(0, 1, grid_size, endpoint=False)
    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
    
    # Evaluate function on physical grid
    X_physical = x0 + length_x * U
    Y_physical = y0 + length_y * V
    function_values = function(X_physical, Y_physical)
    
    # Compute 2D FFT
    fourier_transform = np.fft.fft2(function_values) / (grid_size * grid_size)
    
    # Initialize coefficient arrays
    shape = (max_frequency + 1, max_frequency + 1)
    C_cc = np.zeros(shape)  # cos-cos coefficients
    C_cs = np.zeros(shape)  # cos-sin coefficients  
    C_sc = np.zeros(shape)  # sin-cos coefficients
    C_ss = np.zeros(shape)  # sin-sin coefficients
    
    # Extract coefficients for each frequency combination
    for m in range(max_frequency + 1):
        for n in range(max_frequency + 1):
            # Get FFT indices (with proper wrapping)
            p, q = m % grid_size, n % grid_size
            pm, qn = (-m) % grid_size, (-n) % grid_size
            
            # Extract relevant FFT coefficients
            c1, c2 = fourier_transform[p, q], fourier_transform[pm, q]
            c3, c4 = fourier_transform[p, qn], fourier_transform[pm, qn]
            
            # Compute orthonormal coefficients using symmetry relations
            A_cc = (c1 + c2 + c3 + c4).real / 4
            A_cs = (c4 - c2 + c3 - c1).imag / 4
            A_sc = (c4 + c2 - c3 - c1).imag / 4
            A_ss = -(c4 - c2 - c3 + c1).real / 4
            
            # Apply standard orthonormal normalization
            normalization = np.sqrt((2 - (m == 0)) * (2 - (n == 0)))
            area_scaling = length_x * length_y
            
            C_cc[m, n] = A_cc * normalization * area_scaling
            C_cs[m, n] = A_cs * normalization * area_scaling
            C_sc[m, n] = A_sc * normalization * area_scaling
            C_ss[m, n] = A_ss * normalization * area_scaling
    
    return C_cc, C_cs, C_sc, C_ss
