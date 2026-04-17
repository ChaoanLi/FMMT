"""
Data generation module for statistical validation framework.

Provides functions for generating samples from probability distributions and test scenarios.
Supports different noise types, distributions, and modulation patterns for 1D and 2D cases.
"""

import numpy as np
from scipy.stats import truncnorm, uniform
from .functions import get_scenario_functions


# =============================================================================
# PROBABILITY DISTRIBUTION GENERATORS
# =============================================================================




def generate_uniform_samples(n_samples, domain_bounds):
    """
    Generate samples from uniform distribution over specified domain.
    
    Parameters:
        n_samples (int): Number of samples to generate
        domain_bounds (list): Domain boundaries for each dimension
        
    Returns:
        np.ndarray: Generated samples with shape (n_samples, d)
    """
    dimension = len(domain_bounds)
    samples = np.zeros((n_samples, dimension))
    
    for d, (x_min, x_max) in enumerate(domain_bounds):
        samples[:, d] = np.random.uniform(x_min, x_max, n_samples)
    
    return samples


def generate_truncated_normal_samples(n_samples, domain_bounds, mu=0.5, sigma=0.2):
    """
    Generate samples from truncated normal distribution.
    
    Parameters:
        n_samples (int): Number of samples to generate
        domain_bounds (list): Domain boundaries for each dimension
        mu (float): Mean of normal distribution
        sigma (float): Standard deviation of normal distribution
        
    Returns:
        np.ndarray: Generated samples with shape (n_samples, d)
    """
    dimension = len(domain_bounds)
    samples = np.zeros((n_samples, dimension))
    
    for d, (x_min, x_max) in enumerate(domain_bounds):
        # Standardize bounds
        a = (x_min - mu) / sigma
        b = (x_max - mu) / sigma
        
        # Generate truncated normal samples
        samples[:, d] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_samples)
    
    return samples


# =============================================================================
# NOISE GENERATORS
# =============================================================================

def generate_noise(n_samples, sigma, noise_type='gaussian'):
    """
    Generate noise samples from specified distribution.
    
    Parameters:
        n_samples (int): Number of noise samples
        sigma (float): Noise scale parameter
        noise_type (str): Type of noise ('gaussian', 'uniform', 'laplace')
        
    Returns:
        np.ndarray: Generated noise samples
    """
    if noise_type == 'gaussian':
        return np.random.normal(0, sigma, n_samples)
    elif noise_type == 'uniform':
        # Uniform noise with same variance as Gaussian
        # For uniform on [-a, a], variance = a²/3, so a = σ√3
        a = sigma * np.sqrt(3)
        return np.random.uniform(-a, a, n_samples)
    elif noise_type == 'laplace':
        # Laplace noise with same variance as Gaussian
        # For Laplace with scale b, variance = 2b², so b = σ/√2
        b = sigma / np.sqrt(2)
        return np.random.laplace(0, b, n_samples)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")


# =============================================================================
# DENSITY FUNCTION GENERATORS
# =============================================================================

def get_density_function(dist_type, domain_bounds):
    """
    Get density function for specified distribution type.
    
    Parameters:
        dist_type (str): Distribution type ('uniform', 'truncated_normal')
        domain_bounds (list): Domain boundaries
        
    Returns:
        callable: Density function p(x)
    """
    if dist_type == 'uniform':
        # Uniform density
        def uniform_density(x):
            x = np.atleast_2d(x)
            dimension = len(domain_bounds)
            
            # Check if points are in domain
            in_domain = np.ones(x.shape[0], dtype=bool)
            for d, (x_min, x_max) in enumerate(domain_bounds):
                in_domain &= (x[:, d] >= x_min) & (x[:, d] <= x_max)
            
            # Compute uniform density
            volume = np.prod([x_max - x_min for x_min, x_max in domain_bounds])
            density = np.where(in_domain, 1.0 / volume, 0.0)
            return density
        
        return uniform_density
    
    elif dist_type == 'truncated_normal':
        # Truncated normal density
        def truncated_normal_density(x, mu=0.5, sigma=0.2):
            from scipy.stats import truncnorm
            x = np.atleast_2d(x)
            dimension = len(domain_bounds)
            
            density = np.ones(x.shape[0])
            for d, (x_min, x_max) in enumerate(domain_bounds):
                a = (x_min - mu) / sigma
                b = (x_max - mu) / sigma
                density *= truncnorm.pdf(x[:, d], a, b, loc=mu, scale=sigma)
            
            return density
        
        return truncated_normal_density
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


# =============================================================================
# MAIN SAMPLE GENERATION FUNCTION
# =============================================================================

def generate_samples(omega, n_samples, scenario_name, hypothesis='H0', dimension='1D',
                    sigma=0.1, noise_type='gaussian', dist_type='uniform', c=1.0):
    """
    Generate samples for statistical validation testing.
    
    Parameters:
        omega (array): Domain bounds for 1D or 2D
        n_samples (int): Number of samples to generate
        scenario_name (str): Scenario identifier
        hypothesis (str): 'H0' or 'H1'
        dimension (str): '1D' or '2D' 
        sigma (float): Noise standard deviation
        noise_type (str): Noise distribution type
        dist_type (str): Input distribution type
        c (float): Modulation strength for H1 scenarios
        
    Returns:
        tuple: (X_sample, y_sample)
    """
    # Convert omega to standard format
    if isinstance(omega, list):
        domain_bounds = omega
    else:
        domain_bounds = omega.tolist()
    
    # Generate input samples according to specified distribution
    if dist_type == 'uniform':
        X_sample = generate_uniform_samples(n_samples, domain_bounds)
    elif dist_type == 'truncated_normal':
        X_sample = generate_truncated_normal_samples(n_samples, domain_bounds)
    else:
        raise ValueError(f"Unsupported combination: dist_type='{dist_type}', dimension='{dimension}'")
    
    # Get scenario functions
    scenario = get_scenario_functions(scenario_name, hypothesis, dimension, c)
    
    # Select function based on hypothesis
    if hypothesis == 'H0':
        func = scenario['f1']  # Unmodulated function
    else:  # H1
        func = scenario['f2']  # Modulated function
    
    # Evaluate function at sample points
    if dimension == '1D':
        y_clean = func(X_sample).flatten()
    else:  # 2D
        y_clean = func(X_sample).flatten()
    
    # Add noise
    noise = generate_noise(n_samples, sigma, noise_type)
    y_sample = y_clean + noise
    
    return X_sample, y_sample


# =============================================================================
# HELPER FUNCTIONS FOR SCENARIO VALIDATION
# =============================================================================

def validate_scenario_configuration(scenario_name, hypothesis, dimension, c):
    """
    Validate scenario configuration parameters.
    
    Parameters:
        scenario_name (str): Scenario identifier
        hypothesis (str): Hypothesis type
        dimension (str): Problem dimension
        c (float): Modulation strength
        
    Returns:
        bool: True if valid
    """
    # Validate hypothesis
    if hypothesis not in ['H0', 'H1']:
        raise ValueError(f"hypothesis must be 'H0' or 'H1', got '{hypothesis}'")
    
    # Validate dimension
    if dimension not in ['1D', '2D']:
        raise ValueError(f"dimension must be '1D' or '2D', got '{dimension}'")
    
    # Validate modulation strength
    if not isinstance(c, (int, float)) or c < 0:
        raise ValueError(f"c must be non-negative number, got {c}")
    
    # Check if scenario exists
    try:
        scenario = get_scenario_functions(scenario_name, hypothesis, dimension, c)
        if not all(key in scenario for key in ['f1', 'f2', 'description']):
            raise ValueError(f"Invalid scenario structure for '{scenario_name}'")
    except Exception as e:
        raise ValueError(f"Failed to load scenario '{scenario_name}': {e}")
    
    return True


def compute_scenario_statistics(X_sample, y_sample, scenario_name, hypothesis, dimension, c):
    """
    Compute statistical properties of generated samples for validation.
    
    Parameters:
        X_sample (array): Input samples
        y_sample (array): Output samples
        scenario_name (str): Scenario identifier
        hypothesis (str): Hypothesis type
        dimension (str): Problem dimension
        c (float): Modulation strength
        
    Returns:
        dict: Statistical summary of the generated data
    """
    stats = {
        'scenario_name': scenario_name,
        'hypothesis': hypothesis,
        'dimension': dimension,
        'modulation_strength': c,
        'n_samples': len(X_sample),
        'input_statistics': {},
        'output_statistics': {}
    }
    
    # Input statistics
    for d in range(X_sample.shape[1]):
        stats['input_statistics'][f'x{d+1}'] = {
            'mean': float(np.mean(X_sample[:, d])),
            'std': float(np.std(X_sample[:, d], ddof=1)),
            'min': float(np.min(X_sample[:, d])),
            'max': float(np.max(X_sample[:, d]))
        }
    
    # Output statistics
    stats['output_statistics'] = {
        'mean': float(np.mean(y_sample)),
        'std': float(np.std(y_sample, ddof=1)),
        'min': float(np.min(y_sample)),
        'max': float(np.max(y_sample))
    }
    
    # Compute signal-to-noise ratio estimate
    # Assume noise level can be estimated from residuals of a simple fit
    if dimension == '1D':
        # Simple polynomial fit to estimate noise level
        poly_coeffs = np.polyfit(X_sample.flatten(), y_sample, deg=3)
        y_fit = np.polyval(poly_coeffs, X_sample.flatten())
        residuals = y_sample - y_fit
        noise_std = np.std(residuals, ddof=1)
        signal_std = np.std(y_fit, ddof=1)
    else:  # 2D case
        # Use mean and variance as rough signal measure
        signal_std = np.std(y_sample, ddof=1)
        noise_std = signal_std * 0.1  # Rough estimate
    
    if noise_std > 0:
        snr = signal_std / noise_std
    else:
        snr = np.inf
    
    stats['signal_to_noise_ratio'] = float(snr)
    stats['estimated_noise_std'] = float(noise_std)
    stats['estimated_signal_std'] = float(signal_std)
    
    return stats


def generate_validation_dataset(scenario_configs, n_samples_per_config=100):
    """
    Generate comprehensive validation dataset across multiple scenarios.
    
    Parameters:
        scenario_configs (list): List of scenario configuration dictionaries
        n_samples_per_config (int): Number of samples per configuration
        
    Returns:
        dict: Comprehensive validation dataset
    """
    validation_data = {
        'configurations': scenario_configs,
        'datasets': [],
        'metadata': {
            'n_configurations': len(scenario_configs),
            'n_samples_per_config': n_samples_per_config,
            'total_samples': len(scenario_configs) * n_samples_per_config
        }
    }
    
    for i, config in enumerate(scenario_configs):
        # Validate configuration
        validate_scenario_configuration(
            config['scenario_name'], 
            config['hypothesis'], 
            config['dimension'], 
            config.get('c', 1.0)
        )
        
        # Generate samples
        X_sample, y_sample = generate_samples(
            omega=config['omega'],
            n_samples=n_samples_per_config,
            scenario_name=config['scenario_name'],
            hypothesis=config['hypothesis'],
            dimension=config['dimension'],
            sigma=config.get('sigma', 0.1),
            noise_type=config.get('noise_type', 'gaussian'),
            dist_type=config.get('dist_type', 'uniform'),
            c=config.get('c', 1.0)
        )
        
        # Compute statistics
        stats = compute_scenario_statistics(
            X_sample, y_sample,
            config['scenario_name'],
            config['hypothesis'],
            config['dimension'],
            config.get('c', 1.0)
        )
        
        dataset_entry = {
            'config_index': i,
            'config': config,
            'X_sample': X_sample,
            'y_sample': y_sample,
            'statistics': stats
        }
        
        validation_data['datasets'].append(dataset_entry)
    
    return validation_data


def sample_noise(n_samples, sigma, noise_type):
    """
    Generate different types of noise.
    
    Parameters:
        n_samples (int): Number of samples
        sigma (float): Noise standard deviation
        noise_type (str): Type of noise
            - "gaussian": Gaussian distribution N(0, σ²)
            - "mixture_gaussian": Mixture of Gaussians
            - "uniform": Uniform distribution U(-√3σ, √3σ)
    
    Returns:
        array: Generated noise samples
    """
    if noise_type == "gaussian":
        # Gaussian distribution N(0, sigma^2)
        noise = np.random.normal(loc=0, scale=sigma, size=n_samples)

    elif noise_type == "mixture_gaussian":
        # Bimodal mixture: 50% probability for each peak at ±σ/2
        choices = np.random.choice([-1, 1], size=n_samples, p=[0.5, 0.5])
        noise = np.random.normal(loc=choices*sigma/2, scale=np.sqrt(3/4) * sigma, size=n_samples)

    elif noise_type == "uniform":
        # Uniform distribution U(-√3σ, √3σ)
        a = np.sqrt(3) * sigma
        noise = np.random.uniform(low=-a, high=a, size=n_samples)

    else:
        raise ValueError("Invalid noise_type. Choose from ['gaussian', 'mixture_gaussian', 'uniform'].")

    return noise


def generate_pdf_function(omega, dist_type, mu=0.5, sigma=0.5):
    """
    Generate probability density function for different distributions on specified domain.
    
    Parameters:
        omega (array): Domain boundaries, shape (d, 2) where d is dimension
        dist_type (str): Distribution type
            - "uniform": Uniform distribution on omega
            - "truncated_normal": Truncated normal distribution on omega
        mu (float): Mean parameter for truncated_normal
        sigma (float): Standard deviation parameter for truncated_normal
        
    Returns:
        function: PDF function that takes points with shape (..., d) and returns density values
    """
    omega = np.array(omega)
    d = omega.shape[0]
    
    if dist_type == "uniform":
        # Uniform distribution: constant density over the domain
        volume = np.prod(omega[:, 1] - omega[:, 0])  # Volume of the domain
        density_value = 1.0 / volume  # Normalized to integrate to 1
        
        def pdf(x):
            x = np.array(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Check if points are within domain
            in_domain = np.all((x >= omega[:, 0]) & (x <= omega[:, 1]), axis=1)
            result = np.where(in_domain, density_value, 0.0)
            return result
            
    elif dist_type == "truncated_normal":
        if d == 1:
            # 1D truncated normal
            domain_min, domain_max = omega[0, 0], omega[0, 1]
            a = (domain_min - mu) / sigma
            b = (domain_max - mu) / sigma
            
            def pdf(x):
                x = np.array(x).flatten()
                return truncnorm.pdf(x, a, b, loc=mu, scale=sigma)
                
        else:
            # Multidimensional case: independent truncated normals
            # f(x1, x2, ...) = f1(x1) * f2(x2) * ...
            truncnorm_dists = []
            for i in range(d):
                domain_min, domain_max = omega[i, 0], omega[i, 1]
                a = (domain_min - mu) / sigma
                b = (domain_max - mu) / sigma
                truncnorm_dists.append(truncnorm(a, b, loc=mu, scale=sigma))
            
            def pdf(x):
                x = np.array(x)
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                
                # Compute joint density as product of marginal densities
                joint_density = np.ones(x.shape[0])
                for i in range(d):
                    marginal_density = truncnorm_dists[i].pdf(x[:, i])
                    joint_density *= marginal_density
                
                return joint_density
    
    else:
        raise ValueError("Invalid dist_type. Choose from ['uniform', 'truncated_normal'].")
    
    return pdf


def generate_samples_from_distribution(omega, n_samples, dist_type="uniform", mu=0.5, sigma=0.25):
    """
    Generate samples from specified distribution on given domain.
    
    Parameters:
        omega (array): Domain boundaries, shape (d, 2) where d is dimension
        n_samples (int): Number of samples to generate
        dist_type (str): Distribution type ('uniform', 'truncated_normal')
        mu (float): Mean parameter for truncated_normal
        sigma (float): Standard deviation parameter for truncated_normal
        
    Returns:
        array: Generated samples with shape (n_samples, d)
    """
    d = omega.shape[0]
    X_sample = np.zeros((n_samples, d))
    
    if dist_type == "uniform":
        X_sample = np.random.uniform(low=omega[:, 0], high=omega[:, 1], size=(n_samples, d))
    
    elif dist_type == "truncated_normal":
        for i in range(d):
            # Truncated normal for each dimension
            domain_min, domain_max = omega[i, 0], omega[i, 1]
            a = (domain_min - mu) / sigma
            b = (domain_max - mu) / sigma
            dist = truncnorm(a, b, loc=mu, scale=sigma)
            X_sample[:, i] = dist.rvs(size=n_samples)
    
    else:
        raise ValueError("Invalid dist_type. Choose from ['uniform', 'truncated_normal'].")
    
    return X_sample


def generate_samples_comparison(omega, n_samples, scenario_name, dimension='1D',
                              sigma=0.5, noise_type="gaussian", dist_type="truncated_normal", c=1.0):
    """
    Generate comparison samples for H0 vs H1 testing.
    
    Returns separate datasets for each hypothesis.

    Parameters:
        omega (array): Domain boundaries
        n_samples (int): Number of samples per hypothesis
        scenario_name (str): Scenario identifier
        dimension (str): '1D' or '2D'
        sigma (float): Noise level
        noise_type (str): Type of noise
        dist_type (str): Sample distribution ('uniform', 'truncated_normal')
        c (float): Signal strength

    Returns:
        dict: {'H0': (X_h0, y_h0), 'H1': (X_h1, y_h1)}
    """
    # Generate H0 samples
    X_h0, y_h0 = generate_samples(omega, n_samples, scenario_name, 'H0', dimension,
                                 sigma, noise_type, dist_type, c)
    
    # Generate H1 samples  
    X_h1, y_h1 = generate_samples(omega, n_samples, scenario_name, 'H1', dimension,
                                 sigma, noise_type, dist_type, c)
    
    return {
        'H0': (X_h0, y_h0),
        'H1': (X_h1, y_h1)
    }


 