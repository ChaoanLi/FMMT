"""
Utility functions for statistical validation framework.

Provides functions for reproducibility, data persistence, and result formatting.
"""

import numpy as np
import json
import pickle
from pathlib import Path
import time
from datetime import datetime
import warnings


# =============================================================================
# REPRODUCIBILITY AND RANDOM SEED MANAGEMENT
# =============================================================================

def set_random_seed(seed=42):
    """
    Set random seed for reproducible results.
    
    Parameters:
        seed (int): Random seed value
    """
    np.random.seed(seed)


def create_experiment_id(prefix="exp", include_timestamp=True):
    """
    Generate unique experiment identifier for result tracking.
    
    Parameters:
        prefix (str): Prefix for experiment ID
        include_timestamp (bool): Whether to include timestamp
        
    Returns:
        str: Unique experiment identifier
    """
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    else:
        return f"{prefix}_{int(time.time())}"


# =============================================================================
# DATA PERSISTENCE AND SERIALIZATION
# =============================================================================

def save_experimental_results(results, filename, format='json', metadata=None):
    """
    Save experimental results with optional metadata to file.
    
    Parameters:
        results (dict): Experimental results dictionary
        filename (str or Path): Output filename
        format (str): Serialization format ('json' or 'pickle')
        metadata (dict): Optional experimental metadata
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'results': results,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'format_version': '1.0'
    }
    
    if format == 'json':
        # Convert to JSON-serializable format
        json_data = convert_to_serializable(save_data)
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")


def load_experimental_results(filename, format='json'):
    """
    Load experimental results from file.
    
    Parameters:
        filename (str or Path): Input filename
        format (str): Serialization format ('json' or 'pickle')
        
    Returns:
        dict: Loaded experimental results with metadata
    """
    filepath = Path(filename)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    if format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
    
    return data


def convert_to_serializable(obj):
    """
    Convert complex objects to JSON-serializable format.
    
    This function handles numpy arrays, functions, and other non-serializable
    objects commonly found in statistical validation results.
    
    Parameters:
        obj: Object to convert
        
    Returns:
        Serializable representation of the object
    """
    import types
    
    if isinstance(obj, np.ndarray):
        return {
            '_type': 'numpy_array',
            'data': obj.tolist(),
            'shape': obj.shape,
            'dtype': str(obj.dtype)
        }
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (types.FunctionType, types.LambdaType)):
        return {'_type': 'function', 'name': getattr(obj, '__name__', 'anonymous')}
    elif callable(obj):
        return {'_type': 'callable', 'class': type(obj).__name__}
    elif hasattr(obj, '__dict__'):
        # Handle complex objects (like sklearn models)
        return {'_type': 'object', 'class': type(obj).__name__, 'repr': str(obj)}
    else:
        try:
            # Test if object is JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return {'_type': 'non_serializable', 'repr': str(obj)}


# =============================================================================
# NUMERICAL COMPUTATION UTILITIES
# =============================================================================

def compute_descriptive_statistics(values, name="values"):
    """
    Compute comprehensive descriptive statistics for numerical data.
    
    Parameters:
        values (array-like): Numerical values
        name (str): Variable name for reporting
        
    Returns:
        dict: Comprehensive statistical summary
    """
    values = np.asarray(values)
    
    if values.size == 0:
        return {'name': name, 'count': 0, 'note': 'Empty array'}
    
    # Remove NaN values for computation
    clean_values = values[~np.isnan(values)]
    n_nan = values.size - clean_values.size
    
    if clean_values.size == 0:
        return {'name': name, 'count': values.size, 'nan_count': n_nan, 'note': 'All NaN values'}
    
    stats = {
        'name': name,
        'count': len(clean_values),
        'nan_count': n_nan,
        'mean': float(np.mean(clean_values)),
        'std': float(np.std(clean_values, ddof=1)) if len(clean_values) > 1 else 0.0,
        'min': float(np.min(clean_values)),
        'max': float(np.max(clean_values)),
        'median': float(np.median(clean_values)),
        'q25': float(np.percentile(clean_values, 25)),
        'q75': float(np.percentile(clean_values, 75)),
        'skewness': float(compute_skewness(clean_values)),
        'kurtosis': float(compute_kurtosis(clean_values))
    }
    
    return stats


def compute_skewness(values):
    """
    Compute sample skewness using the adjusted Fisher-Pearson standardized moment coefficient.
    
    Parameters:
        values (array): Numerical values
        
    Returns:
        float: Skewness value
    """
    values = np.asarray(values)
    n = len(values)
    
    if n < 3:
        return np.nan
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=0)
    
    if std_val == 0:
        return np.nan
    
    m3 = np.mean(((values - mean_val) / std_val) ** 3)
    
    # Adjust for sample bias
    skew = m3 * (n * (n - 1)) ** 0.5 / (n - 2)
    return skew


def compute_kurtosis(values):
    """
    Compute sample excess kurtosis using Fisher's definition (normal distribution has kurtosis 0).
    
    Parameters:
        values (array): Numerical values
        
    Returns:
        float: Excess kurtosis value
    """
    values = np.asarray(values)
    n = len(values)
    
    if n < 4:
        return np.nan
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=0)
    
    if std_val == 0:
        return np.nan
    
    m4 = np.mean(((values - mean_val) / std_val) ** 4)
    
    # Adjust for sample bias and subtract 3 for excess kurtosis
    kurt = ((n + 1) * (n - 1) * m4 - 3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return kurt


def compute_confidence_interval(values, confidence_level=0.95):
    """
    Compute confidence interval for sample mean assuming normal distribution.
    
    Parameters:
        values (array): Numerical values
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error)
    """
    from scipy import stats
    
    values = np.asarray(values)
    clean_values = values[~np.isnan(values)]
    
    if len(clean_values) < 2:
        return (np.nan, np.nan, np.nan)
    
    mean_val = np.mean(clean_values)
    sem = stats.sem(clean_values)  # Standard error of mean
    alpha = 1 - confidence_level
    
    # Use t-distribution for small samples
    dof = len(clean_values) - 1
    t_critical = stats.t.ppf(1 - alpha/2, dof)
    
    margin_of_error = t_critical * sem
    lower_bound = mean_val - margin_of_error
    upper_bound = mean_val + margin_of_error
    
    return (lower_bound, upper_bound, margin_of_error)


# =============================================================================
# DOMAIN AND GEOMETRIC UTILITIES
# =============================================================================

def create_domain_bounds(bounds_list):
    """
    Create standardized domain bounds array from various input formats.
    
    Parameters:
        bounds_list (list): List of [min, max] pairs for each dimension
        
    Returns:
        np.ndarray: Domain bounds array with shape (d, 2)
    """
    bounds_array = np.array(bounds_list)
    
    if bounds_array.ndim == 1 and len(bounds_array) == 2:
        # Single dimension case: [min, max] -> [[min, max]]
        bounds_array = bounds_array.reshape(1, 2)
    
    validate_domain_bounds(bounds_array)
    return bounds_array


def validate_domain_bounds(omega):
    """
    Validate domain bounds array for correctness and consistency.
    
    Parameters:
        omega (np.ndarray): Domain bounds array
        
    Returns:
        bool: True if valid (raises exception if invalid)
        
    Raises:
        ValueError: If domain bounds are invalid
    """
    if not isinstance(omega, np.ndarray):
        raise ValueError("Domain bounds must be a numpy array")
    
    if omega.ndim != 2 or omega.shape[1] != 2:
        raise ValueError(f"Domain bounds must have shape (d, 2), got {omega.shape}")
    
    if np.any(omega[:, 0] >= omega[:, 1]):
        invalid_dims = np.where(omega[:, 0] >= omega[:, 1])[0]
        raise ValueError(f"Lower bounds must be less than upper bounds. Invalid dimensions: {invalid_dims}")
    
    if np.any(~np.isfinite(omega)):
        raise ValueError("Domain bounds must be finite")
    
    return True


def compute_domain_volume(omega):
    """
    Compute the volume (or length in 1D) of the domain.
    
    Parameters:
        omega (np.ndarray): Domain bounds array
        
    Returns:
        float: Domain volume
    """
    validate_domain_bounds(omega)
    return np.prod(omega[:, 1] - omega[:, 0])


def is_point_in_domain(points, omega):
    """
    Check if points are within the specified domain bounds.
    
    Parameters:
        points (np.ndarray): Points to check (n_points, d)
        omega (np.ndarray): Domain bounds (d, 2)
        
    Returns:
        np.ndarray: Boolean array indicating which points are in domain
    """
    points = np.atleast_2d(points)
    validate_domain_bounds(omega)
    
    if points.shape[1] != omega.shape[0]:
        raise ValueError(f"Dimension mismatch: points have {points.shape[1]} dims, domain has {omega.shape[0]} dims")
    
    # Check if all coordinates are within bounds
    in_bounds = np.all((points >= omega[:, 0]) & (points <= omega[:, 1]), axis=1)
    return in_bounds


# =============================================================================
# STATISTICAL TESTING UTILITIES
# =============================================================================

def compute_rejection_rate(p_values, alpha=0.05):
    """
    Compute rejection rate for statistical tests.
    
    Parameters:
        p_values (array): Array of p-values
        alpha (float): Significance level
        
    Returns:
        dict: Rejection rate statistics
    """
    p_values = np.asarray(p_values)
    
    if len(p_values) == 0:
        return {'rejection_rate': np.nan, 'n_tests': 0, 'n_rejections': 0}
    
    rejections = p_values < alpha
    n_rejections = np.sum(rejections)
    rejection_rate = n_rejections / len(p_values)
    
    return {
        'rejection_rate': rejection_rate,
        'n_tests': len(p_values),
        'n_rejections': int(n_rejections),
        'alpha': alpha
    }


def compute_multiple_testing_correction(p_values, method='bonferroni'):
    """
    Apply multiple testing correction to p-values.
    
    Parameters:
        p_values (array): Raw p-values
        method (str): Correction method ('bonferroni', 'holm', 'fdr_bh')
        
    Returns:
        dict: Corrected p-values and adjusted alpha levels
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    if m == 0:
        return {'corrected_pvals': [], 'method': method, 'n_tests': 0}
    
    if method == 'bonferroni':
        corrected = np.minimum(p_values * m, 1.0)
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(p_values[idx] * (m - i), 1.0)
            if i > 0:
                corrected[idx] = max(corrected[idx], corrected[sorted_idx[i-1]])
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR control
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros_like(p_values)
        for i in range(m-1, -1, -1):
            idx = sorted_idx[i]
            corrected[idx] = min(p_values[idx] * m / (i + 1), 1.0)
            if i < m - 1:
                corrected[idx] = min(corrected[idx], corrected[sorted_idx[i+1]])
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return {
        'corrected_pvals': corrected,
        'original_pvals': p_values,
        'method': method,
        'n_tests': m
    }


# =============================================================================
# RESULT FORMATTING AND REPORTING
# =============================================================================

def format_validation_summary(results, title="Validation Results", precision=4):
    """
    Format experimental validation results for human-readable display.
    
    Parameters:
        results (dict): Validation results dictionary
        title (str): Summary title
        precision (int): Decimal precision for numerical values
        
    Returns:
        str: Formatted summary string
    """
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"{title:^60}")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    for section, data in results.items():
        summary_lines.append(f"{section.upper().replace('_', ' ')}")
        summary_lines.append("-" * 40)
        
        if isinstance(data, dict):
            for key, value in data.items():
                formatted_value = format_value(value, precision)
                summary_lines.append(f"  {key:30s}: {formatted_value}")
        elif isinstance(data, (list, np.ndarray)):
            if len(data) <= 10:
                formatted_values = [format_value(v, precision) for v in data]
                summary_lines.append(f"  {section:30s}: {formatted_values}")
            else:
                summary_lines.append(f"  {section:30s}: array of length {len(data)}")
                if hasattr(data, 'dtype'):
                    summary_lines.append(f"  {'dtype':30s}: {data.dtype}")
        else:
            formatted_value = format_value(data, precision)
            summary_lines.append(f"  {section:30s}: {formatted_value}")
        
        summary_lines.append("")
    
    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)


def format_value(value, precision=4):
    """
    Format a single value for display with appropriate precision.
    
    Parameters:
        value: Value to format
        precision (int): Decimal precision
        
    Returns:
        str: Formatted value string
    """
    if isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        elif np.isinf(value):
            return "∞" if value > 0 else "-∞"
        elif abs(value) < 10**(-precision):
            return f"{value:.2e}"
        else:
            return f"{value:.{precision}f}"
    elif isinstance(value, np.ndarray):
        if value.size <= 5:
            formatted_items = [format_value(item, precision) for item in value.flatten()]
            return f"[{', '.join(formatted_items)}]"
        else:
            return f"array(shape={value.shape}, dtype={value.dtype})"
    elif isinstance(value, bool):
        return "✓" if value else "✗"
    else:
        return str(value)


def create_performance_report(timing_results, memory_results=None):
    """
    Create performance analysis report from timing and memory measurements.
    
    Parameters:
        timing_results (dict): Timing measurements
        memory_results (dict): Optional memory usage measurements
        
    Returns:
        str: Formatted performance report
    """
    lines = []
    lines.append("PERFORMANCE ANALYSIS REPORT")
    lines.append("=" * 50)
    
    # Timing analysis
    lines.append("\nTIMING RESULTS:")
    lines.append("-" * 30)
    
    for operation, times in timing_results.items():
        if isinstance(times, (list, np.ndarray)):
            stats = compute_descriptive_statistics(times, operation)
            lines.append(f"{operation:25s}: {stats['mean']:.4f}s ± {stats['std']:.4f}s (n={stats['count']})")
        else:
            lines.append(f"{operation:25s}: {times:.4f}s")
    
    # Memory analysis
    if memory_results:
        lines.append("\nMEMORY USAGE:")
        lines.append("-" * 30)
        
        for component, memory in memory_results.items():
            if isinstance(memory, (int, float)):
                lines.append(f"{component:25s}: {memory:.2f} MB")
            else:
                lines.append(f"{component:25s}: {memory}")
    
    return "\n".join(lines)


# =============================================================================
# VALIDATION AND ERROR CHECKING
# =============================================================================

def validate_statistical_inputs(n_samples, sigma, alpha=0.05):
    """
    Validate common statistical test inputs for consistency and validity.
    
    Parameters:
        n_samples (int): Number of samples
        sigma (float): Noise standard deviation
        alpha (float): Significance level
        
    Returns:
        bool: True if valid (raises exception if invalid)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
    
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    # Issue warnings for potentially problematic values
    if n_samples < 30:
        warnings.warn(f"Small sample size (n={n_samples}) may affect test reliability", UserWarning)
    
    if alpha > 0.1:
        warnings.warn(f"Large significance level (α={alpha}) increases Type I error risk", UserWarning)
    
    return True


def check_array_compatibility(arrays, names=None):
    """
    Check if multiple arrays have compatible shapes for operations.
    
    Parameters:
        arrays (list): List of arrays to check
        names (list): Optional names for arrays (for error messages)
        
    Returns:
        bool: True if compatible
        
    Raises:
        ValueError: If arrays are incompatible
    """
    if not arrays:
        return True
    
    names = names or [f"array_{i}" for i in range(len(arrays))]
    
    if len(arrays) != len(names):
        raise ValueError("Number of array names must match number of arrays")
    
    first_shape = arrays[0].shape
    
    for i, (arr, name) in enumerate(zip(arrays[1:], names[1:]), 1):
        if arr.shape != first_shape:
            raise ValueError(f"Shape mismatch: {names[0]} has shape {first_shape}, "
                           f"{name} has shape {arr.shape}")
    
    return True 