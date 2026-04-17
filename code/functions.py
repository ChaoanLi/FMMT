"""
Function definitions for test cases and signal generation.

This module contains various test functions used in the subdomain validation framework.
"""

import numpy as np


def get_scenario_functions(scenario_name, hypothesis='H0', dimension='1D', c=1.0):
    """
    Get functions for specific comparison scenarios with H0/H1 hypotheses.
    
    Parameters:
        scenario_name (str): Scenario identifier (see get_available_scenarios())
        hypothesis (str): 'H0' (null hypothesis) or 'H1' (alternative hypothesis)
        dimension (str): '1D' or '2D' for function dimension
        c (float): Modulation strength for H1 scenarios (default: 1.0)
        
    Returns:
        dict: {'f1': function, 'f2': function, 'description': str}
    """
    if dimension == '1D':
        return _get_1d_scenarios(scenario_name, hypothesis, c)
    elif dimension == '2D':
        return _get_2d_scenarios(scenario_name, hypothesis, c)
    else:
        raise ValueError("dimension must be '1D' or '2D'")


def _get_1d_scenarios(scenario_name, hypothesis, c):
    """Get 1D scenario functions."""
    scenarios = {
        # Basic comparison scenarios (global)
        'constant_vs_linear': {
            'H0': {
                'f1': lambda x: np.ones_like(x), 
                'f2': lambda x: np.ones_like(x),
                'description': 'H0: Both functions are constant'
            },
            'H1': {
                'f1': lambda x: np.ones_like(x), 
                'f2': lambda x: 1 + c * x,
                'description': f'H1: constant vs constant + {c}*x'
            }
        },
        'exp_vs_exp_linear': {
            'H0': {
                'f1': lambda x: np.exp(x), 
                'f2': lambda x: np.exp(x),
                'description': 'H0: Both functions are exponential'
            },
            'H1': {
                'f1': lambda x: np.exp(x), 
                'f2': lambda x: np.exp(x) + c * x,
                'description': f'H1: exp(x) vs exp(x) + {c}*x'
            }
        },
        'sine_vs_sine_linear': {
            'H0': {
                'f1': lambda x: np.sin(2 * np.pi * x), 
                'f2': lambda x: np.sin(2 * np.pi * x),
                'description': 'H0: Both functions are sine'
            },
            'H1': {
                'f1': lambda x: np.sin(2 * np.pi * x), 
                'f2': lambda x: np.sin(2 * np.pi * x) + c * x,
                'description': f'H1: sin(2πx) vs sin(2πx) + {c}*x'
            }
        },
        'constant_vs_sine': {
            'H0': {
                'f1': lambda x: np.ones_like(x), 
                'f2': lambda x: np.ones_like(x),
                'description': 'H0: Both functions are constant'
            },
            'H1': {
                'f1': lambda x: np.ones_like(x), 
                'f2': lambda x: 1 + c * np.sin(2 * np.pi * x),
                'description': f'H1: constant vs constant + {c}*sin(2πx)'
            }
        },
        'exp_vs_exp_sine': {
            'H0': {
                'f1': lambda x: np.exp(x), 
                'f2': lambda x: np.exp(x),
                'description': 'H0: Both functions are exponential'
            },
            'H1': {
                'f1': lambda x: np.exp(x), 
                'f2': lambda x: np.exp(x) + c * np.sin(2 * np.pi * x),
                'description': f'H1: exp(x) vs exp(x) + {c}*sin(2πx)'
            }
        },
        'sine_vs_double_sine': {
            'H0': {
                'f1': lambda x: np.sin(2 * np.pi * x), 
                'f2': lambda x: np.sin(2 * np.pi * x),
                'description': 'H0: Both functions are sine'
            },
            'H1': {
                'f1': lambda x: np.sin(2 * np.pi * x), 
                'f2': lambda x: (1 + c) * np.sin(2 * np.pi * x),
                'description': f'H1: sin(2πx) vs {1+c}*sin(2πx)'
            }
        },
        
        # Subdomain-specific scenarios
        'exp_vs_sub1_sine': {
            'H0': {
                'f1': lambda x: np.exp(x),
                'f2': lambda x: np.exp(x),
                'description': 'H0: Both functions are exp(x)'
            },
            'H1': {
                'f1': lambda x: np.exp(x),
                'f2': lambda x: np.exp(x) + c * np.sin(6 * np.pi * x) * ((x >= 0) & (x < 1/3)).astype(int),
                'description': f'H1: exp(x) vs exp(x) + {c}*sin(6πx) on subdomain [0,1/3)'
            }
        },
        'exp_vs_sub1_cosine': {
            'H0': {
                'f1': lambda x: np.exp(x),
                'f2': lambda x: np.exp(x),
                'description': 'H0: Both functions are exp(x)'
            },
            'H1': {
                'f1': lambda x: np.exp(x),
                'f2': lambda x: np.exp(x) + c * np.cos(12 * np.pi * x) * ((x >= 0) & (x < 1/3)).astype(int),
                'description': f'H1: exp(x) vs exp(x) + {c}*cos(12πx) on subdomain [0,1/3)'
            }
        },
        'exp_vs_sub12_sine': {
            'H0': {
                'f1': lambda x: np.exp(x),
                'f2': lambda x: np.exp(x),
                'description': 'H0: Both functions are exp(x)'
            },
            'H1': {
                'f1': lambda x: np.exp(x),
                'f2': lambda x: np.exp(x) + c * np.sin(6 * np.pi * x) * (((x >= 0) & (x < 1/3)) | ((x >= 1/3) & (x < 2/3))).astype(int),
                'description': f'H1: exp(x) vs exp(x) + {c}*sin(6πx) on subdomains [0,2/3)'
            }
        }
    }
    
    if scenario_name not in scenarios:
        available_scenarios = list(scenarios.keys())
        raise ValueError(f"Invalid scenario_name. Choose from {available_scenarios}")
    
    if hypothesis not in ['H0', 'H1']:
        raise ValueError("hypothesis must be 'H0' or 'H1'")
    
    return scenarios[scenario_name][hypothesis]


def _get_2d_scenarios(scenario_name, hypothesis, c):
    """Get 2D scenario functions."""
    scenarios = {
        # 2D Subdomain scenarios (4 quadrants)
        'exp_2d_vs_quad1_modulation': {
            'H0': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_base(pts),
                'description': 'H0: Both functions are polynomial-trigonometric combination'
            },
            'H1': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_subdomain_modulation(pts, quad1=True, quad2=False, quad3=False, quad4=False, c=c),
                'description': f'H1: base_func vs base_func + {c}*sin(2πx1)*sin(2πx2) on quadrant1'
            }
        },
        'exp_2d_vs_quad2_modulation': {
            'H0': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_base(pts),
                'description': 'H0: Both functions are polynomial-trigonometric combination'
            },
            'H1': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_subdomain_modulation(pts, quad1=False, quad2=True, quad3=False, quad4=False, c=c),
                'description': f'H1: base_func vs base_func + {c}*sin(2πx1)*sin(2πx2) on quadrant2'
            }
        },
        'exp_2d_vs_quad3_modulation': {
            'H0': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_base(pts),
                'description': 'H0: Both functions are polynomial-trigonometric combination'
            },
            'H1': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_subdomain_modulation(pts, quad1=False, quad2=False, quad3=True, quad4=False, c=c),
                'description': f'H1: base_func vs base_func + {c}*sin(2πx1)*sin(2πx2) on quadrant3'
            }
        },
        'exp_2d_vs_quad4_modulation': {
            'H0': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_base(pts),
                'description': 'H0: Both functions are polynomial-trigonometric combination'
            },
            'H1': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_subdomain_modulation(pts, quad1=False, quad2=False, quad3=False, quad4=True, c=c),
                'description': f'H1: base_func vs base_func + {c}*sin(2πx1)*sin(2πx2) on quadrant4'
            }
        },
        'exp_2d_vs_multi_quad_modulation': {
            'H0': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_base(pts),
                'description': 'H0: Both functions are polynomial-trigonometric combination'
            },
            'H1': {
                'f1': lambda pts: exp_2d_base(pts),
                'f2': lambda pts: exp_2d_subdomain_modulation(pts, quad1=True, quad2=True, quad3=False, quad4=False, c=c),
                'description': f'H1: base_func vs base_func + {c}*sin(2πx1)*sin(2πx2) on quadrant1,2'
            }
        },
        'exp_2d_vs_all_quad_modulation': {
        'H0': {
            'f1': lambda pts: exp_2d_base(pts),
            'f2': lambda pts: exp_2d_base(pts),
            'description': 'H0: Both functions are polynomial-trigonometric combination'
            },
        'H1': {
            'f1': lambda pts: exp_2d_base(pts),
            'f2': lambda pts: exp_2d_subdomain_modulation(
                pts,
                quad1=True,
                quad2=True,
                quad3=True,
                quad4=True,
                c=c
            ),
            'description': f'H1: base_func vs base_func + {c}*sin(2πx1)*sin(2πx2) on all quadrants'
            }
        },
        'exp_2d_vs_all_quad_constant': {
        'H0': {
            'f1': lambda pts: exp_2d_base(pts),
            'f2': lambda pts: exp_2d_base(pts),
            'description': 'H0: Both functions are polynomial-trigonometric combination'
            },
        'H1': {
            'f1': lambda pts: exp_2d_base(pts),
            'f2': lambda pts: exp_2d_subdomain_constant_modulation(
                pts,
                quad1=True,
                quad2=True,
                quad3=True,
                quad4=True,
                c=c
            ),
            'description': f'H1: base_func vs base_func + {c} on all quadrants'
            }
        }
    }
    
    if scenario_name not in scenarios:
        available_scenarios = list(scenarios.keys())
        raise ValueError(f"Invalid scenario_name. Choose from {available_scenarios}")
    
    if hypothesis not in ['H0', 'H1']:
        raise ValueError("hypothesis must be 'H0' or 'H1'")
    
    return scenarios[scenario_name][hypothesis]


# Helper functions for 2D subdomain scenarios
def exp_2d_base(pts):
    """
    Base 2D signal function with reasonable numerical range.
    
    Uses a polynomial-trigonometric combination with reasonable range
    instead of exponential functions that cause numerical issues.
    Values are approximately in [0.5, 2.0], which is appropriate for noise σ=0.1.
    """
    x1 = pts[:, 0]
    x2 = pts[:, 1]
    
    # Use a polynomial-trigonometric combination with reasonable range
    base_value = (1.0 + 0.5 * (x1 + x2) + 0.3 * x1 * x2 + 0.2 * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2))
    
    return base_value


def _subdomain_indicator_2d(x1, x2, quad1=False, quad2=False, quad3=False, quad4=False):
    """
    Indicator function for 2D subdomains (quadrants).
    
    Quadrants:
    - quad1: (0.5, 1] × (0.5, 1]  (top-right)
    - quad2: [0, 0.5] × (0.5, 1]  (top-left)  
    - quad3: [0, 0.5] × [0, 0.5]  (bottom-left)
    - quad4: (0.5, 1] × [0, 0.5]  (bottom-right)
    """
    indicator = np.zeros_like(x1)
    
    if quad1:  # Top-right
        indicator += ((x1 > 0.5) & (x1 <= 1) & (x2 > 0.5) & (x2 <= 1)).astype(int)
    if quad2:  # Top-left
        indicator += ((x1 >= 0) & (x1 <= 0.5) & (x2 > 0.5) & (x2 <= 1)).astype(int)
    if quad3:  # Bottom-left
        indicator += ((x1 >= 0) & (x1 <= 0.5) & (x2 >= 0) & (x2 <= 0.5)).astype(int)
    if quad4:  # Bottom-right
        indicator += ((x1 > 0.5) & (x1 <= 1) & (x2 >= 0) & (x2 <= 0.5)).astype(int)
    
    return indicator


def exp_2d_subdomain_modulation(pts, quad1=False, quad2=False, quad3=False, quad4=False, c=1.0):
    """
    2D base function with subdomain modulation.
    
    Uses sin(2πx1)*sin(2πx2) modulation to ensure non-zero signal in target quadrants.
    """
    x1 = pts[:, 0]
    x2 = pts[:, 1]
    
    base = exp_2d_base(pts)
    
    modulation_func = np.sin(2 * np.pi * x1) * np.sin(2 * np.pi * x2)  
    
    # Apply subdomain indicators
    indicator = _subdomain_indicator_2d(x1, x2, quad1, quad2, quad3, quad4)
    
    # Apply modulation only in specified quadrants
    modulation = c * modulation_func * indicator
    
    return base + modulation


def exp_2d_subdomain_constant_modulation(pts, quad1=False, quad2=False, quad3=False, quad4=False, c=1.0):
    """
    2D base function with constant subdomain modulation.
    
    Uses constant c modulation in target quadrants.
    """
    x1 = pts[:, 0]
    x2 = pts[:, 1]
    
    base = exp_2d_base(pts)
    
    # Apply subdomain indicators
    indicator = _subdomain_indicator_2d(x1, x2, quad1, quad2, quad3, quad4)
    
    # Apply constant modulation only in specified quadrants
    modulation = c * indicator
    
    return base + modulation


def get_available_scenarios(dimension='1D'):
    """
    Get list of available scenario names for specified dimension.
    
    Parameters:
        dimension (str): '1D' or '2D'
        
    Returns:
        list: Available scenario identifiers
    """
    if dimension == '1D':
        return [
            # Global scenarios
            'constant_vs_linear',
            'exp_vs_exp_linear', 
            'sine_vs_sine_linear',
            'constant_vs_sine',
            'exp_vs_exp_sine',
            'sine_vs_double_sine',
            # Subdomain scenarios
            'exp_vs_sub1_sine',
            'exp_vs_sub1_cosine',
            'exp_vs_sub12_sine'
        ]
    elif dimension == '2D':
        return [
            'exp_2d_vs_quad1_modulation',
            'exp_2d_vs_quad2_modulation',
            'exp_2d_vs_quad3_modulation', 
            'exp_2d_vs_quad4_modulation',
            'exp_2d_vs_multi_quad_modulation',
            'exp_2d_vs_all_quad_modulation',
            'exp_2d_vs_all_quad_constant'
        ]
    else:
        raise ValueError("dimension must be '1D' or '2D'")