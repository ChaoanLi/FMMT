"""
FMMT: Fourier Maximum Modulus Test for Computer Model Validation.

Core methodology package implementing the methods described in:
  "Statistical Validation of Computer Models: Global and Subdomain
   Hypothesis Testing" (Li, Zhang, Tuo)

Public API:
    fmmt_1d  - run FMMT on 1D data
    fmmt_2d  - run FMMT on 2D data
"""

from .statistical_tests import (
    compute_1d_pvalue_kde,
    compute_2d_pvalue_kde,
    compute_1d_max_statistic,
    compute_1d_max_statistic_cdf,
    compute_2d_max_statistic,
    compute_2d_max_statistic_cdf,
    compute_1d_test_statistic_kde_fft,
    compute_2d_test_statistic_global,
    compute_2d_test_statistic_subdomain,
)
from .kernel_ridge import MaternKernelRidge, kernel_ridge_regression
from .fourier_basis import (
    get_basis_frequency,
    create_decay_weight_function,
    compute_orthonormal_fourier_coefficients,
)
from .kde import KDE
from .utils import set_random_seed

__all__ = [
    "compute_1d_pvalue_kde",
    "compute_2d_pvalue_kde",
    "compute_1d_max_statistic",
    "compute_1d_max_statistic_cdf",
    "compute_2d_max_statistic",
    "compute_2d_max_statistic_cdf",
    "compute_1d_test_statistic_kde_fft",
    "compute_2d_test_statistic_global",
    "compute_2d_test_statistic_subdomain",
    "MaternKernelRidge",
    "kernel_ridge_regression",
    "get_basis_frequency",
    "create_decay_weight_function",
    "compute_orthonormal_fourier_coefficients",
    "KDE",
    "set_random_seed",
]
