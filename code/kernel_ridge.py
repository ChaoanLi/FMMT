"""
Kernel ridge regression module for nonparametric function estimation.

This module implements KRR with isotropic Matérn kernel (ν=3.5, θ=1.0)
and proper noise variance estimation using effective degrees of freedom.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import gamma, kv
from sklearn.model_selection import KFold
import warnings


def matern_kernel(X1, X2=None, nu=3.5, theta=1.0):
    """
    Compute isotropic Matérn kernel matrix.
    
    Parameters:
        X1 (array): First set of points (n1, d)
        X2 (array): Second set of points (n2, d), if None use X1
        nu (float): Smoothness parameter (default: 3.5)
        theta (float): Scale parameter (default: 1.0)
        
    Returns:
        array: Kernel matrix (n1, n2)
    """
    if X2 is None:
        X2 = X1
        
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    
    # Compute pairwise distances
    distances = cdist(X1, X2, metric='euclidean')
    
    # Avoid division by zero
    distances = np.maximum(distances, 1e-12)
    
    # Matérn kernel computation
    sqrt_2nu = np.sqrt(2 * nu)
    scaled_distances = sqrt_2nu * distances / theta
    
    # Special case: when distance is 0
    kernel_matrix = np.ones_like(distances)
    
    # Non-zero distances
    nonzero_mask = scaled_distances > 1e-10
    if np.any(nonzero_mask):
        nonzero_distances = scaled_distances[nonzero_mask]
        
        # Matérn kernel: (2^(1-ν) / Γ(ν)) * (√(2ν) * r / θ)^ν * K_ν(√(2ν) * r / θ)
        coefficient = (2**(1 - nu)) / gamma(nu)
        power_term = nonzero_distances ** nu
        bessel_term = kv(nu, nonzero_distances)
        
        kernel_matrix[nonzero_mask] = coefficient * power_term * bessel_term
    
    return kernel_matrix


class MaternKernelRidge:
    """
    Kernel Ridge Regression with isotropic Matérn kernel.
    
    Implements the theoretical approach with:
    - Matérn kernel (ν=3.5, θ=1.0)
    - Regularization λ_n = C_n / n
    - Cross-validation for C_n optimization
    - Proper noise variance estimation using effective degrees of freedom
    """
    
    def __init__(self, nu=3.5, theta=1.0, cv_folds=5):
        """
        Initialize Matérn KRR estimator.
        
        Parameters:
            nu (float): Matérn kernel smoothness parameter
            theta (float): Matérn kernel scale parameter  
            cv_folds (int): Number of cross-validation folds
        """
        self.nu = nu
        self.theta = theta
        self.cv_folds = cv_folds
        
        # Will be set during fitting
        self.X_train = None
        self.y_train = None
        self.alpha_coeff = None
        self.optimal_C = None
        self.gram_matrix = None
        self.smoother_matrix = None
        self.noise_variance = None
        self.n_samples = None
        
    def _compute_gram_matrix(self, X):
        """Compute Gram matrix K using Matérn kernel."""
        return matern_kernel(X, X, self.nu, self.theta)
    
    def _cross_validate_regularization(self, X, y):
        """
        Cross-validate regularization parameter C_n.
        
        Search over log-spaced grid from 10^(-9) to 1 with 10 values.
        """
        n = len(y)
        C_candidates = np.logspace(-9, 0, 10)  # 10 values from 10^(-9) to 1
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for C in C_candidates:
            fold_scores = []
            lambda_n = C / n
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                # Compute Gram matrix for training fold
                K_train = self._compute_gram_matrix(X_train_fold)
                n_train = len(train_idx)
                
                # Solve for coefficients
                try:
                    alpha_fold = np.linalg.solve(K_train + lambda_n * np.eye(n_train), y_train_fold)
                except np.linalg.LinAlgError:
                    # Fallback to pseudoinverse if singular
                    alpha_fold = np.linalg.lstsq(K_train + lambda_n * np.eye(n_train), y_train_fold, rcond=None)[0]
                
                # Predict on validation set
                K_val = matern_kernel(X_val_fold, X_train_fold, self.nu, self.theta)
                y_pred = K_val @ alpha_fold
                
                # Compute MSE
                mse = np.mean((y_val_fold - y_pred) ** 2)
                fold_scores.append(mse)
            
            cv_scores.append(np.mean(fold_scores))
        
        # Select optimal C
        optimal_idx = np.argmin(cv_scores)
        return C_candidates[optimal_idx]
    
    def _estimate_noise_variance(self, X, y, lambda_n):
        """
        Estimate noise variance using effective degrees of freedom.
        
        Formula: σ̂_n^2 = Σ(y_i - f̂_n(x_i))^2 / (n - tr(S))
        where S = K(K + λ_n I)^(-1)
        """
        n = len(y)
        K = self._compute_gram_matrix(X)
        
        # Compute smoother matrix S = K(K + λ_n I)^(-1)
        try:
            K_reg_inv = np.linalg.inv(K + lambda_n * np.eye(n))
            S = K @ K_reg_inv
            self.smoother_matrix = S
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            K_reg_inv = np.linalg.pinv(K + lambda_n * np.eye(n))
            S = K @ K_reg_inv
            self.smoother_matrix = S
        
        # Compute trace of smoother matrix
        trace_S = np.trace(S)
        
        # Compute fitted values
        y_fitted = S @ y
        
        # Compute residuals
        residuals = y - y_fitted
        sum_squared_residuals = np.sum(residuals ** 2)
        
        # Estimate noise variance with effective degrees of freedom
        if n > trace_S:
            # Standard formula
            sigma_squared = sum_squared_residuals / (n - trace_S)
        else:
            # Fallback for numerical stability when n ≤ tr(S)
            sigma_squared = sum_squared_residuals / n
            warnings.warn("Using fallback noise variance estimator: n ≤ tr(S)")
        
        return max(sigma_squared, 1e-8)  # Ensure positive variance
    
    def fit(self, X, y):
        """
        Fit Matérn KRR model.
        
        Parameters:
            X (array): Input samples (n_samples, n_features)
            y (array): Target values (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.array(y).ravel()
        
        if X.shape[0] != len(y):
            raise ValueError(f"Sample size mismatch: X has {X.shape[0]} rows, y has {len(y)} elements")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.n_samples = len(y)
        
        # Cross-validate regularization parameter
        self.optimal_C = self._cross_validate_regularization(X, y)
        lambda_n = self.optimal_C / self.n_samples
        
        # Compute Gram matrix
        self.gram_matrix = self._compute_gram_matrix(X)
        
        # Solve for coefficients: α = (K + λ_n I)^(-1) y
        try:
            self.alpha_coeff = np.linalg.solve(self.gram_matrix + lambda_n * np.eye(self.n_samples), y)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            self.alpha_coeff = np.linalg.lstsq(self.gram_matrix + lambda_n * np.eye(self.n_samples), y, rcond=None)[0]
        
        # Estimate noise variance
        self.noise_variance = self._estimate_noise_variance(X, y, lambda_n)
        
        return self
    
    def predict(self, X):
        """
        Predict using fitted Matérn KRR model.
        
        Parameters:
            X (array): Input samples (n_test, n_features)
            
        Returns:
            array: Predictions (n_test,)
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.atleast_2d(X)
        
        # Compute kernel matrix between test and training points
        K_test = matern_kernel(X, self.X_train, self.nu, self.theta)
        
        # Predict: f(x) = K(x, X_train) @ α
        predictions = K_test @ self.alpha_coeff
        
        return predictions
    
    def get_noise_std(self):
        """Get estimated noise standard deviation."""
        if self.noise_variance is None:
            raise ValueError("Model must be fitted first")
        return np.sqrt(self.noise_variance)
    
    def get_effective_dof(self):
        """Get effective degrees of freedom (trace of smoother matrix)."""
        if self.smoother_matrix is None:
            raise ValueError("Model must be fitted first")
        return np.trace(self.smoother_matrix)


def kernel_ridge_regression(X, y, kernel='matern', alpha_range=None, gamma_range=None, 
                          cv_folds=5, scoring='neg_mean_squared_error', 
                          noise_estimation_method='effective_dof'):
    """
    Perform kernel ridge regression with Matérn kernel.
    
    This function provides a compatibility interface for the statistical tests,
    using the theoretically correct Matérn KRR implementation.
    
    Parameters:
        X (array): Input samples (n_samples, n_features)
        y (array): Target values (n_samples,)
        kernel (str): Kernel type (only 'matern' supported in this implementation)
        alpha_range (array): Ignored (regularization optimized via cross-validation)
        gamma_range (array): Ignored (Matérn parameters fixed)
        cv_folds (int): Number of cross-validation folds
        scoring (str): Ignored (uses MSE internally)
        noise_estimation_method (str): Must be 'effective_dof'
        
    Returns:
        tuple: (fitted_model, estimated_noise_std)
    """
    if kernel != 'matern':
        warnings.warn(f"Only Matérn kernel supported, ignoring kernel='{kernel}'")
    
    if noise_estimation_method != 'effective_dof':
        warnings.warn(f"Only effective_dof noise estimation supported, ignoring method='{noise_estimation_method}'")
    
    # Fit Matérn KRR model
    model = MaternKernelRidge(nu=3.5, theta=1.0, cv_folds=cv_folds)
    model.fit(X, y)
    
    # Get noise standard deviation
    noise_std = model.get_noise_std()
    
    return model, noise_std


def estimate_noise_variance(X, y, method='effective_dof'):
    """
    Estimate noise variance using effective degrees of freedom method.
    
    This function provides compatibility with existing code while ensuring
    the theoretically correct noise estimation is used.
    
    Parameters:
        X (array): Input samples (n_samples, n_features)
        y (array): Target values (n_samples,)
        method (str): Must be 'effective_dof'
        
    Returns:
        float: Estimated noise variance
    """
    if method != 'effective_dof':
        warnings.warn(f"Only effective_dof method supported, ignoring method='{method}'")
    
    # Fit temporary model to estimate noise
    model = MaternKernelRidge(nu=3.5, theta=1.0, cv_folds=5)
    model.fit(X, y)
    
    return model.noise_variance


def validate_krr_performance(X_train, y_train, X_test, y_test, fitted_model):
    """
    Validate Matérn kernel ridge regression performance on test data.
    
    Parameters:
        X_train (array): Training input samples
        y_train (array): Training target values
        X_test (array): Test input samples
        y_test (array): Test target values
        fitted_model: Fitted MaternKernelRidge model
        
    Returns:
        dict: Performance metrics and diagnostics
    """
    # Training predictions
    y_train_pred = fitted_model.predict(X_train)
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    train_r2 = 1 - np.var(y_train - y_train_pred) / np.var(y_train) if np.var(y_train) > 0 else 0
    
    # Test predictions
    y_test_pred = fitted_model.predict(X_test)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    test_r2 = 1 - np.var(y_test - y_test_pred) / np.var(y_test) if np.var(y_test) > 0 else 0
    
    # Bias analysis
    train_bias = np.mean(y_train - y_train_pred)
    test_bias = np.mean(y_test - y_test_pred)
    
    # Overfitting indicator
    overfitting_ratio = test_mse / train_mse if train_mse > 0 else np.inf
    
    # Model-specific diagnostics
    effective_dof = fitted_model.get_effective_dof()
    noise_std = fitted_model.get_noise_std()
    
    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_bias': train_bias,
        'test_bias': test_bias,
        'overfitting_ratio': overfitting_ratio,
        'is_overfitting': overfitting_ratio > 2.0,
        'effective_dof': effective_dof,
        'noise_std': noise_std,
        'optimal_C': fitted_model.optimal_C,
        'n_samples': fitted_model.n_samples
    }


def adaptive_kernel_selection(X, y, kernels=['matern'], cv_folds=5):
    """
    Kernel selection function for compatibility.
    
    In this implementation, only Matérn kernel is supported as per theoretical requirements.
    
    Parameters:
        X (array): Input samples
        y (array): Target values
        kernels (list): Ignored (only Matérn supported)
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        str: Always returns 'matern'
    """
    if 'matern' not in kernels:
        warnings.warn("Only Matérn kernel supported, overriding kernel selection")
    
    return 'matern'



