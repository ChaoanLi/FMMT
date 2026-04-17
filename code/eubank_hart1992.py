"""
Eubank & Hart (1992) Goodness-of-Fit Test via Order Selection

Reference:
    Eubank, R.L. & Hart, J.D. (1992). Testing Goodness-of-Fit in Regression
    Via Order Selection Criteria. The Annals of Statistics, 20(3), 1412-1425.

Mathematical specification (verified against original paper, pp. 1412-1425):

Problem setting (Section 2):
    Y_j = g(x_j) + eps_j,  j = 1,...,n
    Fixed design: 0 < x_1 < ... < x_n < 1
    H0: g(x) = sum_{j=1}^p beta_j t_j(x)  [parametric null with p known basis functions]
    Ha: g(x) = sum beta_j t_j(x) + f(x)   [f not a linear combination of t_1,...,t_p]

Orthonormal basis (Section 2, conditions above Eq 2.1):
    Functions u_{jn} satisfy:
        sum_r u_{jn}(x_r) u_{ln}(x_r) = n * delta_{jl}    [Eq above 2.1, orthonormality]
        sum_r u_{jn}(x_r) t_l(x_r) = 0  for l=1,...,p     [orthogonality to null space]

Sample Fourier coefficients (Eq 2.1):
    a_{jn} = (1/n) sum_{r=1}^n u_{jn}(x_r) Y_r

Objective function (Eq 2.6):
    r(0) = 0
    r(k) = sum_{j=1}^k a_{jn}^2  -  k * c_alpha * sigma_hat^2 / n,  k=1,...,n-p

Decision rule (Eq 2.7):
    k_hat = argmax_{k in {0,...,n-p}} r(k)
    Reject H0  iff  k_hat >= 1

Critical values c_alpha (Table 1, p. 1416):
    alpha = 0.01 -> c_alpha = 6.74
    alpha = 0.05 -> c_alpha = 4.18
    alpha = 0.10 -> c_alpha = 3.22
    alpha = 0.20 -> c_alpha = 2.38
    alpha = 0.29 -> c_alpha = 2.00

Approximation formula for c_alpha (Eq 2.8):
    1 - alpha = exp{ -sum_{j=1}^inf (1/j) P(chi^2_j > c_alpha * j) }

Example 1 -- Testing for no effect (p. 1416):
    p=1, t_1(x) = 1 (constant null), uniform design x_r = (r-0.5)/n
    Basis: u_j(x) = sqrt(2) * cos(pi * j * x) for j = 1,...,n-1

Adaptation to model validator:
    Applied with p=0 (null model y^s(x) is fully known, no free parameters).
    Residuals r_i = Y_i - y^s(x_i) are tested for E[r|x] = 0.
    With p=0, the full orthonormal basis at design points is used,
    including the constant term (detects mean shifts).
    The c_alpha values remain valid for any p by Theorem 3.1.
"""

import numpy as np
import scipy.stats as stats


# =============================================================================
# CRITICAL VALUE LOOKUP AND COMPUTATION
# =============================================================================

# Table 1, p. 1416 of Eubank & Hart (1992)
_C_ALPHA_TABLE = {
    0.01:  6.74,
    0.05:  4.18,
    0.10:  3.22,
    0.20:  2.38,
    0.29:  2.00,
}


def _eq28_lhs(c_alpha, n_terms=500):
    """
    Compute sum_{j=1}^{n_terms} (1/j) P(chi^2_j > c_alpha * j)
    from the approximation formula (Eq 2.8, p. 1416).

    Parameters:
        c_alpha (float): Threshold candidate
        n_terms (int): Number of terms in the truncated sum

    Returns:
        float: The sum (should equal -log(1 - alpha) at the correct c_alpha)
    """
    total = 0.0
    for j in range(1, n_terms + 1):
        p_exceed = 1.0 - stats.chi2.cdf(c_alpha * j, df=j)
        total += p_exceed / j
        if p_exceed < 1e-14:
            break
    return total


def compute_c_alpha(alpha, tol=1e-5, n_terms=500):
    """
    Compute c_alpha by numerically solving Eq 2.8 of Eubank & Hart (1992):
        1 - alpha = exp{ -sum_{j=1}^inf (1/j) P(chi^2_j > c_alpha * j) }

    First checks the published Table 1 values; uses the formula otherwise.

    Parameters:
        alpha (float): Target significance level
        tol (float): Numerical tolerance for bisection
        n_terms (int): Number of terms in the approximating sum

    Returns:
        float: c_alpha threshold
    """
    if alpha in _C_ALPHA_TABLE:
        return _C_ALPHA_TABLE[alpha]

    # Target: _eq28_lhs(c_alpha) = -log(1 - alpha)
    target = -np.log(1.0 - alpha)

    # Bisection on c_alpha in [1.0, 20.0]
    lo, hi = 1.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = _eq28_lhs(mid, n_terms)
        if val > target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return (lo + hi) / 2.0


# =============================================================================
# VARIANCE ESTIMATION
# =============================================================================

def difference_based_variance(Y, X=None):
    """
    Rice (1984) / Hall-Kay-Titterington difference-based variance estimator.

    sigma_hat^2 = 0.5 * mean( (Y_{r+1} - Y_r)^2 )

    Applied to Y sorted by X. This estimator is consistent under H0 and
    approximately unbiased regardless of the functional form of g.

    Reference: Rice (1984), Hall, Kay & Titterington (1990), both cited in
    Eubank & Hart (1992) p. 1415 as recommended estimators for sigma^2.

    Parameters:
        Y (array): Response values, shape (n,)
        X (array, optional): Design points for sorting. If None, Y is assumed sorted.

    Returns:
        float: Estimated variance sigma_hat^2
    """
    Y = np.asarray(Y).ravel()
    if X is not None:
        sort_idx = np.argsort(np.asarray(X).ravel())
        Y = Y[sort_idx]
    diffs = np.diff(Y)
    return 0.5 * np.mean(diffs ** 2)


# =============================================================================
# ORTHONORMAL BASIS CONSTRUCTION
# =============================================================================

def _cosine_basis_matrix(X, n_basis):
    """
    Evaluate cos-basis functions at design points.

    For Example 1 of E&H (p=1, constant null), the starting set is:
        phi_j(x) = sqrt(2) * cos(pi * j * x),  j = 1,...,n_basis

    These are orthogonal to the constant 1 in the continuous L2 sense.
    They are used as the starting set for Gram-Schmidt orthonormalization
    in the (1/n)-scaled discrete inner product.

    Parameters:
        X (array): Sorted design points, shape (n,)
        n_basis (int): Number of basis functions to generate

    Returns:
        array: Phi of shape (n, n_basis)
    """
    n = len(X)
    Phi = np.empty((n, n_basis))
    for j in range(n_basis):
        Phi[:, j] = np.sqrt(2.0) * np.cos(np.pi * (j + 1) * X)
    return Phi


def _full_basis_matrix(X, n_basis):
    """
    Full cosine basis including the constant term (for p=0 null).

    Columns: [1, sqrt(2)*cos(pi*x), sqrt(2)*cos(2*pi*x), ...]
    Total n_basis functions spanning all of R^n at the design points.

    Parameters:
        X (array): Sorted design points, shape (n,)
        n_basis (int): Number of basis functions (<= n)

    Returns:
        array: Phi of shape (n, n_basis)
    """
    n = len(X)
    Phi = np.empty((n, n_basis))
    Phi[:, 0] = 1.0  # constant term
    for j in range(1, n_basis):
        Phi[:, j] = np.sqrt(2.0) * np.cos(np.pi * j * X)
    return Phi


def construct_orthonormal_basis(X, n_basis, include_constant=False):
    """
    Construct orthonormal basis at design points via QR decomposition.

    Produces a matrix U of shape (n, n_basis) whose columns satisfy the
    discrete orthonormality condition used in Eubank & Hart (1992):

        (1/n) * U.T @ U = I_{n_basis}

    Procedure:
      1. Build the starting cosine basis Phi at design points X.
      2. Scale: Phi_tilde = Phi / sqrt(n)  (maps to standard inner product).
      3. Thin QR: Phi_tilde = Q R  =>  Q.T @ Q = I.
      4. U = sqrt(n) * Q satisfies (1/n) U.T @ U = I.

    Parameters:
        X (array): Design points (will be sorted internally), shape (n,)
        n_basis (int): Number of orthonormal basis functions to construct
        include_constant (bool): If True, include the constant function as the
            first basis element (appropriate when p=0).
            If False, use cosine-only basis (appropriate when p=1, constant null).

    Returns:
        tuple: (U, X_sorted)
            U (array): Orthonormal basis matrix, shape (n, n_basis),
                       satisfying (1/n) U.T @ U = I.
            X_sorted (array): Sorted design points, shape (n,).
    """
    X = np.asarray(X).ravel()
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    n = len(X_sorted)

    n_basis = min(n_basis, n)

    if include_constant:
        Phi = _full_basis_matrix(X_sorted, n_basis)
    else:
        Phi = _cosine_basis_matrix(X_sorted, n_basis)

    # Scale to standard inner product space
    Phi_tilde = Phi / np.sqrt(n)

    # Thin QR factorization: Phi_tilde = Q R, Q.T @ Q = I
    Q, _ = np.linalg.qr(Phi_tilde, mode='reduced')

    # Restore to (1/n)-inner-product orthonormality: U = sqrt(n) * Q
    U = np.sqrt(n) * Q

    return U, X_sorted


# =============================================================================
# CORE TEST STATISTIC
# =============================================================================

def compute_sample_fourier_coefficients(U, r, n):
    """
    Compute sample Fourier coefficients a_{jn} = (1/n) sum_r u_{jn}(x_r) r_r.

    This is Eq (2.1) from Eubank & Hart (1992), applied to the residuals r.

    Parameters:
        U (array): Orthonormal basis matrix, shape (n, n_basis),
                   satisfying (1/n) U.T @ U = I.
        r (array): Residuals at sorted design points, shape (n,)
        n (int): Sample size

    Returns:
        array: Coefficients a, shape (n_basis,)
    """
    return (1.0 / n) * (U.T @ r)


def compute_objective_function(a, sigma_hat_sq, n, c_alpha):
    """
    Compute the objective function r(k) for k = 0, 1, ..., K.

    From Eq (2.6) of Eubank & Hart (1992):
        r(0) = 0
        r(k) = sum_{j=1}^k a_j^2  -  k * c_alpha * sigma_hat^2 / n,  k >= 1

    Parameters:
        a (array): Sample Fourier coefficients, shape (K,)
        sigma_hat_sq (float): Estimated noise variance sigma^2
        n (int): Sample size
        c_alpha (float): Critical value for significance level alpha

    Returns:
        array: r_k of shape (K+1,) where r_k[0] = 0 and r_k[k] = r(k) for k>=1
    """
    K = len(a)
    r_k = np.zeros(K + 1)
    cumsum_a2 = np.cumsum(a ** 2)
    penalty_per_term = c_alpha * sigma_hat_sq / n
    for k in range(1, K + 1):
        r_k[k] = cumsum_a2[k - 1] - k * penalty_per_term
    return r_k


# =============================================================================
# MAIN HYPOTHESIS TEST
# =============================================================================

def eubank_hart_test(X, Y, computer_model,
                     alpha=0.05,
                     sigma=None,
                     n_basis_max=None,
                     null_type='p0',
                     return_details=False):
    """
    Eubank & Hart (1992) goodness-of-fit test for regression models.

    Tests the null hypothesis:
        H0: E[Y | x] = computer_model(x)  for all x in [0, 1]

    The test is adapted for the model validator setting where the null
    model is fully specified (no free parameters). Two variants are
    available via `null_type`:

      - 'p0' (recommended for model validation):
            Null model is fully specified (p=0 free parameters).
            Residuals r_i = Y_i - computer_model(X_i) are tested against
            E[r|x] = 0 using a full orthonormal basis that includes the
            constant function. Detects mean shifts AND functional deviations.

      - 'p1' (standard Example 1 of E&H):
            Null model is constant (p=1, t_1 = 1).
            Uses cosine basis orthogonal to the constant.
            Does NOT detect pure mean shifts (constant bias in residuals).
            Matches the "testing for no effect" setting of Example 1.

    The c_alpha values from Table 1 of E&H are valid for both variants
    (Theorem 3.1 holds for any p).

    Parameters:
        X (array): Covariate values, shape (n,) or (n,1)
        Y (array): Response values, shape (n,)
        computer_model (callable): Known null function  x (1D array) -> y (1D array)
        alpha (float): Significance level. Must be in {0.01, 0.05, 0.10} for exact
            calibration from E&H Table 1, or any value in (0, 0.5) for the
            numerical approximation via Eq (2.8).
        sigma (float, optional): Known noise std. If None, estimated from residuals
            using the difference-based Rice (1984) estimator.
        n_basis_max (int, optional): Maximum number of basis functions k_max.
            Defaults to n-1 for p1 and n for p0.
        null_type (str): 'p0' or 'p1' (see above).
        return_details (bool): If True, return a dict instead of a boolean.

    Returns:
        bool: True if H0 is rejected, False otherwise.
        OR (if return_details=True): dict containing:
            'reject' (bool): Test decision.
            'k_hat' (int): Selected order k̂.
            'alpha' (float): Significance level.
            'c_alpha' (float): Threshold used.
            'sigma_hat' (float): Noise std estimate.
            'a' (array): Sample Fourier coefficients.
            'r_k' (array): Objective function values.
            'n_basis_used' (int): Number of basis functions used.

    Notes:
        For non-uniform random design (e.g., truncated normal), the c_alpha
        values from Table 1 are asymptotically valid but finite-sample
        Type I error may deviate slightly from nominal alpha.

        For uniform fixed design x_r = (r-0.5)/n, the calibration is exact.
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    n = len(Y)

    # Compute residuals: r_i = Y_i - computer_model(X_i)
    r_raw = Y - computer_model(X)

    # Sort by design points (E&H requires ordered x_1 < ... < x_n)
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    r_sorted = r_raw[sort_idx]

    # Variance estimation
    if sigma is not None:
        sigma_hat_sq = float(sigma) ** 2
    else:
        sigma_hat_sq = difference_based_variance(r_sorted)
        sigma_hat_sq = max(sigma_hat_sq, 1e-12)

    # Number of basis functions
    if null_type == 'p1':
        # Cosine basis orthogonal to constant (n-1 terms, p=1 null)
        n_basis_default = n - 1
        include_constant = False
    else:
        # Full basis including constant (n terms, p=0 null)
        n_basis_default = n
        include_constant = True

    if n_basis_max is None:
        n_basis_max = n_basis_default
    n_basis_max = min(n_basis_max, n_basis_default)

    # Critical value
    c_alpha = compute_c_alpha(alpha)

    # Construct orthonormal basis at sorted design points
    U, _ = construct_orthonormal_basis(X_sorted, n_basis_max, include_constant)

    # Sample Fourier coefficients
    a = compute_sample_fourier_coefficients(U, r_sorted, n)

    # Objective function r(k)
    r_k = compute_objective_function(a, sigma_hat_sq, n, c_alpha)

    # Optimal order
    k_hat = int(np.argmax(r_k))

    # Decision rule (Eq 2.7): reject H0 if k_hat >= 1
    reject = k_hat >= 1

    if return_details:
        return {
            'reject': reject,
            'k_hat': k_hat,
            'alpha': alpha,
            'c_alpha': c_alpha,
            'sigma_hat': float(np.sqrt(sigma_hat_sq)),
            'a': a,
            'r_k': r_k,
            'n_basis_used': n_basis_max,
        }
    return reject


# =============================================================================
# SIMULATION-BASED DATA GENERATION FOR COMPARISON
# =============================================================================

def generate_eh_comparison_data(f1, f2, n, sigma, domain=(0.0, 1.0),
                                 dist_type='uniform', mu=0.5, sigma_x=0.2):
    """
    Generate one-sample data for E&H comparison experiments.

    The model validator setting requires one sample of (X, Y) where
    the null model y^s = f1 is known. The test is:
        H0: f = f1    vs    H1: f = f2 != f1

    Under H0: generate Y = f1(X) + noise(sigma)
    Under H1: generate Y = f2(X) + noise(sigma)
    Then test against computer model y^s = f1.

    Parameters:
        f1 (callable): Null (computer) model
        f2 (callable): True data-generating model
        n (int): Sample size
        sigma (float): Noise standard deviation
        domain (tuple): Covariate domain (a, b)
        dist_type (str): 'uniform' or 'truncated_normal'
        mu (float): Mean for truncated normal design
        sigma_x (float): Std for truncated normal design

    Returns:
        tuple: (X, Y, f1) where f1 is the computer model callable
    """
    a, b = domain

    if dist_type == 'uniform':
        X = np.random.uniform(a, b, n)
    elif dist_type == 'truncated_normal':
        from scipy.stats import truncnorm
        lo, hi = (a - mu) / sigma_x, (b - mu) / sigma_x
        X = truncnorm.rvs(lo, hi, loc=mu, scale=sigma_x, size=n)
    else:
        raise ValueError(f"Unknown dist_type '{dist_type}'")

    Y = f2(X) + np.random.normal(0.0, sigma, n)

    def computer_model(x):
        return f1(np.asarray(x).ravel())

    return X, Y, computer_model


def eh_hypothesis_test(f1, f2, n, sigma, alpha=0.05,
                        dist_type='truncated_normal',
                        null_type='p0', n_basis_max=None):
    """
    Run one Monte Carlo trial of the E&H goodness-of-fit test.

    Under H0: f1 = f2 (same data-generating function as computer model).
    Under H1: f2 != f1 (computer model is misspecified).

    Parameters:
        f1 (callable): Computer model (null function)
        f2 (callable): True data-generating function
        n (int): Sample size
        sigma (float): Noise standard deviation
        alpha (float): Significance level
        dist_type (str): Covariate distribution type
        null_type (str): 'p0' or 'p1' (see eubank_hart_test)
        n_basis_max (int, optional): Max number of basis functions

    Returns:
        dict: {'reject': bool, 'k_hat': int, 'sigma_hat': float}
    """
    X, Y, computer_model = generate_eh_comparison_data(
        f1, f2, n, sigma, dist_type=dist_type
    )
    result = eubank_hart_test(
        X, Y, computer_model,
        alpha=alpha,
        n_basis_max=n_basis_max,
        null_type=null_type,
        return_details=True,
    )
    return result
