import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity # For kernel density estimation

# =============================================================================
# Helper Functions (Kernels, etc.)
# =============================================================================

def epanechnikov_kernel(u):
    """
    Epanechnikov kernel function.
    This kernel is used in the simulation section of the paper.
    """
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

def I_w_gaussian(t):
    """
    Computes the function I_w(t).
    Corresponds to Step 3. We choose w(t) as the standard normal PDF,
    which gives I_w(t) a simple analytical form.
    I_w(t) = integral(cos(tx)w(x)dx) = Re{phi_w(t)}.
    For a standard normal distribution, phi_w(t) = exp(-t^2/2), which is real.
    """
    return np.exp(-t**2 / 2)

def D2_Iw_gaussian(t):
    """
    Computes the second derivative of I_w(t).
    Required for estimating matrix A in Step 4.
    d^2/dt^2 [exp(-t^2/2)] = (t^2 - 1)exp(-t^2/2).
    """
    return (t**2 - 1) * np.exp(-t**2 / 2)


# =============================================================================
# Main Implementation Class
# =============================================================================

class RegressionCurveTester:
    """
    A class to perform the non-parametric ANOVA-type test for regression curves
    based on characteristic functions.
    """
    def __init__(self, datasets, h, verbose=True):
        """
        Initializes the tester.

        Args:
            datasets (list of tuples): A list where each element is a tuple (X, Y),
                                     representing the sample data for a group.
                                     X and Y are numpy arrays.
            h (float): The bandwidth (smoothing parameter) for all kernel estimations.
            verbose (bool): Whether to print detailed progress information.
        """
        self.datasets = datasets
        self.k = len(datasets) # Number of groups
        self.n_j = [len(data[0]) for data in datasets] # Sample size for each group
        self.n = sum(self.n_j) # Total sample size
        self.h = h
        self.verbose = verbose
        
        # To store intermediate results
        self.residuals = {}
        self.estimators = {}

    def _nadaraya_watson(self, X_train, Y_train, x_eval):
        """
        Performs Nadaraya-Watson kernel regression.
        
        Args:
            X_train (np.array): Covariates of the training data.
            Y_train (np.array): Responses of the training data.
            x_eval (np.array or float): Points at which to evaluate the regression.
        
        Returns:
            np.array: Estimated values at x_eval points.
        """
        x_eval = np.asarray(x_eval)
        # Use broadcasting for efficient computation of kernel weights
        u = (x_eval[:, np.newaxis] - X_train) / self.h
        K_u = epanechnikov_kernel(u)
        
        # Numerator: sum(K_h * Y)
        numerator = np.sum(K_u * Y_train, axis=1)
        # Denominator: sum(K_h)
        denominator = np.sum(K_u, axis=1)
        
        # Avoid division by zero
        denominator[denominator == 0] = np.nan
        
        return numerator / denominator

    def _kernel_density_estimator(self, X_train, x_eval):
        """
        Performs kernel density estimation using sklearn's KernelDensity,
        which is more stable and efficient than a manual implementation.
        """
        kde = KernelDensity(kernel='epanechnikov', bandwidth=self.h).fit(X_train[:, np.newaxis])
        log_dens = kde.score_samples(x_eval[:, np.newaxis])
        return np.exp(log_dens)

    def step2_nonparametric_estimation(self):
        """
        Executes Step 2: Perform all necessary non-parametric estimations.
        """
        if self.verbose:
            print("Step 2: Performing non-parametric estimations...")
        
        # 1. Estimate m_j(x) and sigma_j^2(x) for each group
        m_j_hats = []
        sigma2_j_hats = []
        
        for j in range(self.k):
            X_j, Y_j = self.datasets[j]
            
            # Estimate m_j(X_jl)
            m_j_hat = self._nadaraya_watson(X_j, Y_j, X_j)
            m_j_hats.append(m_j_hat)
            
            # Estimate sigma_j^2(X_jl)
            Y2_j = Y_j**2
            m_j_hat_sq = self._nadaraya_watson(X_j, Y2_j, X_j)
            sigma2_j_hat = m_j_hat_sq - m_j_hat**2
            # Ensure variance is positive
            sigma2_j_hat[sigma2_j_hat < 1e-6] = 1e-6
            sigma2_j_hats.append(sigma2_j_hat)

        self.estimators['m_j_hats'] = m_j_hats
        self.estimators['sigma2_j_hats'] = sigma2_j_hats
        
        # 2. Estimate the common regression function m_0(x) under H0
        # This requires density estimates f_j(x) first
        all_X = np.concatenate([data[0] for data in self.datasets])
        f_j_hats_at_all_X = []
        
        for j in range(self.k):
            X_j, _ = self.datasets[j]
            # Estimate f_j at all X points
            f_j_hat = self._kernel_density_estimator(X_j, all_X)
            f_j_hats_at_all_X.append(f_j_hat)
            
        # Compute the mixture density f_mix(x)
        f_mix_hat = np.zeros_like(all_X)
        for j in range(self.k):
            f_mix_hat += (self.n_j[j] / self.n) * f_j_hats_at_all_X[j]
        
        # Estimate m_j(x) at all X points
        m_j_hats_at_all_X = []
        for j in range(self.k):
            X_j, Y_j = self.datasets[j]
            m_j_hat_at_all_X = self._nadaraya_watson(X_j, Y_j, all_X)
            m_j_hats_at_all_X.append(m_j_hat_at_all_X)

        # Compute m_0(x)
        m0_hat_at_all_X = np.zeros_like(all_X)
        for j in range(self.k):
            # To avoid division by zero if f_mix_hat is close to zero
            weight = np.divide((self.n_j[j] / self.n) * f_j_hats_at_all_X[j], f_mix_hat, 
                               out=np.zeros_like(f_mix_hat), where=f_mix_hat!=0)
            m0_hat_at_all_X += weight * m_j_hats_at_all_X[j]
            
        # Distribute m0_hat back to each group
        m0_hats = []
        current_pos = 0
        for nj in self.n_j:
            m0_hats.append(m0_hat_at_all_X[current_pos : current_pos + nj])
            current_pos += nj
        self.estimators['m0_hats'] = m0_hats

        # 3. Compute the two sets of residuals
        eps_j_hats = []
        eps_0j_hats = []
        for j in range(self.k):
            _, Y_j = self.datasets[j]
            sigma_j_hat = np.sqrt(sigma2_j_hats[j])
            
            eps_j_hat = (Y_j - m_j_hats[j]) / sigma_j_hat
            eps_0j_hat = (Y_j - m0_hats[j]) / sigma_j_hat
            
            eps_j_hats.append(eps_j_hat)
            eps_0j_hats.append(eps_0j_hat)
            
        self.residuals['eps_j_hats'] = eps_j_hats
        self.residuals['eps_0j_hats'] = eps_0j_hats
        if self.verbose:
            print("Non-parametric estimation complete.")

    def step3_calculate_test_statistic(self):
        """
        Executes Step 3: Calculate the test statistic nT_1n using the
        convenient formula from Remark 2.1 of the paper.
        """
        if self.verbose:
            print("Step 3: Calculating the test statistic nT_1n...")
        
        nT_1n = 0
        eps_j_hats = self.residuals['eps_j_hats']
        eps_0j_hats = self.residuals['eps_0j_hats']
        
        for j in range(self.k):
            nj = self.n_j[j]
            eps_j = eps_j_hats[j]
            eps_0j = eps_0j_hats[j]
            
            # Compute difference matrices
            diff_eps_j = eps_j[:, np.newaxis] - eps_j
            diff_eps_0j = eps_0j[:, np.newaxis] - eps_0j
            diff_eps_j_0j = eps_j[:, np.newaxis] - eps_0j
            
            # Compute the three double summation terms
            term1 = np.sum(I_w_gaussian(diff_eps_j))
            term2 = np.sum(I_w_gaussian(diff_eps_0j))
            term3 = np.sum(I_w_gaussian(diff_eps_j_0j))
            
            # Accumulate to the total statistic
            nT_1n += (1 / nj) * (term1 + term2 - 2 * term3)
            
        self.test_statistic_ = nT_1n
        if self.verbose:
            print(f"Test statistic nT_1n = {self.test_statistic_:.4f}")
        return self.test_statistic_

    def step4_estimate_null_distribution(self):
        """
        Executes Step 4: Estimate the parameters of the asymptotic null distribution.
        """
        if self.verbose:
            print("Step 4: Estimating null distribution parameters (Matrix A and Sigma)...")
        
        # 1. Estimate matrix A (diagonal)
        a_hats = []
        eps_j_hats = self.residuals['eps_j_hats']
        for j in range(self.k):
            nj = self.n_j[j]
            eps_j = eps_j_hats[j]
            
            if nj < 2:
                a_hats.append(0)
                continue
            
            # Use formula (5) from the paper
            # Create all pairs (r < s)
            r, s = np.triu_indices(nj, k=1)
            diffs = eps_j[r] - eps_j[s]
            
            # Compute the U-statistic-like estimator
            a_j_hat = -np.mean(D2_Iw_gaussian(diffs))
            a_hats.append(a_j_hat)
            
        A_hat = np.diag(a_hats)
        self.estimators['A_hat'] = A_hat

        # 2. Estimate matrix Sigma
        Sigma_hat = np.zeros((self.k, self.k))
        p_j = np.array(self.n_j) / self.n
        
        # Simplified estimation for Sigma as in Remark 3.1
        # This assumes f_1=...=f_k and sigma_1=...=sigma_k.
        # The full estimation is considerably more complex to implement.
        
        sqrt_p = np.sqrt(p_j)
        Sigma_hat = np.identity(self.k) - np.outer(sqrt_p, sqrt_p)
            
        self.estimators['Sigma_hat'] = Sigma_hat

        # Compute the eigenvalues of the product matrix A_hat * Sigma_hat
        matrix_prod = np.dot(A_hat, Sigma_hat)
        eigenvalues = np.linalg.eigvals(matrix_prod)
        # Eigenvalues should be real; take real part to avoid numerical noise
        self.eigenvalues_ = np.real(eigenvalues)
        
        if self.verbose:
            print(f"Estimated eigenvalues (betas) = {self.eigenvalues_}")
        return self.eigenvalues_

    def step5_perform_hypothesis_test(self, n_simulations=10000):
        """
        Executes Step 5: Compute the p-value via Monte Carlo simulation and make a decision.
        """
        if self.verbose:
            print("Step 5: Calculating p-value via Monte Carlo simulation...")
        
        # Generate random samples from a chi2(1) distribution
        chi2_samples = chi2.rvs(df=1, size=(n_simulations, self.k))
        
        # Compute the simulated distribution of the weighted sum
        # Use only positive eigenvalues for the simulation
        positive_eigenvalues = self.eigenvalues_[self.eigenvalues_ > 1e-8]
        
        if len(positive_eigenvalues) == 0:
            if self.verbose:
                print("WARNING: No positive eigenvalues found. Cannot compute p-value.")
            return 1.0

        # Simulated test statistic values
        simulated_stats = chi2_samples[:, :len(positive_eigenvalues)] @ positive_eigenvalues
        
        # Compute the p-value
        p_value = np.mean(simulated_stats > self.test_statistic_)
        
        if self.verbose:
            print(f"P-value = {p_value:.4f}")
        return p_value

    def run_test(self, alpha=0.05):
        """
        Runs the complete testing procedure.
        """
        self.step2_nonparametric_estimation()
        self.step3_calculate_test_statistic()
        self.step4_estimate_null_distribution()
        p_value = self.step5_perform_hypothesis_test()
        
        if self.verbose:
            print("\n--- Test Conclusion ---")
            print(f"Significance level alpha = {alpha}")
            if p_value < alpha:
                print(f"P-value ({p_value:.4f}) < alpha ({alpha}).")
                print("Conclusion: Reject the null hypothesis H0. There is significant evidence that the regression curves are not identical.")
            else:
                print(f"P-value ({p_value:.4f}) >= alpha ({alpha}).")
                print("Conclusion: Fail to reject the null hypothesis H0. There is not enough evidence to conclude that the regression curves are different.")
        
        return self.test_statistic_, self.eigenvalues_, p_value


# =============================================================================
# Example Usage (for testing purposes only)
# =============================================================================
if __name__ == '__main__':
    print("PJG2015 RegressionCurveTester is ready for use.")
    print("Please use comparison_pjg2015_experiment.py for comprehensive testing.")
