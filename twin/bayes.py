"""Bayesian tracking of the friction parameters (the Module 7 conjugate regression, run cycle after cycle).

Model for one cycle:  y = X beta + noise,  y = sensor - model torque on the moving samples,
X = [1, sign(v), v, a], beta = (c0, tau_c, b, dJ), noise ~ N(0, sigma^2).
Prior: beta | sigma^2 ~ N(m, sigma^2 S), sigma^2 ~ InvGamma(a, b)  (Normal-Inverse-Gamma).
After each cycle the posterior becomes the prior for the next one, with a bit of forgetting so the
reference can follow the machine (it drifts a bit between recording batches) instead of being stuck on the
first cycles.
"""
import numpy as np


class NIG:
    """Normal-Inverse-Gamma posterior over the regression parameters."""

    def __init__(self, m, S, a, b):
        self.m = np.asarray(m, float).copy()
        self.S = np.asarray(S, float).copy()
        self.a = float(a)
        self.b = float(b)
        self.n_seen = 0

    @classmethod
    def from_cycles(cls, betas, noise, n_eff=20.0):
        """A prior from a handful of plain least-squares fits: mean of the fits, spread of the fits,
        worth about n_eff cycles of evidence."""
        betas = np.asarray(betas, float)
        m = betas.mean(axis=0)
        sigma2 = float(np.mean(np.asarray(noise) ** 2))
        cov = np.cov(betas.T) + np.eye(betas.shape[1]) * 1e-9
        S = cov / sigma2                          # so that sigma^2 * S = covariance of the parameters
        a = n_eff / 2.0
        b = a * sigma2
        return cls(m, S, a, b)

    def update(self, X, y):
        """Conjugate update with the samples of one cycle."""
        X = np.asarray(X, float); y = np.asarray(y, float)
        n = len(y)
        S0inv = np.linalg.inv(self.S)
        Sn_inv = S0inv + X.T @ X
        Sn = np.linalg.inv(Sn_inv)
        mn = Sn @ (S0inv @ self.m + X.T @ y)
        an = self.a + n / 2.0
        bn = self.b + 0.5 * float(y @ y + self.m @ S0inv @ self.m - mn @ Sn_inv @ mn)
        self.m, self.S, self.a, self.b = mn, Sn, an, max(bn, 1e-12)
        self.n_seen += 1

    def forget(self, lam):
        """Exponential forgetting: keep the mean, discount the evidence by lam (0 < lam <= 1).
        lam = 0.98 means a memory of roughly 50 cycles."""
        self.S = self.S / lam
        self.a = max(lam * self.a, 1.0)
        self.b = lam * self.b

    @property
    def sigma2(self):
        """Posterior mean of the noise variance."""
        return self.b / max(self.a - 1.0, 1e-9)

    def param_std(self):
        """Marginal standard deviation of each parameter (Student-t with 2a degrees of freedom)."""
        dof = 2 * self.a
        scale = np.sqrt(np.diag(self.S) * self.b / self.a)
        return scale * np.sqrt(dof / max(dof - 2, 1e-9))

    def predictive(self, X):
        """Mean and variance of a new observation at X (posterior predictive, Student-t approximated by its variance)."""
        X = np.asarray(X, float)
        mean = X @ self.m
        var = self.sigma2 * (1.0 + np.einsum("ij,jk,ik->i", X, self.S, X))
        return mean, var

    def sample(self, n, rng=None):
        """Draw parameter vectors from the posterior (for Monte Carlo propagation)."""
        rng = rng or np.random.default_rng()
        sig2 = 1.0 / rng.gamma(self.a, 1.0 / self.b, size=n)
        out = np.empty((n, len(self.m)))
        L = np.linalg.cholesky(self.S + np.eye(len(self.m)) * 1e-12)
        for i in range(n):
            out[i] = self.m + np.sqrt(sig2[i]) * (L @ rng.standard_normal(len(self.m)))
        return out


class ReferenceTracker:
    """What 'normal' looks like right now, for one joint: a NIG posterior over the load-side friction fit
    and a second one over the gear-side fit, updated only with cycles that passed the health gate."""

    def __init__(self, load_prior, gear_prior, lam=0.98):
        self.load = load_prior
        self.gear = gear_prior
        self.lam = lam

    def step(self, X, y_load, y_gear, healthy=True):
        """One cycle. If the cycle is judged healthy it refreshes the reference; otherwise the reference only forgets a little."""
        for post, y in ((self.load, y_load), (self.gear, y_gear)):
            post.forget(self.lam)
            if healthy:
                post.update(X, y)

    def zscore(self, beta_cycle, beta_std, which="load", idx=1):
        """How far a per-cycle estimate sits from the reference, in standard deviations
        (per-cycle spread and reference uncertainty added in quadrature)."""
        post = self.load if which == "load" else self.gear
        ref_std = post.param_std()[idx]
        return (beta_cycle[idx] - post.m[idx]) / np.sqrt(beta_std[idx] ** 2 + ref_std ** 2)
