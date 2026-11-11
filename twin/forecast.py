"""Prognostic layer: follow the trend of tau_c with a Kalman filter and turn it into 'cycles until the limit'.

State x = (level, slope): tau_c now and how much it changes per cycle (local linear trend, Module 9 Kalman filter).
Observation: the per-cycle tau_c estimate with its known noise (0.17 N m on joint 2, Deliverable 2).
Forecast: draw (level, slope) from the posterior, roll the trend forward with its process noise, and read off
the first cycle at which it crosses tau_max. The spread of those crossings is the uncertainty of N*.
"""
import numpy as np


class TrendKF:
    def __init__(self, level0, obs_std=0.17, level_std0=0.3, slope_std0=0.01, q_level=0.01, q_slope=0.001):
        self.x = np.array([level0, 0.0])
        self.P = np.diag([level_std0 ** 2, slope_std0 ** 2])
        self.R = obs_std ** 2
        self.Q = np.diag([q_level ** 2, q_slope ** 2])
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])

    def step(self, y):
        """Predict one cycle ahead, then update with the new estimate y (or only predict if y is None)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if y is not None and np.isfinite(y):
            S = (self.H @ self.P @ self.H.T)[0, 0] + self.R     # 1x1 array -> number (numpy 2 refuses float() on it)
            K = (self.P @ self.H.T) / S
            innov = y - (self.H @ self.x)[0]
            self.x = self.x + (K[:, 0] * innov)
            self.P = (np.eye(2) - K @ self.H) @ self.P
            return innov / np.sqrt(S)                       # standardised innovation, a quick model check
        return np.nan

    @property
    def level(self):
        return float(self.x[0])

    @property
    def slope(self):
        return float(self.x[1])

    def level_std(self):
        return float(np.sqrt(self.P[0, 0]))

    def cycles_to_limit(self, tau_max, horizon=1000, n=2000, rng=None):
        """Monte Carlo over the posterior and the process noise. Returns the crossing cycles (horizon where none)."""
        rng = rng or np.random.default_rng(0)
        X = rng.multivariate_normal(self.x, self.P, size=n)        # (level, slope) samples
        level, slope = X[:, 0].copy(), X[:, 1].copy()
        crossed = np.full(n, horizon, dtype=int)
        alive = level < tau_max
        crossed[~alive] = 0
        ql, qs = np.sqrt(self.Q[0, 0]), np.sqrt(self.Q[1, 1])
        for k in range(1, horizon + 1):
            level = level + slope + rng.normal(0, ql, n)
            slope = slope + rng.normal(0, qs, n)
            hit = alive & (level >= tau_max)
            crossed[hit] = k
            alive &= ~hit
            if not alive.any():
                break
        return crossed


def summarize_crossings(c, horizon):
    """Median and 90 % interval of N*, plus the probability that the limit is crossed within the horizon."""
    return {"median": float(np.median(c)), "lo5": float(np.percentile(c, 5)), "hi95": float(np.percentile(c, 95)),
            "p_within_horizon": float((c < horizon).mean()),
            "p_within_20": float((c <= 20).mean()), "p_within_50": float((c <= 50).mean())}


class WindowTrend:
    """Local linear trend from a sliding window of the last n per-cycle estimates (ordinary least squares).
    Unlike the Kalman filter above it has no memory beyond the window, so it does not lag behind a drift that
    starts after a long healthy history; the price is noisier estimates. Same forecast interface."""

    def __init__(self, n=40, obs_std=0.17, q_level=0.01):
        self.n = n
        self.R = obs_std ** 2
        self.q_level = q_level
        self.hist = []
        self.x = np.array([np.nan, 0.0])
        self.P = np.diag([obs_std ** 2, 1.0])

    def step(self, y):
        if y is not None and np.isfinite(y):
            self.hist.append(float(y))
        h = np.array(self.hist[-self.n:])
        m = len(h)
        if m < 3:
            self.x = np.array([h.mean() if m else np.nan, 0.0]); self.P = np.diag([self.R, 1.0])
            return np.nan
        t = np.arange(m) - (m - 1)                       # time relative to now (now = 0)
        A = np.column_stack([np.ones(m), t])
        beta, res, _, _ = np.linalg.lstsq(A, h, rcond=None)
        s2 = float(res[0] / max(m - 2, 1)) if len(res) else self.R
        s2 = max(s2, 0.25 * self.R)                      # never more confident than the known noise allows
        self.x = beta
        self.P = s2 * np.linalg.inv(A.T @ A)
        return (h[-1] - beta[0]) / np.sqrt(s2)

    @property
    def level(self):
        return float(self.x[0])

    @property
    def slope(self):
        return float(self.x[1])

    def level_std(self):
        return float(np.sqrt(self.P[0, 0]))

    def cycles_to_limit(self, tau_max, horizon=1000, n=2000, rng=None):
        rng = rng or np.random.default_rng(0)
        if not np.isfinite(self.x[0]):
            return np.full(n, horizon, dtype=int)
        X = rng.multivariate_normal(self.x, self.P, size=n)
        level, slope = X[:, 0].copy(), X[:, 1].copy()
        crossed = np.full(n, horizon, dtype=int)
        alive = level < tau_max
        crossed[~alive] = 0
        for k in range(1, horizon + 1):
            level = level + slope + rng.normal(0, self.q_level, n)
            hit = alive & (level >= tau_max)
            crossed[hit] = k
            alive &= ~hit
            if not alive.any():
                break
        return crossed


class RestartingTrendKF(TrendKF):
    """The Kalman trend above, restarted when a drift sets in. After a long healthy history the filter is sure the
    slope is zero and lags behind a drift that starts later; the reference tracker knows when tau_c leaves its
    healthy band, so at that moment the trend is re-estimated from scratch over the last `lookback` cycles
    (fresh prior, then the recent observations replayed), which gives it the slope it had been missing."""

    def __init__(self, level0, ref_mean, ref_std, k_onset=2.0, n_onset=3, lookback=40, **kw):
        super().__init__(level0, **kw)
        self.kw = kw
        self.ref_mean, self.ref_std = ref_mean, ref_std
        self.k_onset, self.n_onset, self.lookback = k_onset, n_onset, lookback
        self.votes = 0
        self.onset = None
        self.t = 0
        self.buf = []

    def step(self, y):
        innov = super().step(y)
        self.t += 1
        if y is not None and np.isfinite(y):
            self.buf.append(float(y)); self.buf = self.buf[-self.lookback:]
            if self.onset is None:
                self.votes = self.votes + 1 if (y - self.ref_mean) > self.k_onset * self.ref_std else 0
                if self.votes >= self.n_onset:
                    self.onset = self.t
                    recent = self.buf
                    self.x = np.array([self.ref_mean, 0.0])
                    self.P = np.diag([self.ref_std ** 2, self.kw.get("slope_std0", 0.01) ** 2])
                    for v in recent:                       # replay the recent past with an open mind about the slope
                        TrendKF.step(self, v)
        return innov

    def update_reference(self, ref_mean, ref_std):
        """Follow the reference tracker while no drift has been declared."""
        if self.onset is None:
            self.ref_mean, self.ref_std = ref_mean, ref_std
