"""Friction model of one joint, fitted cycle by cycle.

The drive gives us three torques per axis:
  computed_torque  - what the controller's rigid-body model expects at the joint (gravity + inertia)
  torque_sensor_a  - measured at the joint output (after the gearbox)
  motor_torque     - motor side (it is basically motor_iq times a constant)

So the differences are physical:
  load side:  sensor - computed = friction/extra load after the gearbox + model error
  gear side:  motor - sensor    = gearbox friction
Each one is fitted with  c0 + tau_c*sign(v) + b*v + dJ*a  on the samples where the joint is moving.
tau_c (the Coulomb-type term) is the health indicator we track. Treat it as an effective number, not the true
friction torque: on joint 3 it comes out negative, so the controller model probably has some friction in it
already. What we rely on is that it moves when friction is added to the joint.
"""
import numpy as np

MOVING = 0.02          # rad/s, below this the sign of the velocity is just noise
PAIRS = {"load": ("torque_sensor_a", "computed_torque"),
         "gear": ("motor_torque", "torque_sensor_a"),
         "total": ("motor_torque", "computed_torque")}


def fit_friction(torque_diff, velocity, acceleration, moving=MOVING):
    """Least squares of torque_diff on [1, sign(v), v, a] while moving. Returns a dict of parameters."""
    v = np.asarray(velocity, float)
    r = np.asarray(torque_diff, float)
    a = np.asarray(acceleration, float)
    m = np.abs(v) > moving
    out = {"c0": np.nan, "coulomb": np.nan, "viscous": np.nan, "inertia": np.nan, "noise": np.nan,
           "rms": np.sqrt(np.mean(r ** 2)), "n_moving": int(m.sum())}
    if m.sum() < 20:
        return out
    X = np.column_stack([np.ones(m.sum()), np.sign(v[m]), v[m], a[m]])
    beta, _, _, _ = np.linalg.lstsq(X, r[m], rcond=None)
    e = r[m] - X @ beta
    out.update(c0=beta[0], coulomb=beta[1], viscous=beta[2], inertia=beta[3], noise=float(np.std(e)))
    # standard errors of the fit: textbook one (white noise) and one corrected for the residuals being
    # correlated in time, which they are - the corrected one is the honest number
    n = len(e)
    s2 = float(e @ e / max(n - 4, 1))
    cov = s2 * np.linalg.inv(X.T @ X)
    rho = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if n > 5 else 0.0
    rho = min(max(rho, 0.0), 0.99)
    infl = (1 + rho) / (1 - rho)                       # variance inflation for an AR(1)-like residual
    se = np.sqrt(np.diag(cov))
    out.update(se_coulomb=float(se[1]), se_coulomb_corr=float(se[1] * np.sqrt(infl)), resid_rho1=rho)
    return out


def design_matrix(velocity, acceleration, moving=MOVING):
    """Same regressors as fit_friction, for the Bayesian update later on."""
    v = np.asarray(velocity, float); a = np.asarray(acceleration, float)
    m = np.abs(v) > moving
    return np.column_stack([np.ones(m.sum()), np.sign(v[m]), v[m], a[m]]), m


def cycle_design(g, moving=MOVING):
    """Design matrix and the two residual torques of one cycle (moving samples only), for the Bayesian update."""
    X, m = design_matrix(g["joint_velocity"].values, g["target_acceleration"].values, moving)
    y_load = (g["torque_sensor_a"].values - g["computed_torque"].values)[m]
    y_gear = (g["motor_torque"].values - g["torque_sensor_a"].values)[m]
    return X, y_load, y_gear


def residual_series(g, moving=MOVING):
    """Whole-cycle residual after the load-side friction fit: what the model does not explain, sample by sample."""
    v = g["joint_velocity"].values; a = g["target_acceleration"].values
    r = g["torque_sensor_a"].values - g["computed_torque"].values
    m = np.abs(v) > moving
    X = np.column_stack([np.ones(len(v)), np.sign(v), v, a])
    if m.sum() < 20:
        return r - r.mean()
    beta, _, _, _ = np.linalg.lstsq(X[m], r[m], rcond=None)
    e = r - X @ beta
    e[~m] = (r - beta[0] - a * beta[3])[~m]          # at rest there is no friction term to subtract
    return e


def cycle_fits(g, pairs=PAIRS):
    """All three friction fits for one cycle (g = frame of that cycle). Keys look like load_coulomb."""
    out = {}
    for name, (hi, lo) in pairs.items():
        f = fit_friction(g[hi].values - g[lo].values, g["joint_velocity"].values, g["target_acceleration"].values)
        out.update({f"{name}_{k}": v for k, v in f.items()})
    return out
