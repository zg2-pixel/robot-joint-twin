"""Noise, resolution and bias of each signal, estimated from the data itself (no datasheet needed).

Three ideas:
  - quantization: the smallest step between distinct values tells the resolution the drive actually logged
  - noise floor: while the joint is at rest the signal should be flat, so whatever moves is noise
  - bias: two sensors measuring the same thing (torque A vs B, encoder vs command, sensor vs model) disagree by
    a steady offset = systematic error, plus a spread = random error
"""
import numpy as np
from scipy.signal import welch

REST_V = 0.005        # rad/s, below this (and no commanded motion) the joint counts as at rest
MIN_SEG = 30          # samples, shorter rest segments are ignored

UNITS = {"joint_position": "rad", "motor_position": "rad", "target_position": "rad",
         "joint_velocity": "rad/s", "motor_velocity": "rad/s", "target_velocity": "rad/s",
         "target_acceleration": "rad/s^2", "torque_sensor_a": "N m", "torque_sensor_b": "N m",
         "motor_torque": "N m", "computed_torque": "N m", "target_torque": "N m", "computed_inertia": "kg m^2",
         "motor_iq": "A", "motor_id": "A", "robot_current": "A"}
SOURCE = {"joint_position": "absolute encoder at the joint output", "motor_position": "motor encoder / gear ratio",
          "joint_velocity": "derived from the joint encoder", "motor_velocity": "derived from the motor encoder",
          "target_position": "controller (commanded)", "target_velocity": "controller (commanded)",
          "target_acceleration": "controller (commanded)", "target_torque": "controller (feed-forward)",
          "torque_sensor_a": "joint torque sensor A", "torque_sensor_b": "joint torque sensor B",
          "motor_torque": "drive, from the measured motor current", "computed_torque": "controller rigid-body model",
          "computed_inertia": "controller rigid-body model", "motor_iq": "drive current measurement (torque-forming)",
          "motor_id": "drive current measurement (magnetising)", "robot_current": "supply current of the whole arm"}


def quant_step(x, max_n=400000):
    """Smallest positive difference between distinct values. Floats that were never quantised give ~1e-9 or less."""
    u = np.unique(np.asarray(x, float)[:max_n])
    d = np.diff(u)
    d = d[d > 0]
    return float(d.min()) if len(d) else np.nan


def rest_mask(g, v_thr=REST_V):
    """Samples where this joint is not moving and not commanded to move."""
    return (np.abs(g["joint_velocity"].values) < v_thr) & (g["target_velocity"].values == 0)


def segments(mask, min_len=MIN_SEG):
    """Index arrays of contiguous True runs of at least min_len samples."""
    out, start = [], None
    m = np.asarray(mask, bool)
    for i, v in enumerate(np.r_[m, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append(np.arange(start, i))
            start = None
    return out


def psd_summary(cycles, signal, fs=100.0, nperseg=256):
    """Average Welch spectrum over cycles; returns f, P and the frequencies holding 95 / 99 % of the power."""
    Ps = []
    for g in cycles:
        x = g[signal].values
        x = x - x.mean()
        if len(x) < nperseg:
            continue
        f, P = welch(x, fs=fs, nperseg=nperseg)
        Ps.append(P)
    P = np.mean(Ps, axis=0)
    cum = np.cumsum(P) / P.sum()
    f95 = float(f[np.searchsorted(cum, 0.95)])
    f99 = float(f[np.searchsorted(cum, 0.99)])
    above40 = float(P[f >= 40].sum() / P.sum())
    return f, P, {"f95_hz": f95, "f99_hz": f99, "power_above_40hz": above40}


def sampling_stats(df):
    """Time step statistics and dropped samples, per the whole file."""
    dt = df.groupby("sample")["time"].diff().dropna().values
    med = float(np.median(dt))
    return {"dt_median_ms": med * 1e3, "dt_min_ms": float(dt.min()) * 1e3, "dt_max_ms": float(dt.max()) * 1e3,
            "dt_std_ms": float(dt.std()) * 1e3, "jitter_p99_ms": float(np.percentile(np.abs(dt - med), 99)) * 1e3,
            "gaps_over_1p5x": int((dt > 1.5 * med).sum()), "samples_per_cycle_mean": float(df.groupby("sample").size().mean()),
            "cycle_duration_mean_s": float(df.groupby("sample")["time"].max().mean())}
