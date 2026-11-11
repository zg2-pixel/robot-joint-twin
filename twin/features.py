"""Per-cycle features. Mean levels are kept as separate columns so the classifier can't hide behind an offset."""
import numpy as np
import pandas as pd
from .physics import cycle_fits, residual_series, MOVING


def cycle_features(g):
    """One cycle in, one row of numbers out."""
    tq = g["motor_torque"].values
    ts = g["torque_sensor_a"].values
    err = (g["joint_position"] - g["target_position"]).values
    v = g["joint_velocity"].values
    moving = np.abs(v) > MOVING
    f = {
        "moving_frac": float(moving.mean()),
        "speed_peak": float(np.max(np.abs(v))),
        # torque levels: the mean is the DC part, rms is taken after removing it
        "torque_mean": float(tq.mean()),
        "torque_rms_ac": float(np.std(tq)),
        "torque_peak": float(np.max(np.abs(tq))),
        "sensor_mean": float(ts.mean()),
        "sensor_rms_ac": float(np.std(ts)),
        "iq_rms": float(np.sqrt(np.mean(g["motor_iq"].values ** 2))),
        # how well the joint follows the commanded path (mrad, the numbers are tiny in rad)
        "track_err_rms": float(np.sqrt(np.mean(err ** 2)) * 1e3),
        "track_err_max": float(np.max(np.abs(err)) * 1e3),
        # two torque sensors on the same joint: their disagreement is a sensor health check
        "sensor_ab_mean": float((g["torque_sensor_a"] - g["torque_sensor_b"]).mean()),
        "sensor_ab_std": float((g["torque_sensor_a"] - g["torque_sensor_b"]).std()),
        "robot_current_mean": float(g["robot_current"].mean()) if "robot_current" in g else np.nan,
    }
    f.update(cycle_fits(g))
    # transient part of the residual: a collision or a dropped can shows up as a short burst, not as a mean shift
    e = residual_series(g)
    roll = pd.Series(e ** 2).rolling(10, center=True).mean().pow(0.5).values
    f["resid_peak"] = float(np.nanmax(roll))
    f["resid_p99"] = float(np.percentile(np.abs(e), 99))
    return pd.Series(f)


def build_cycle_table(df, labels):
    """Features for every cycle, joined with the labels table from dataset.cycle_labels."""
    cols = ["time", "motor_torque", "torque_sensor_a", "torque_sensor_b", "computed_torque", "joint_position",
            "target_position", "joint_velocity", "target_acceleration", "motor_iq", "robot_current"]
    cols = [c for c in cols if c in df.columns]
    F = df.groupby("sample")[cols].apply(cycle_features)
    return labels.join(F)
