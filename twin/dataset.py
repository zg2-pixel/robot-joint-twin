"""Load the voraus-AD parquet one axis at a time (the full file is 137 columns, we only need ~20)."""
import os
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "data", "voraus-ad-dataset-100hz.parquet")

META = ["time", "sample", "anomaly", "category", "setting", "action", "active"]
# per-axis signals, the file has these for axes 1..6 with a _k suffix
AXIS_SIGNALS = ["target_position", "target_velocity", "target_acceleration", "target_torque",
                "computed_torque", "computed_inertia", "motor_position", "motor_velocity",
                "joint_position", "joint_velocity", "motor_torque", "torque_sensor_a", "torque_sensor_b",
                "motor_iq", "motor_id"]

# category codes from the official voraus_ad.py
CATEGORY = {0: "axis_friction", 1: "axis_weight", 2: "collision_foam", 3: "collision_cable",
            4: "collision_carton", 5: "miss_can", 6: "lose_can", 7: "can_weight", 8: "entangled",
            9: "invalid_position", 10: "motor_commutation", 11: "wobbling_station", 12: "normal"}

# variant codes (the 'setting' column), also from voraus_ad.py. A2_B = axis 2, level B, etc.
VARIANT = {0: "A1_A", 1: "A1_B", 2: "A2_A", 3: "A2_B", 4: "A3_A", 5: "A3_B", 6: "A3_C", 7: "A4_A", 8: "A4_B",
           9: "A4_C", 10: "A5_A", 11: "A5_B", 12: "A6_A", 13: "A6_B",
           14: "A1_115G", 15: "A1_231G", 16: "A1_500G", 17: "A2_500G", 18: "A2_231G", 19: "A2_115G",
           20: "A3_115G", 21: "A3_231G", 22: "A3_500G", 23: "A4_500G", 24: "A4_231G", 25: "A4_115G",
           26: "A5_115G", 27: "A5_231G", 28: "A5_500G", 29: "A6_500G", 30: "A6_231G", 31: "A6_115G",
           32: "LARGE_FULL", 33: "SMALL_FULL", 34: "SMALL_STRIPE", 35: "SMALL_HANGING_FULL",
           36: "LARGE_HANGING_FULL", 37: "SMALL_HANGING_STRIPE", 38: "CONNECTOR_TOP", 39: "CONNECTOR_SIDE",
           40: "LAN", 41: "STRIPE", 42: "FULL_A", 43: "FULL_B", 44: "CONVEYOR", 45: "NO_VACUUM",
           46: "DEFECT_GRIPPER_DROP_CAN", 47: "DROP_1400MS", 48: "DROP_1100MS", 49: "DROP_100MS",
           50: "DROP_200MS", 51: "LOSE_200MS", 52: "LOSE_300MS", 53: "LOSE_PLACE_100MS",
           54: "DECREASE_95G", 55: "DECREASE_165G", 56: "EMPTY", 57: "INCREASE_242G", 58: "INCREASE_373G",
           59: "INCREASE_407G", 60: "CABLE", 61: "PICK", 62: "A1_HIGH", 63: "A1_MEDIUM", 64: "A2_MEDIUM",
           65: "A3_MEDIUM", 66: "A4_MEDIUM", 67: "A5_MEDIUM", 68: "A6_MEDIUM", 69: "LOW_DAMPING",
           70: "MEDIUM_DAMPING", 71: "HIGH_DAMPING", 72: "PRE_A", 73: "PRE_B", 74: "BETWEEN_A",
           75: "BETWEEN_B", 76: "BETWEEN_C"}


def variant_axis(setting):
    """Which axis an anomaly variant targets (A3_B -> 3), or None if it is not axis specific."""
    name = VARIANT.get(int(setting), "")
    if len(name) > 2 and name[0] == "A" and name[1].isdigit() and name[2] == "_":
        return int(name[1])
    return None


def load_axis(axis, path=DEFAULT_PATH, extra=("robot_current",)):
    """Read the meta columns plus the signals of one axis. Column names come back without the _k suffix."""
    cols = META + list(extra) + [f"{s}_{axis}" for s in AXIS_SIGNALS]
    df = pq.read_table(path, columns=cols).to_pandas()
    df = df.rename(columns={f"{s}_{axis}": s for s in AXIS_SIGNALS})
    df.attrs["axis"] = axis
    return df


def cycle_labels(df):
    """One row per cycle: what the cycle is, and whether the anomaly is on this axis."""
    axis = df.attrs.get("axis")
    g = df.groupby("sample")
    t = g.agg(category=("category", "first"), setting=("setting", "first"), anomaly=("anomaly", "first"),
              duration=("time", "max"), n=("time", "size"))
    t["category_name"] = t["category"].map(CATEGORY)
    t["variant"] = t["setting"].map(lambda s: VARIANT.get(int(s), str(s)))
    t["variant_axis"] = t["setting"].map(variant_axis)
    # label used in the plots: own-axis anomalies get their own name, the rest is grouped
    def group(r):
        c = r["category_name"]
        if c == "normal":
            return "normal"
        if c in ("axis_friction", "axis_weight", "motor_commutation"):
            if r["variant_axis"] == axis:
                return c + "_own" + ("_" + r["variant"].split("_")[1] if c == "axis_friction" else "")
            return c + "_other_axis"
        if c.startswith("collision"):
            return "collision"
        if c == "can_weight":
            return "can_weight"
        return "other"
    t["group"] = t.apply(group, axis=1)
    return t

