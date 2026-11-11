"""Step 1: what is in the dataset, and does the friction model see the injected faults?

    python scripts/01_explore.py                 # all six axes, plots for axis 2
    python scripts/01_explore.py --axes 2 3      # only some axes
    python scripts/01_explore.py --main-axis 3   # plots for another axis

Writes results/01_explore/: overview.json, cycles_axisK.csv (one row per cycle, all features),
summary.md (tables per axis) and a few png figures.
"""
import os, sys, json, argparse, time
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin import dataset as D
from twin.features import build_cycle_table

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=D.DEFAULT_PATH)
ap.add_argument("--axes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
ap.add_argument("--main-axis", type=int, default=2)
ap.add_argument("--out", default=os.path.join(D.ROOT, "results", "01_explore"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
np.random.seed(0)          # only for the jitter in the scatter plots

GROUPS = ["normal", "axis_friction_own_A", "axis_friction_own_B", "axis_friction_own_C", "axis_friction_other_axis",
          "axis_weight_own", "axis_weight_other_axis", "can_weight", "collision",
          "motor_commutation_own", "motor_commutation_other_axis", "other"]
KEYS = ["load_coulomb", "load_c0", "load_viscous", "gear_coulomb", "gear_viscous", "track_err_rms", "sensor_ab_mean", "torque_rms_ac"]

# ---------------------------------------------------------------- overview
pf = pq.ParquetFile(args.data)
overview = {"file": os.path.basename(args.data), "rows": pf.metadata.num_rows, "columns": pf.metadata.num_columns}
tables = {}
summary = ["# Step 1 - dataset overview and per-cycle friction fits\n"]

for axis in args.axes:
    t0 = time.time()
    df = D.load_axis(axis, args.data)
    labels = D.cycle_labels(df)
    if axis == args.axes[0]:
        dt = df.groupby("sample")["time"].diff().median()
        overview.update(cycles=int(labels.shape[0]), sampling_hz=round(1 / dt), mean_duration_s=round(float(labels.duration.mean()), 2))
        overview["cycles_per_category"] = labels.groupby("category_name").size().to_dict()
        overview["variants"] = {c: sorted(g.variant.unique().tolist()) for c, g in labels.groupby("category_name")}
        print(f"rows {overview['rows']:,}  cycles {overview['cycles']}  {overview['sampling_hz']} Hz  mean cycle {overview['mean_duration_s']} s")
        print(labels.groupby("category_name").size().to_string())
    T = build_cycle_table(df, labels)
    T.to_csv(os.path.join(args.out, f"cycles_axis{axis}.csv"))
    tables[axis] = T
    # mean and std of the key numbers per group, for the summary file
    rows = []
    for gname in GROUPS:
        sub = T[T.group == gname]
        if len(sub) == 0:
            continue
        rows.append([gname, len(sub)] + [f"{sub[k].mean():.2f} ± {sub[k].std():.2f}" for k in KEYS])
    head = ["group", "cycles"] + KEYS
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)
    summary.append(f"\n## Axis {axis}\n\n" + md)
    # how far the own-axis friction cycles sit from normal, in units of the normal spread
    normal = T[T.group == "normal"]["load_coulomb"]
    for gname in ["axis_friction_own_A", "axis_friction_own_B", "axis_friction_own_C"]:
        sub = T[T.group == gname]["load_coulomb"]
        if len(sub):
            z = (sub.mean() - normal.mean()) / normal.std()
            summary.append(f"- {gname}: load_coulomb {sub.mean():.2f} vs normal {normal.mean():.2f} (std {normal.std():.2f}) -> {z:.1f} sigma\n")
    print(f"axis {axis}: {len(T)} cycles, {time.time() - t0:.0f} s")
    del df

json.dump(overview, open(os.path.join(args.out, "overview.json"), "w"), indent=1)
open(os.path.join(args.out, "summary.md"), "w").write("".join(summary))

# ---------------------------------------------------------------- figures (main axis)
ax_main = args.main_axis
if ax_main in tables:
    T = tables[ax_main]
    df = D.load_axis(ax_main, args.data)

    # 1. one normal cycle vs one own-axis friction cycle
    s_norm = T[T.group == "normal"].index[0]
    fr = T[T.group.str.startswith("axis_friction_own")]
    s_fric = fr.index[0] if len(fr) else None
    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for s, lab, c in [(s_norm, "normal", "k"), (s_fric, "added friction on this axis", "r")]:
        if s is None:
            continue
        g = df[df["sample"] == s]
        axs[0].plot(g.time, g.joint_position, c, label=lab)
        axs[1].plot(g.time, g.joint_velocity, c)
        axs[2].plot(g.time, g.torque_sensor_a, c, label=f"joint torque sensor ({lab})")
        axs[2].plot(g.time, g.computed_torque, c, ls="--", alpha=0.6, label=f"controller model ({lab})")
        axs[3].plot(g.time, g.torque_sensor_a - g.computed_torque, c)
    axs[0].set_ylabel("position [rad]"); axs[1].set_ylabel("velocity [rad/s]")
    axs[2].set_ylabel("torque [Nm]"); axs[3].set_ylabel("sensor - model [Nm]"); axs[3].set_xlabel("time [s]")
    axs[0].legend(); axs[2].legend(fontsize=8)
    axs[0].set_title(f"axis {ax_main}: one normal cycle and one cycle with added friction")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"traces_axis{ax_main}.png"), dpi=130); plt.close(fig)

    # 2. residual vs velocity: this is the picture behind the friction fit
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, (s, lab) in zip(axs, [(s_norm, "normal"), (s_fric, "added friction")]):
        if s is None:
            continue
        g = df[df["sample"] == s]
        a.scatter(g.joint_velocity, g.torque_sensor_a - g.computed_torque, s=3, alpha=0.4)
        a.set_title(f"axis {ax_main}, {lab}: load_coulomb = {T.loc[s, 'load_coulomb']:.2f} Nm")
        a.set_xlabel("joint velocity [rad/s]"); a.axvline(0, color="gray", lw=0.5)
    axs[0].set_ylabel("sensor - model torque [Nm]")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"residual_vs_velocity_axis{ax_main}.png"), dpi=130); plt.close(fig)

    # 3. session drift: gearbox viscous friction across the normal recording batches
    fig, a = plt.subplots(figsize=(8, 4))
    nv = T[T.group == "normal"]
    for i, (vname, sub) in enumerate(nv.groupby("variant")):
        a.scatter(np.full(len(sub), i) + np.random.uniform(-0.25, 0.25, len(sub)), sub.gear_viscous, s=6, alpha=0.5, label=f"{vname} ({len(sub)})")
    a.set_xticks(range(nv.variant.nunique())); a.set_xticklabels(sorted(nv.variant.unique()))
    a.set_ylabel("gear_viscous [Nm s/rad]"); a.set_title(f"axis {ax_main}: gearbox viscous friction per normal recording batch")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"session_drift_axis{ax_main}.png"), dpi=130); plt.close(fig)

# 4. the health indicator per cycle, all axes side by side
fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharey=False)
for a, axis in zip(axs.ravel(), sorted(tables)):
    T = tables[axis]
    groups = [g for g in GROUPS if (T.group == g).any()]
    for i, gname in enumerate(groups):
        y = T.loc[T.group == gname, "load_coulomb"]
        a.scatter(np.full(len(y), i) + np.random.uniform(-0.25, 0.25, len(y)), y, s=5, alpha=0.5)
    a.set_xticks(range(len(groups))); a.set_xticklabels([g.replace("axis_", "").replace("_other_axis", "\n(other axis)").replace("_own", "\n(this axis)") for g in groups], rotation=60, ha="right", fontsize=7)
    a.set_title(f"axis {axis}"); a.set_ylabel("load_coulomb [Nm]")
fig.suptitle("Coulomb friction on the load side, one dot per cycle")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "load_coulomb_all_axes.png"), dpi=130); plt.close(fig)

# 5. sensor A vs B disagreement, all axes
fig, axs = plt.subplots(1, len(tables), figsize=(3 * len(tables), 3.2), sharey=True)
for a, axis in zip(np.atleast_1d(axs), sorted(tables)):
    T = tables[axis]
    a.hist(T.sensor_ab_mean, bins=40, color="gray")
    a.set_title(f"axis {axis}"); a.set_xlabel("mean(sensor A - B) [Nm]")
fig.suptitle("two torque sensors per joint: how much they disagree per cycle")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "sensor_ab_all_axes.png"), dpi=130); plt.close(fig)

print("done ->", args.out)
