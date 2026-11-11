"""Step 2: how good are the signals? Sampling, resolution, noise floors, biases, batch-to-batch drift, and what all
of that does to the friction estimate. These are the numbers for the uncertainty budget table.

    python scripts/02_uncertainty.py                  # all six axes, figures for axis 2
    python scripts/02_uncertainty.py --axes 2 3       # a subset

Writes results/02_uncertainty/: sampling.json, budget_axisK.csv (one row per signal), bias_axisK.csv,
batches_axisK.csv, propagation.json, summary.md and figures.
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin import dataset as D
from twin import uncertainty as U
from twin.features import build_cycle_table

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=D.DEFAULT_PATH)
ap.add_argument("--axes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
ap.add_argument("--main-axis", type=int, default=2)
ap.add_argument("--out", default=os.path.join(D.ROOT, "results", "02_uncertainty"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
np.random.seed(0)          # only for the jitter in the scatter plots

SIGNALS = ["joint_position", "joint_velocity", "motor_position", "motor_velocity", "torque_sensor_a", "torque_sensor_b",
           "motor_torque", "motor_iq", "motor_id", "computed_torque", "target_position", "target_velocity",
           "target_acceleration", "robot_current"]
PSD_SIGNALS = ["torque_sensor_a", "motor_torque", "motor_iq", "joint_velocity"]


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


summary = ["# Step 2 - signal quality and uncertainty budget\n"]
prop_all = {}
psd_all = {}

for axis in args.axes:
    t0 = time.time()
    df = D.load_axis(axis, args.data)
    labels = D.cycle_labels(df)
    healthy_ids = labels.index[labels.category_name == "normal"]
    H = df[df["sample"].isin(healthy_ids)]
    cycles = [g for _, g in H.groupby("sample", sort=True)]

    if axis == args.axes[0]:
        samp = U.sampling_stats(df)
        json.dump(samp, open(os.path.join(args.out, "sampling.json"), "w"), indent=1)
        summary.append("## Sampling (whole file)\n\n" + "\n".join(f"- {k}: {v:.4g}" for k, v in samp.items()) + "\n")
        print("sampling:", {k: round(v, 3) for k, v in samp.items()})

    # ---------------- per-signal budget: resolution + noise at rest
    rest_var = {s: [] for s in SIGNALS}
    diff_var = {s: [] for s in SIGNALS}
    ab_rest, sm_rest, pe_rest, ms_rest = [], [], [], []       # A-B, sensor-model, position error, motor-sensor at rest
    ab_rest_std, pe_move = [], []
    ab_pts = []                                               # (torque level, A-B) per rest segment, for the gain check
    for g in cycles:
        segs = U.segments(U.rest_mask(g))
        if not segs:
            continue
        for s in SIGNALS:
            x = g[s].values
            for idx in segs:
                xs = x[idx]
                rest_var[s].append(np.var(xs - xs.mean()))
                diff_var[s].append(np.var(np.diff(xs)) / 2)
        idx = np.concatenate(segs)
        ab = g["torque_sensor_a"].values - g["torque_sensor_b"].values
        ab_rest.append(ab[idx].mean()); ab_rest_std.append(ab[idx].std())
        for sg in segs:
            ab_pts.append((g["torque_sensor_a"].values[sg].mean(), ab[sg].mean()))
        sm_rest.append((g["torque_sensor_a"].values - g["computed_torque"].values)[idx].mean())
        ms_rest.append((g["motor_torque"].values - g["torque_sensor_a"].values)[idx].mean())
        pe = (g["joint_position"].values - g["target_position"].values)
        pe_rest.append(pe[idx].mean())
        mv = ~U.rest_mask(g)
        pe_move.append(np.sqrt(np.mean(pe[mv] ** 2)) if mv.any() else np.nan)
    rows = []
    for s in SIGNALS:
        x = H[s].values
        step = U.quant_step(x)
        rng = float(x.max() - x.min())
        bits = np.log2(rng / step) if step > 1e-9 and rng > 0 else np.nan
        rows.append({"signal": s, "units": U.UNITS[s], "source": U.SOURCE[s], "rate_hz": 100,
                     "range_min": float(x.min()), "range_max": float(x.max()),
                     "quant_step": step if step > 1e-9 else 0.0, "effective_bits": bits,
                     "sigma_rest": float(np.sqrt(np.mean(rest_var[s]))), "sigma_white": float(np.sqrt(np.mean(diff_var[s])))})
    budget = pd.DataFrame(rows)
    budget.to_csv(os.path.join(args.out, f"budget_axis{axis}.csv"), index=False)

    # ---------------- systematic errors: things that should agree but don't
    ab_rest = np.array(ab_rest); sm_rest = np.array(sm_rest); pe_rest = np.array(pe_rest); ms_rest = np.array(ms_rest)
    # does the A-B disagreement grow with the torque level? then it is a gain mismatch, not just an offset
    ab_pts = np.array(ab_pts)
    gain_slope, gain_icpt = np.polyfit(ab_pts[:, 0], ab_pts[:, 1], 1)
    ab_after_gain = ab_pts[:, 1] - (gain_icpt + gain_slope * ab_pts[:, 0])
    if np.ptp(ab_pts[:, 0]) < 10:        # a few N m of range cannot pin down a gain, don't pretend it can
        gain_slope = np.nan
    bias = pd.DataFrame([
        {"comparison": "torque sensor A - B, at rest", "units": "N m", "bias_mean": ab_rest.mean(), "cycle_to_cycle_std": ab_rest.std(),
         "within_cycle_std": float(np.mean(ab_rest_std)), "note": "two sensors on the same joint; bias = calibration offset, spread = noise"},
        {"comparison": "A - B vs torque level: slope (gain mismatch)", "units": "N m per N m", "bias_mean": float(gain_slope), "cycle_to_cycle_std": np.nan,
         "within_cycle_std": float(ab_after_gain.std()), "note": f"intercept {gain_icpt:.3f} N m; within_cycle_std here = spread left after removing offset and gain"},
        {"comparison": "torque sensor A - controller model, at rest", "units": "N m", "bias_mean": sm_rest.mean(), "cycle_to_cycle_std": sm_rest.std(),
         "within_cycle_std": np.nan, "note": "gravity model error of the controller, absorbed into c0"},
        {"comparison": "motor torque - torque sensor A, at rest", "units": "N m", "bias_mean": ms_rest.mean(), "cycle_to_cycle_std": ms_rest.std(),
         "within_cycle_std": np.nan, "note": "holding torque lost in the gearbox at standstill"},
        {"comparison": "joint position - commanded, at rest", "units": "rad", "bias_mean": pe_rest.mean(), "cycle_to_cycle_std": pe_rest.std(),
         "within_cycle_std": np.nan, "note": "controller steady-state error, not a sensor error"},
        {"comparison": "joint position - commanded, while moving (rms)", "units": "rad", "bias_mean": float(np.nanmean(pe_move)), "cycle_to_cycle_std": float(np.nanstd(pe_move)),
         "within_cycle_std": np.nan, "note": "tracking lag of the controller"},
    ])
    bias.to_csv(os.path.join(args.out, f"bias_axis{axis}.csv"), index=False)

    # ---------------- batch-to-batch drift of the healthy state (per-cycle fits)
    T = build_cycle_table(H, labels.loc[healthy_ids])
    keys = ["load_coulomb", "load_c0", "load_viscous", "gear_coulomb", "gear_viscous", "sensor_ab_mean", "torque_mean", "duration"]
    by = T.groupby("variant")[keys]
    batches = by.mean().round(4)
    batches["cycles"] = by.size()
    within = T.groupby("variant")[keys].std().mean()          # typical spread inside a batch
    between = T.groupby("variant")[keys].mean().std()         # spread of the batch means
    drift = pd.DataFrame({"within_batch_std": within, "between_batch_std": between, "ratio_between_over_within": between / within})
    batches.to_csv(os.path.join(args.out, f"batches_axis{axis}.csv"))
    drift.to_csv(os.path.join(args.out, f"drift_axis{axis}.csv"))

    # ---------------- propagation: sensor noise -> friction estimate
    se_white = T["load_se_coulomb"].mean()
    se_corr = T["load_se_coulomb_corr"].mean()
    rho = T["load_resid_rho1"].mean()
    emp_within = float(T.groupby("variant")["load_coulomb"].std().mean())
    emp_all = float(T["load_coulomb"].std())
    n_mov = float(T["load_n_moving"].mean())
    lsb = budget.loc[budget.signal == "torque_sensor_a", "quant_step"].item()
    prop = {"fit_residual_std_Nm": float(T["load_noise"].mean()), "n_moving_samples": n_mov,
            "se_coulomb_white_Nm": float(se_white), "se_coulomb_corrected_Nm": float(se_corr), "residual_lag1_corr": float(rho),
            "empirical_std_within_batch_Nm": emp_within, "empirical_std_all_healthy_Nm": emp_all,
            "quantization_contribution_Nm": float(lsb / np.sqrt(12) / np.sqrt(n_mov)),
            "ratio_empirical_over_corrected_se": emp_within / se_corr if se_corr > 0 else np.nan}
    prop_all[axis] = prop

    # ---------------- spectra
    psd = {}
    spectra = {}
    for s in PSD_SIGNALS:
        f, P, info = U.psd_summary(cycles[:300], s)
        psd[s] = info; spectra[s] = (f, P)
    psd_all[axis] = psd

    # ---------------- summary text
    summary.append(f"\n## Axis {axis}\n\n### Resolution and noise floor per signal (healthy cycles, at rest)\n\n" +
                   md_table(budget[["signal", "units", "quant_step", "effective_bits", "sigma_rest", "sigma_white"]]))
    summary.append("### Systematic errors\n\n" + md_table(bias[["comparison", "units", "bias_mean", "cycle_to_cycle_std", "within_cycle_std"]]))
    summary.append("### Healthy state per recording batch\n\n" + md_table(batches.reset_index()))
    summary.append("### Drift: between-batch vs within-batch spread\n\n" + md_table(drift.reset_index().rename(columns={"index": "quantity"})))
    summary.append("### Propagation to the Coulomb friction estimate\n\n" + "\n".join(f"- {k}: {v:.4g}" for k, v in prop.items()) + "\n")
    summary.append("### Spectral content (healthy cycles)\n\n" + md_table(pd.DataFrame(psd).T.reset_index().rename(columns={"index": "signal"})))
    print(f"axis {axis}: {len(cycles)} healthy cycles, se_corr {se_corr:.3f} vs empirical {emp_within:.3f}, {time.time()-t0:.0f} s")

    # ---------------- figures for the main axis
    if axis == args.main_axis:
        # spectra
        fig, ax = plt.subplots(figsize=(8, 4))
        for s in PSD_SIGNALS:
            f, P = spectra[s]
            ax.semilogy(f, P / P.max(), label=f"{s} (99 % below {psd[s]['f99_hz']:.0f} Hz)")
        ax.axvline(50, color="gray", ls="--", lw=0.8); ax.text(50, 0.5, " Nyquist", fontsize=8, color="gray")
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("power (normalised)"); ax.set_title(f"axis {axis}: where the signal power sits (healthy cycles)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(args.out, f"psd_axis{axis}.png"), dpi=130); plt.close(fig)

        # one rest segment, zoomed, for the three torques
        g = cycles[0]; segs = U.segments(U.rest_mask(g))
        idx = max(segs, key=len)
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        t = g["time"].values[idx]
        for a, s in zip(ax, ["torque_sensor_a", "torque_sensor_b", "motor_torque"]):
            x = g[s].values[idx]
            a.plot(t, x - x.mean(), lw=0.8); a.set_ylabel(f"{s}\n- mean [N m]", fontsize=8)
            a.set_title(f"std {np.std(x):.3f} N m", fontsize=9, loc="right")
        ax[-1].set_xlabel("time [s]"); ax[0].set_title(f"axis {axis}: noise while the joint stands still (one rest segment)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"rest_noise_axis{axis}.png"), dpi=130); plt.close(fig)

        # batch drift of four quantities
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
        for a, k in zip(axs, ["load_coulomb", "gear_viscous", "sensor_ab_mean", "load_c0"]):
            for i, (vname, sub) in enumerate(T.groupby("variant")):
                a.scatter(np.full(len(sub), i) + np.random.uniform(-0.25, 0.25, len(sub)), sub[k], s=4, alpha=0.4)
            a.set_xticks(range(T.variant.nunique())); a.set_xticklabels(sorted(T.variant.unique()), rotation=45, fontsize=7); a.set_title(k, fontsize=9)
        fig.suptitle(f"axis {axis}: healthy cycles by recording batch - what drifts and what does not")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"batch_drift_axis{axis}.png"), dpi=130); plt.close(fig)

        # the two torque sensors against each other: offset + gain
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(ab_pts[:, 0], ab_pts[:, 1], s=4, alpha=0.3, label="one rest segment each")
        xx = np.linspace(ab_pts[:, 0].min(), ab_pts[:, 0].max(), 10)
        ax.plot(xx, gain_icpt + gain_slope * xx, "r", label=f"A - B = {gain_icpt:.2f} + {gain_slope:.4f} x A")
        ax.set_xlabel("torque sensor A [N m]"); ax.set_ylabel("A - B [N m]"); ax.legend(fontsize=8)
        ax.set_title(f"axis {axis}: disagreement of the two torque sensors vs torque level (at rest)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"sensor_ab_gain_axis{axis}.png"), dpi=130); plt.close(fig)

        # propagation: fit standard error vs the spread actually seen
        fig, ax = plt.subplots(figsize=(7, 4))
        big = T[T.variant == T.variant.value_counts().idxmax()]
        ax.hist(big["load_coulomb"], bins=40, color="gray", alpha=0.7, label=f"healthy cycles, batch {big.variant.iloc[0]} (std {big.load_coulomb.std():.3f})")
        m = big["load_coulomb"].mean()
        ax.axvspan(m - se_white, m + se_white, color="C3", alpha=0.6, label=f"± fit s.e., white noise ({se_white:.3f})")
        ax.axvspan(m - se_corr, m + se_corr, color="C1", alpha=0.3, label=f"± fit s.e., correlated residual ({se_corr:.3f})")
        ax.set_xlabel("load_coulomb [N m]"); ax.set_ylabel("cycles"); ax.legend(fontsize=8)
        ax.set_title(f"axis {axis}: per-cycle fit uncertainty vs cycle-to-cycle spread")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"propagation_axis{axis}.png"), dpi=130); plt.close(fig)
    del df, H

json.dump({"propagation": prop_all, "psd": psd_all}, open(os.path.join(args.out, "propagation.json"), "w"), indent=1)
open(os.path.join(args.out, "summary.md"), "w").write("".join(summary))
print("done ->", args.out)
