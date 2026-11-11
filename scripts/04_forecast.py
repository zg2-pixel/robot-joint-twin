"""Step 4: predict how many cycles are left before the friction crosses the limit, and check that the intervals are honest.

    python scripts/04_forecast.py                 # joint 2, 50 simulated degradation runs
    python scripts/04_forecast.py --runs 20       # quicker

The dataset has no slow wear, so the long runs come from the joint simulator (twin/joint_sim.py), calibrated to the
healthy cycles and checked against them. The forecaster (twin/forecast.py) never sees the true tau_c, only the
per-cycle estimates with their real-looking noise. Writes results/04_forecast/: sim_validation.json, runs.csv,
forecast_metrics.json, staircase_forecast.csv and figures.
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin import dataset as D
from twin.physics import fit_friction, cycle_design, residual_series
from twin.joint_sim import JointSimulator
from twin.forecast import TrendKF, WindowTrend, RestartingTrendKF, summarize_crossings

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=D.DEFAULT_PATH)
ap.add_argument("--axis", type=int, default=2)
ap.add_argument("--runs", type=int, default=50)
ap.add_argument("--cycles", type=int, default=300)
ap.add_argument("--tau-max", type=float, default=1.8, help="limit on tau_c [N m]; 1.8 is the milder injected friction level of joint 2")
ap.add_argument("--every", type=int, default=5, help="forecast every k-th cycle (speed)")
ap.add_argument("--q-slope", type=float, default=1e-4, help="process noise of the drift rate per cycle")
ap.add_argument("--warmup", type=int, default=60, help="healthy cycles before the drift starts (a deployed twin always has history)")
ap.add_argument("--model", default="restart", choices=["kf", "window", "restart"], help="trend model: plain Kalman filter, sliding-window regression, or Kalman filter restarted at the detected onset")
ap.add_argument("--window", type=int, default=40)
ap.add_argument("--k-onset", type=float, default=1.5)
ap.add_argument("--out", default=os.path.join(D.ROOT, "results", "04_forecast"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(1)
t0 = time.time()
STEP3 = os.path.join(D.ROOT, "results", "03_detect")
if not os.path.exists(os.path.join(STEP3, f"tracking_axis{args.axis}.csv")):
    sys.exit("run scripts/03_detect.py first, this step reads its tracking and staircase tables")

# ------------------------------------------------------------------ healthy cycles -> simulator parameters
df = D.load_axis(args.axis, args.data)
labels = D.cycle_labels(df)
pre_a = labels.index[labels.variant == "PRE_A"]
cycles = {s: g for s, g in df[df["sample"].isin(pre_a)].groupby("sample", sort=True)}
del df
ids = list(cycles)
fits = []
for s in ids[:300]:
    X, yl, yg = cycle_design(cycles[s])
    f = fit_friction(yl, X[:, 2], X[:, 3]); g = fit_friction(yg, X[:, 2], X[:, 3])
    e = residual_series(cycles[s])
    rho = np.corrcoef(e[:-1], e[1:])[0, 1]
    fits.append({"c0": f["c0"], "tc": f["coulomb"], "b": f["viscous"], "dJ": f["inertia"], "noise": f["noise"], "rho": rho,
                 "gc0": g["c0"], "gcoul": g["coulomb"], "gvisc": g["viscous"]})
fits = pd.DataFrame(fits)
params = (fits.c0.mean(), fits.tc.mean(), fits.b.mean())
obs_std = float(fits.tc.std())                    # the cycle-to-cycle spread of the estimate: the forecaster's observation noise
sim = JointSimulator([cycles[s] for s in ids[:300]], params, seed=0)
print(f"healthy joint {args.axis}: c0 {params[0]:.2f}, tau_c {params[1]:.2f} +- {obs_std:.3f}, b {params[2]:.2f}, residual {fits.noise.mean():.2f} (rho {fits.rho.mean():.2f})")

# ------------------------------------------------------------------ does the simulator look like the real joint?
def estimate(frame):
    X, yl, yg = cycle_design(frame)
    f = fit_friction(yl, X[:, 2], X[:, 3])
    return f["coulomb"], f["c0"], f["viscous"], f["noise"]

syn = pd.DataFrame([estimate(sim.cycle()) for _ in range(300)], columns=["tc", "c0", "b", "noise"])
real = fits[["tc", "c0", "b", "noise"]]
val = {k: {"real_mean": float(real[k].mean()), "real_std": float(real[k].std()), "sim_mean": float(syn[k].mean()), "sim_std": float(syn[k].std())} for k in real.columns}
inj = pd.DataFrame([estimate(sim.cycle(tau_c=params[1] + 1.2)) for _ in range(100)], columns=["tc", "c0", "b", "noise"])
val["injected_plus_1p2"] = {"tc_mean": float(inj.tc.mean()), "tc_std": float(inj.tc.std()), "expected": float(params[1] + 1.2)}
json.dump(val, open(os.path.join(args.out, "sim_validation.json"), "w"), indent=1)
print("simulator check (real vs synthetic healthy cycles):", {k: (round(v["real_mean"], 3), round(v["real_std"], 3), round(v["sim_mean"], 3), round(v["sim_std"], 3)) for k, v in val.items() if "real_mean" in v}, "| injected +1.2:", val["injected_plus_1p2"])

# ------------------------------------------------------------------ Kalman settings and a model check on the real healthy run
KF = dict(obs_std=obs_std, level_std0=0.3, slope_std0=0.01, q_level=0.01, q_slope=args.q_slope)
track = pd.read_csv(os.path.join(STEP3, f"tracking_axis{args.axis}.csv"))
kf = TrendKF(level0=track.tc.iloc[0], **KF)
innov = np.array([kf.step(y) for y in track.tc.values[1:]])
innov_std = float(np.nanstd(innov))
print(f"model check on the real healthy run: standardised innovations std {innov_std:.2f} (1.0 = noise level is right), final slope {kf.slope:+.5f} N m/cycle")

# ------------------------------------------------------------------ simulated degradation runs
rows = []
run_meta = []
example = None
for r in range(args.runs):
    rate = rng.uniform(0.004, 0.012)                       # N m per cycle -> crossing after ~100-300 cycles
    frames, truth = sim.degradation_run(args.cycles, rate=rate, walk_std=0.01, seed=100 + r)
    est = np.array([estimate(f)[0] for f in frames])
    crossing = int(np.argmax(truth >= args.tau_max)) if (truth >= args.tau_max).any() else None
    if args.model == "kf":
        kf = TrendKF(level0=est[0], **KF)
    elif args.model == "window":
        kf = WindowTrend(n=args.window, obs_std=obs_std, q_level=0.01)
    else:
        kf = RestartingTrendKF(level0=est[0], ref_mean=params[1], ref_std=obs_std, k_onset=args.k_onset, **KF)
    for _ in range(args.warmup):                               # history of healthy cycles before the drift begins
        kf.step(estimate(sim.cycle())[0])
    for k in range(args.cycles):
        kf.step(est[k])
        if crossing is None or k % args.every or k >= crossing:
            continue
        c = kf.cycles_to_limit(args.tau_max, horizon=800, n=1500, rng=rng)
        s = summarize_crossings(c, 800)
        remaining = crossing - k
        rows.append({"run": r, "cycle": k, "frac_life": k / crossing, "remaining": remaining, "median": s["median"], "lo5": s["lo5"], "hi95": s["hi95"],
                     "covered": s["lo5"] <= remaining <= s["hi95"], "rel_err": abs(s["median"] - remaining) / remaining,
                     "kf_level": kf.level, "kf_level_std": kf.level_std(), "kf_slope": kf.slope, "true_tc": truth[k], "est_tc": est[k], "p_within_50": s["p_within_50"]})
    run_meta.append({"run": r, "rate": rate, "crossing": crossing})
    if example is None and crossing is not None and 120 < crossing < 220:
        example = (r, frames, truth, est, crossing)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(args.out, "runs.csv"), index=False)
pd.DataFrame(run_meta).to_csv(os.path.join(args.out, "runs_meta.csv"), index=False)

# ------------------------------------------------------------------ metrics
def at(frac, tol=0.06):
    return R[(R.frac_life > frac - tol) & (R.frac_life < frac + tol)]
met = {"settings": {"tau_max": args.tau_max, "runs": args.runs, "cycles": args.cycles, "kf": KF, "every": args.every, "model": args.model, "window": args.window, "warmup": args.warmup},
       "crossed_runs": int(sum(m["crossing"] is not None for m in run_meta)),
       "innovation_std_healthy_run": innov_std,
       "coverage_90_overall": float(R[(R.frac_life > 0.2) & (R.frac_life < 0.95)].covered.mean()),
       "by_life_fraction": {}}
for frac in [0.25, 0.5, 0.75]:
    sub = at(frac)
    met["by_life_fraction"][str(frac)] = {"n": len(sub), "coverage_90": float(sub.covered.mean()), "median_rel_err": float(sub.rel_err.median()),
                                         "mean_interval_width_cycles": float((sub.hi95 - sub.lo5).mean()), "median_remaining": float(sub.remaining.median())}
# prognostic horizon: from which fraction of life onwards the interval keeps containing the truth
hz = []
for r, g in R.groupby("run"):
    g = g.sort_values("cycle")
    ok = g.covered.values[::-1]
    n_tail = 0
    for v in ok:
        if v: n_tail += 1
        else: break
    hz.append(1.0 if n_tail == 0 else g.frac_life.values[len(g) - n_tail])
met["prognostic_horizon_median_frac_life"] = float(np.median(hz))
json.dump(met, open(os.path.join(args.out, "forecast_metrics.json"), "w"), indent=1)
print("forecast:", json.dumps({k: met[k] for k in ["coverage_90_overall", "by_life_fraction", "prognostic_horizon_median_frac_life"]}))

# ------------------------------------------------------------------ the recorded staircase (healthy -> B -> A), forecast to the A level
stair = pd.read_csv(os.path.join(STEP3, f"staircase_axis{args.axis}.csv"))
is_A = stair.variant.str.match(r"^A\d_A$")                 # the stronger injected level (not the PRE_A healthy batch)
tau_A = float(stair.tc[is_A].mean()) if is_A.any() else 3.0
kf = TrendKF(level0=stair.tc.iloc[0], **KF)
srows = []
for k, y in enumerate(stair.tc.values):
    kf.step(y)
    c = kf.cycles_to_limit(tau_A, horizon=400, n=1500, rng=rng)
    s = summarize_crossings(c, 400)
    srows.append({"cycle": k, "variant": stair.variant.iloc[k], "tc": y, "kf_level": kf.level, "kf_slope": kf.slope, "median": s["median"], "lo5": s["lo5"], "hi95": s["hi95"], "p_within_20": s["p_within_20"]})
S = pd.DataFrame(srows); S.to_csv(os.path.join(args.out, f"staircase_forecast_axis{args.axis}.csv"), index=False)

# ------------------------------------------------------------------ figures
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for a, k, lab in zip(axs, ["tc", "c0", "noise"], ["tau_c [N m]", "c0 [N m]", "residual std [N m]"]):
    a.hist(real[k], bins=30, alpha=0.6, label="real healthy cycles"); a.hist(syn[k], bins=30, alpha=0.6, label="simulator")
    a.set_xlabel(lab); a.legend(fontsize=8)
fig.suptitle(f"joint {args.axis}: per-cycle estimates on real and simulated healthy cycles")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "sim_validation.png"), dpi=130); plt.close(fig)

if example is not None:
    r, frames, truth, est, crossing = example
    g = R[R.run == r]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(est, ".", ms=3, color="gray", label="per-cycle estimate")
    ax[0].plot(truth, "k", lw=1, label="true tau_c (simulator)")
    ax[0].plot(g.cycle, g.kf_level, "r", lw=1.2, label="Kalman level"); ax[0].fill_between(g.cycle, g.kf_level - 2 * g.kf_level_std, g.kf_level + 2 * g.kf_level_std, color="r", alpha=0.2)
    ax[0].axhline(args.tau_max, color="k", ls="--", lw=0.8); ax[0].text(2, args.tau_max + 0.03, "limit", fontsize=8)
    ax[0].axvline(crossing, color="k", ls=":", lw=0.8); ax[0].set_ylabel("tau_c [N m]"); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].plot(g.cycle, g.remaining, "k", lw=1, label="true cycles remaining")
    ax[1].plot(g.cycle, g["median"], "r", lw=1.2, label="forecast median"); ax[1].fill_between(g.cycle, g.lo5, g.hi95, color="r", alpha=0.2, label="90 % interval")
    ax[1].set_ylim(0, max(g.hi95.max(), crossing) * 1.05); ax[1].set_ylabel("cycles to limit"); ax[1].set_xlabel("cycle"); ax[1].legend(fontsize=8)
    fig.suptitle(f"one simulated degradation run (joint {args.axis} parameters, limit crossed at cycle {crossing})")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "run_example.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
bins = np.linspace(0.1, 1.0, 10)
R["bin"] = pd.cut(R.frac_life, bins)
cov = R.groupby("bin", observed=True).covered.mean(); err = R.groupby("bin", observed=True).rel_err.median()
centers = [b.mid for b in cov.index]
axs[0].plot(centers, cov.values, "o-"); axs[0].axhline(0.9, color="k", ls="--", lw=0.8); axs[0].set_ylim(0, 1.02)
axs[0].set_xlabel("fraction of life elapsed"); axs[0].set_ylabel("empirical coverage of the 90 % interval"); axs[0].set_title(f"{args.runs} simulated runs: is the interval honest?")
axs[1].plot(centers, err.values, "o-"); axs[1].set_xlabel("fraction of life elapsed"); axs[1].set_ylabel("median |error| / remaining cycles"); axs[1].set_title("how far off the median forecast is")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "coverage.png"), dpi=130); plt.close(fig)

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(S.cycle, S.tc, "o", ms=3, color="gray", label="per-cycle tau_c (recorded)"); ax[0].plot(S.cycle, S.kf_level, "r", lw=1.2, label="Kalman level")
ax[0].axhline(tau_A, color="k", ls="--", lw=0.8); ax[0].set_ylabel("tau_c [N m]"); ax[0].legend(fontsize=8)
ax[1].plot(S.cycle, S["median"], "r", lw=1.2, label="forecast median cycles to the A level"); ax[1].fill_between(S.cycle, S.lo5, S.hi95, color="r", alpha=0.2, label="90 % interval")
ax[1].set_ylim(0, 400); ax[1].set_ylabel("cycles to limit"); ax[1].set_xlabel("cycle"); ax[1].legend(fontsize=8)
fig.suptitle("recorded staircase (healthy -> B -> A): the forecaster reacts to the step at cycle 40")
fig.tight_layout(); fig.savefig(os.path.join(args.out, f"staircase_forecast_axis{args.axis}.png"), dpi=130); plt.close(fig)
print("done ->", args.out, f"({time.time()-t0:.0f} s)")
