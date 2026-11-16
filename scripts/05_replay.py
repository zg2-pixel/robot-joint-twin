"""Step 5: the whole twin as an OODA loop, replayed on recorded and simulated cycles, against two baselines.

    python scripts/05_replay.py                 # joint 2
    python scripts/05_replay.py --runs 5        # fewer degradation runs (faster)

Scenarios (from the proposal):
  S1 held-out healthy and injected-fault cycles      S4 sensor offset on torque sensor A
  S2 degradation runs on the simulator               S5 an anomaly type the classifier never saw
  S3 task change without a fault (payload)           S6 sweep over drift rate and extra noise
Baselines: B1 fixed threshold on the torque RMS (the usual alarm), B2 the classifier alone (no physics tracking,
no forecast, no hysteresis). Writes results/05_replay/: one log per scenario, summary.json, figures.
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin import dataset as D
from twin.features import cycle_features
from twin.physics import cycle_design, fit_friction
from twin import detect as DT
from twin.joint_sim import JointSimulator
from twin.loop import OODATwin
from twin.forecast import RestartingTrendKF, summarize_crossings
from twin.decide import DEFAULT_COSTS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=D.DEFAULT_PATH)
ap.add_argument("--axis", type=int, default=2)
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--tau-max", type=float, default=1.8)
ap.add_argument("--out", default=os.path.join(D.ROOT, "results", "05_replay"))
ap.add_argument("--step3", default=os.path.join(D.ROOT, "results", "03_detect"),
                help="directory containing step-3 results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(0)
t0 = time.time()
C = DEFAULT_COSTS
STEP3 = args.step3
if not os.path.exists(os.path.join(STEP3, f"cycles_axis{args.axis}.csv")):
    sys.exit("run scripts/03_detect.py first, this step reads its cycle table and classifier results")

# ------------------------------------------------------------------ data, split, classifier
df = D.load_axis(args.axis, args.data)
frames = {s: g for s, g in df.groupby("sample", sort=True)}
del df
T = pd.read_csv(os.path.join(STEP3, f"cycles_axis{args.axis}.csv"), index_col=0)      # the per-cycle table step 3 wrote
T["label"] = DT.label_for_joint(T, args.axis)
lab = T[T.label.isin(DT.CLASSES)]
train_idx, test_idx = train_test_split(lab.index, test_size=0.3, stratify=lab.label, random_state=0)
clf = DT.make_classifier("nb").fit(lab.loc[train_idx, DT.FEATURES], lab.loc[train_idx, "label"].to_numpy(dtype=object))
healthy_train = T.index[(T.category_name == "normal") & (T.index.isin(train_idx))]
pre_a_train = [s for s in healthy_train if T.loc[s, "variant"] == "PRE_A"]
ref_table = T.loc[pre_a_train[:300]]
warm_ids = pre_a_train[300:360]                                   # 60 healthy cycles to warm the tracker up
b1_thr = float(ref_table.torque_rms_ac.mean() + 3 * ref_table.torque_rms_ac.std())
print(f"classifier trained on {len(train_idx)} cycles, {len(test_idx)} held out; B1 threshold on torque rms = {b1_thr:.2f} N m")


cv = json.load(open(os.path.join(STEP3, f"classifier_axis{args.axis}.json")))
trusted = tuple(c for c in ["payload", "collision", "electrical"] if cv["nb"]["recall"].get(c, 0) >= 0.5)
print("classes the decision acts on (cross-validated recall >= 0.5):", trusted)


def new_twin():
    return OODATwin(ref_table, clf, tau_max=args.tau_max, seed=int(rng.integers(1e9)), trusted=trusted)


def baselines(rec_frame_features):
    """B1 and B2 verdicts for one cycle from its feature row."""
    f = rec_frame_features
    b1 = f["torque_rms_ac"] > b1_thr
    p = dict(zip(clf.classes_, clf.predict_proba(pd.DataFrame([f[DT.FEATURES]]))[0]))
    b2 = (1 - p.get("healthy", 0)) > 0.5
    return bool(b1), bool(b2)


def replay(seq, truth_fn=None, sim=None, sim_kwargs=None):
    """Run a twin over a list of cycle ids (recorded) or of simulator parameter dicts (sim)."""
    tw = new_twin()
    out = []
    for item in seq:
        if sim is None:
            fr = frames[item]
            truth = {"sample": item, "label": T.loc[item, "label"], "variant": T.loc[item, "variant"]}
        else:
            kw = dict(sim_kwargs or {}); kw.update({k: v for k, v in item.items() if not k.startswith("true_")})
            fr = sim.cycle(speed=tw.speed, **kw)
            truth = {k: v for k, v in item.items() if k.startswith("true_")}
        applied = tw.speed if sim is not None else 1.0
        rec = tw.step(fr, truth=truth, speed_applied=applied)
        f = cycle_features(fr); f["duration"] = float(fr["time"].max()) * applied
        rec["b1_stop"], rec["b2_stop"] = baselines(f)
        rec["torque_rms_ac"] = f["torque_rms_ac"]
        out.append(rec)
    return pd.DataFrame(out)

summary = {"costs": C, "b1_threshold": b1_thr, "tau_max": args.tau_max, "trusted_classes": list(trusted)}

# the classifier on the held-out cycles (fitted and calibrated on the training cycles only)
P_test = pd.DataFrame(clf.predict_proba(lab.loc[test_idx, DT.FEATURES]), columns=clf.classes_, index=test_idx)
m = DT.summarize(P_test, lab.loc[test_idx, "label"].to_numpy(dtype=object))
summary["classifier_heldout"] = {"n": len(test_idx), "accuracy": m["accuracy"], "recall": m["recall"], "brier": m["brier"]}
print("classifier on held-out cycles:", json.dumps(summary["classifier_heldout"]))
warm = list(warm_ids)

# ------------------------------------------------------------------ S1: held-out cycles, class by class
s1 = {}
logs = []
for c in DT.CLASSES:
    ids = [s for s in test_idx if T.loc[s, "label"] == c]
    if not ids:
        continue
    L = replay(warm + ids)
    L["scenario"] = f"S1_{c}"; logs.append(L)
    part = L.iloc[len(warm):]
    s1[c] = {"n": len(ids), "actions": part.action.value_counts(normalize=True).round(3).to_dict(),
             "mean_p_fault": float(part.p_fault.mean()), "b1_stop_rate": float(part.b1_stop.mean()), "b2_stop_rate": float(part.b2_stop.mean())}
    if c == "healthy":
        s1[c]["unnecessary_stops_per_100"] = float(100 * (part.action == "stop").mean())
summary["S1_heldout"] = s1
print("S1:", json.dumps({c: (v["actions"], round(v["b1_stop_rate"], 2), round(v["b2_stop_rate"], 2)) for c, v in s1.items()}))

# ------------------------------------------------------------------ S2: degradation runs on the simulator
fits = []
for s in pre_a_train[:300]:
    X, yl, yg = cycle_design(frames[s]); f = fit_friction(yl, X[:, 2], X[:, 3]); fits.append((f["c0"], f["coulomb"], f["viscous"]))
healthy_params = tuple(np.mean(fits, axis=0))
sim = JointSimulator([frames[s] for s in pre_a_train[:300]], healthy_params, seed=1)


def cost_of_run(L, crossing):
    """What a run cost under the twin's decisions and under each baseline. Stopping or servicing early also
    throws away useful life (waste per cycle), which is what keeps 'stop at the first doubt' from looking free."""
    res = {}
    sched = L.index[L.action == "schedule"]
    stop = L.index[L.action == "stop"]
    slow_cycles = int((L.action == "slow").sum())
    cost = slow_cycles * C["slow"]
    first_stop = int(stop[0]) if len(stop) else None
    planned_at = None
    if len(sched):
        k = int(sched[0]); planned_at = k + int(L.loc[k, "t_plan"]); res["twin_first_schedule"] = k; res["twin_planned_at"] = planned_at
    if crossing is None:                                        # the joint never reached the limit in this run
        if first_stop is not None:
            cost += C["stop"]; outcome = "unnecessary stop"
        elif planned_at is not None and planned_at < len(L):
            cost += C["sched"]; outcome = "unnecessary maintenance"
        else:
            outcome = "no failure, kept running"
        res.update({"twin_cost": cost, "twin_outcome": outcome, "twin_slow_cycles": slow_cycles})
        for b in ["b1", "b2"]:
            trips = L.index[L[f"{b}_stop"]]
            res[f"{b}_cost"] = C["stop"] if len(trips) else 0.0
            res[f"{b}_outcome"] = "unnecessary stop" if len(trips) else "no failure, kept running"
        return res
    if first_stop is not None and first_stop < crossing and (planned_at is None or first_stop < planned_at):
        cost += C["stop"] + (crossing - first_stop) * C["waste"]; res["twin_first_stop"] = first_stop
        outcome = "early stop, ticket open" if planned_at is not None else "unplanned stop"
    elif planned_at is not None and planned_at <= crossing:
        cost += C["sched"] + (crossing - planned_at) * C["waste"]; outcome = "planned in time"
    elif planned_at is not None:
        cost += C["missed"]; outcome = "planned too late"
    else:
        cost += C["missed"]; outcome = "missed"
    res.update({"twin_cost": cost, "twin_outcome": outcome, "twin_slow_cycles": slow_cycles})
    for b in ["b1", "b2"]:
        trips = L.index[L[f"{b}_stop"]]
        if len(trips) and trips[0] < crossing:
            res[f"{b}_cost"] = C["stop"] + (crossing - int(trips[0])) * C["waste"]; res[f"{b}_outcome"] = "unplanned stop"; res[f"{b}_trip"] = int(trips[0])
        else:
            res[f"{b}_cost"] = C["missed"]; res[f"{b}_outcome"] = "missed"
    return res

s2 = []
example = None
for r in range(args.runs):
    rate = rng.uniform(0.004, 0.012)
    path = sim.degradation_path(300, rate=rate, walk_std=0.01, seed=200 + r)
    crossing = int(np.argmax(path >= args.tau_max)) if (path >= args.tau_max).any() else None
    seq = [{"tau_c": healthy_params[1], "true_tc": healthy_params[1]} for _ in range(len(warm))] + [{"tau_c": v, "true_tc": v} for v in path]
    L = replay(seq, sim=sim)
    L = L.iloc[len(warm):].reset_index(drop=True)
    L["cycle"] = np.arange(len(L))                              # count from the start of the degradation, not of the warm-up
    L["scenario"] = f"S2_run{r}"; logs.append(L)
    res = cost_of_run(L, crossing); res.update({"run": r, "rate": rate, "crossing": crossing}); s2.append(res)
    if example is None and crossing is not None and 120 < crossing < 220:
        example = (r, L, crossing)
S2 = pd.DataFrame(s2); S2.to_csv(os.path.join(args.out, "S2_runs.csv"), index=False)
summary["S2_degradation"] = {"runs": args.runs, "mean_cost": {k: float(S2[f"{k}_cost"].mean()) for k in ["twin", "b1", "b2"]},
                             "outcomes": {k: S2[f"{k}_outcome"].value_counts().to_dict() for k in ["twin", "b1", "b2"]},
                             "twin_mean_slow_cycles": float(S2.twin_slow_cycles.mean()),
                             "planned_margin_cycles": float((S2.crossing - S2.twin_planned_at).dropna().mean()) if "twin_planned_at" in S2 else None}
print("S2:", json.dumps(summary["S2_degradation"]))

# ------------------------------------------------------------------ S3: task change (payload), S5: unseen anomaly
for name, c in [("S3_payload", "payload"), ("S5_unseen_wobble", None)]:
    ids = [s for s in test_idx if T.loc[s, "label"] == c] if c else T.index[T.category_name == "wobbling_station"].tolist()
    L = replay(warm + ids); L["scenario"] = name; logs.append(L)
    part = L.iloc[len(warm):]
    summary[name] = {"n": len(ids), "actions": part.action.value_counts(normalize=True).round(3).to_dict(), "mean_p_fault": float(part.p_fault.mean()),
                     "unknown_flag_rate": float(part.unknown_flag.mean()), "b1_stop_rate": float(part.b1_stop.mean()), "b2_stop_rate": float(part.b2_stop.mean())}
    print(name, json.dumps(summary[name]))

# ------------------------------------------------------------------ S4: sensor offset on torque sensor A
seq = [{"sensor_offset": 0.0} for _ in range(len(warm))] + [{"sensor_offset": 0.5} for _ in range(40)]
L = replay(seq, sim=sim); L["scenario"] = "S4_sensor_offset"; logs.append(L)
part = L.iloc[len(warm):]
summary["S4_sensor_offset"] = {"n": 40, "abstain_rate": float((part.action == "abstain").mean()), "actions": part.action.value_counts(normalize=True).round(3).to_dict(),
                               "b1_stop_rate": float(part.b1_stop.mean()), "b2_stop_rate": float(part.b2_stop.mean()),
                               "b2_would_call": "payload (c0 shifts) - a false fault type"}
print("S4:", json.dumps(summary["S4_sensor_offset"]))

# ------------------------------------------------------------------ S6: sweep over drift rate and extra noise (forecaster only)
def tau_c_of(frame):
    """Per-cycle tau_c of one frame, plain least squares."""
    X, yl, _ = cycle_design(frame)
    return fit_friction(yl, X[:, 2], X[:, 3])["coulomb"]

obs_std = float(ref_table.load_coulomb.std())
sweep = []
for rate in [0.003, 0.006, 0.012]:
    for noise in [0.0, 1.0, 2.0]:
        cov, err = [], []
        for r in range(4):
            fr, truth = sim.degradation_run(500, rate=rate, walk_std=0.01, seed=500 + r, extra_noise_std=noise)
            est = np.array([tau_c_of(f) for f in fr])
            crossing = int(np.argmax(truth >= args.tau_max)) if (truth >= args.tau_max).any() else None
            if crossing is None:
                continue
            kf = RestartingTrendKF(level0=est[0], ref_mean=healthy_params[1], ref_std=obs_std, k_onset=1.5, obs_std=obs_std, q_level=0.01, q_slope=1e-4)
            for _ in range(60):                                    # healthy history first, as in the loop
                kf.step(tau_c_of(sim.cycle(extra_noise_std=noise)))
            for k in range(crossing):
                kf.step(est[k])
                if k % 10 or k < 0.2 * crossing or k > 0.9 * crossing:
                    continue
                c = kf.cycles_to_limit(args.tau_max, horizon=800, n=1000, rng=rng); s = summarize_crossings(c, 800)
                rem = crossing - k
                cov.append(s["lo5"] <= rem <= s["hi95"]); err.append(abs(s["median"] - rem) / rem)
        sweep.append({"rate": rate, "extra_noise": noise, "coverage_90": float(np.mean(cov)) if cov else np.nan, "median_rel_err": float(np.median(err)) if err else np.nan,
                      "est_spread": float(np.std(est[:50]))})
S6 = pd.DataFrame(sweep); S6.to_csv(os.path.join(args.out, "S6_sweep.csv"), index=False)
summary["S6_sweep"] = sweep
print("S6:", S6.round(3).to_string())

pd.concat(logs).to_csv(os.path.join(args.out, "replay_log.csv"), index=False)
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ figures
colors = {"continue": "#4caf50", "slow": "#ffb300", "schedule": "#1e88e5", "stop": "#e53935", "abstain": "#9e9e9e"}
if example is not None:
    r, L, crossing = example
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax[0].plot(L.cycle, L.true_tc, "k", lw=1, label="true tau_c"); ax[0].plot(L.cycle, L.tau_c, ".", ms=3, color="gray", label="per-cycle estimate")
    ax[0].plot(L.cycle, L.kf_level, "r", lw=1.2, label="Kalman level"); ax[0].axhline(args.tau_max, color="k", ls="--", lw=0.8)
    ax[0].axvline(crossing, color="k", ls=":", lw=0.8); ax[0].set_ylabel("tau_c [N m]"); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].plot(L.cycle, L.n_median, "r", lw=1.2, label="cycles to limit, median"); ax[1].fill_between(L.cycle, L.n_lo5, L.n_hi95, color="r", alpha=0.2, label="90 % interval")
    ax[1].plot(L.cycle, crossing - L.cycle, "k", lw=1, label="truth"); ax[1].set_ylim(0, 400); ax[1].set_ylabel("cycles"); ax[1].legend(fontsize=8)
    for i, a in enumerate(L.action):
        ax[2].axvspan(i - 0.5, i + 0.5, color=colors.get(a, "white"), alpha=0.8)
    for b, y, m in [("b1", 0.8, "v"), ("b2", 0.2, "^")]:
        trips = L.index[L[f"{b}_stop"]]
        if len(trips):
            ax[2].plot(trips[0], y, m, color="k", ms=8); ax[2].text(trips[0] + 2, y, f"{b.upper()} first stop", fontsize=8, va="center")
    ax[2].set_yticks([]); ax[2].set_xlabel("cycle"); ax[2].set_title("twin's action per cycle: green continue, orange slow, blue schedule, red stop, grey abstain", fontsize=9)
    res = S2[S2.run == r].iloc[0]
    fig.suptitle(f"S2, one run: twin -> {res.twin_outcome} (cost {res.twin_cost:.1f}); B1 -> {res.b1_outcome}; B2 -> {res.b2_outcome}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "S2_example.png"), dpi=130); plt.close(fig)

# actions per scenario
rows = [(k, v["actions"]) for k, v in [(f"S1 {c}", s1[c]) for c in s1] + [("S3 payload", summary["S3_payload"]), ("S4 sensor offset", summary["S4_sensor_offset"]), ("S5 unseen", summary["S5_unseen_wobble"])]]
fig, ax = plt.subplots(figsize=(10, 4))
left = np.zeros(len(rows))
for a in ["continue", "slow", "schedule", "stop", "abstain"]:
    vals = np.array([r[1].get(a, 0.0) for r in rows])
    ax.barh(range(len(rows)), vals, left=left, color=colors[a], label=a); left += vals
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlabel("fraction of cycles"); ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02))
ax.set_title("what the twin decides, scenario by scenario (held-out recorded cycles; S4 synthetic)", pad=28, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "actions_by_scenario.png"), dpi=130); plt.close(fig)

# cost comparison on S2
fig, ax = plt.subplots(figsize=(6, 3.6))
m = summary["S2_degradation"]["mean_cost"]
ax.bar(["twin", "B1 threshold", "B2 classifier only"], [m["twin"], m["b1"], m["b2"]], color=["#1e88e5", "#9e9e9e", "#757575"])
ax.set_ylabel("mean cost per degradation run"); ax.set_title(f"S2: {args.runs} simulated runs (missed = {C['missed']:.0f}, unplanned stop = {C['stop']:.0f}, planned = {C['sched']:.0f})", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "S2_cost.png"), dpi=130); plt.close(fig)

# S6 heatmaps
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for a, col, title in zip(axs, ["coverage_90", "median_rel_err"], ["coverage of the 90 % interval", "median relative error of N*"]):
    P = S6.pivot(index="extra_noise", columns="rate", values=col)
    im = a.imshow(P.values, cmap="viridis", vmin=0, vmax=1)
    a.set_xticks(range(len(P.columns))); a.set_xticklabels(P.columns); a.set_yticks(range(len(P.index))); a.set_yticklabels(P.index)
    a.set_xlabel("drift rate [N m / cycle]"); a.set_ylabel("extra noise std [N m]"); a.set_title(title, fontsize=9)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            a.text(j, i, f"{P.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "S6_sweep.png"), dpi=130); plt.close(fig)
print("done ->", args.out, f"({time.time()-t0:.0f} s)")
