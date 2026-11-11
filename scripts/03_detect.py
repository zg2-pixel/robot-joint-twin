"""Step 3: track what 'healthy' looks like, detect changes, classify them.

    python scripts/03_detect.py               # joint 2
    python scripts/03_detect.py --axis 3

Writes results/03_detect/: tracking_axisK.csv (reference state cycle by cycle through the five healthy batches),
scenarios.json (false alarms, detection delay on the injected steps, other-axis faults, sensor offset, unseen
anomalies), classifier metrics and confusion matrices, figures.
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin import dataset as D
from twin.features import build_cycle_table
from twin.physics import cycle_design, fit_friction
from twin.bayes import NIG, ReferenceTracker
from twin import detect as DT

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=D.DEFAULT_PATH)
ap.add_argument("--axis", type=int, default=2)
ap.add_argument("--lam", type=float, default=0.98, help="forgetting factor of the reference (0.98 ~ 50 cycles of memory)")
ap.add_argument("--k", type=float, default=4.0, help="z-score threshold for a flag")
ap.add_argument("--out", default=os.path.join(D.ROOT, "results", "03_detect"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(0)
K = args.k

t0 = time.time()
df = D.load_axis(args.axis, args.data)
labels = D.cycle_labels(df)
T = build_cycle_table(df, labels)
T["label"] = DT.label_for_joint(T, args.axis)
T.to_csv(os.path.join(args.out, f"cycles_axis{args.axis}.csv"))
print(f"cycle table: {len(T)} cycles, {time.time()-t0:.0f} s"); print(T.label.value_counts().to_string())

# design matrices of every cycle, for the Bayesian updates
design = {s: cycle_design(g) for s, g in df.groupby("sample", sort=True)}
del df

BATCHES = ["PRE_A", "PRE_B", "BETWEEN_A", "BETWEEN_B", "BETWEEN_C"]
healthy = T[T.category_name == "normal"]
order = [s for b in BATCHES for s in healthy.index[healthy.variant == b]]      # one long healthy run, batch after batch


def ls(s):
    """Plain per-cycle least squares (what Deliverable 1 used), with standard errors."""
    X, yl, yg = design[s]
    fl = fit_friction(yl, X[:, 2], X[:, 3]); fg = fit_friction(yg, X[:, 2], X[:, 3])
    bl = np.array([fl["c0"], fl["coulomb"], fl["viscous"], fl["inertia"]])
    bg = np.array([fg["c0"], fg["coulomb"], fg["viscous"], fg["inertia"]])
    return bl, bg, fl["se_coulomb"], fg["se_coulomb"]


def make_tracker(warm):
    """Prior from the first cycles of the run, then a tracker with forgetting."""
    bl = []; bg = []; nl = []; ng = []
    for s in warm:
        a, b, _, _ = ls(s); bl.append(a); bg.append(b)
        X, yl, yg = design[s]
        nl.append(np.std(yl - X @ a)); ng.append(np.std(yg - X @ b))
    return ReferenceTracker(NIG.from_cycles(bl, nl), NIG.from_cycles(bg, ng), lam=args.lam)

# healthy-band statistics for the per-cycle spread (Deliverable 2: the empirical spread is the honest number)
spread = healthy.groupby("variant")[["load_coulomb", "load_c0", "load_viscous", "gear_viscous"]].std().mean()


def run(tracker, seq, gate=True, extra=None):
    """Replay a sequence of cycles through the tracker. Returns one row per cycle."""
    rows = []
    for s in seq:
        X, yl, yg = design[s]
        if extra is not None:
            yl = yl + extra                                   # synthetic sensor offset on torque sensor A
        bl, bg, _, _ = ls(s)
        if extra is not None:
            bl[0] += extra
        ref_l = tracker.load; ref_g = tracker.gear
        z_tc = (bl[1] - ref_l.m[1]) / np.sqrt(spread["load_coulomb"] ** 2 + ref_l.param_std()[1] ** 2)
        z_c0 = (bl[0] - ref_l.m[0]) / np.sqrt(spread["load_c0"] ** 2 + ref_l.param_std()[0] ** 2)
        z_bv = (bg[2] - ref_g.m[2]) / np.sqrt(spread["gear_viscous"] ** 2 + ref_g.param_std()[2] ** 2)
        ok = (abs(z_tc) < K) and (abs(z_c0) < K) if gate else True
        rows.append({"sample": s, "variant": T.loc[s, "variant"], "label": T.loc[s, "label"],
                     "tc": bl[1], "ref_tc": ref_l.m[1], "ref_tc_std": ref_l.param_std()[1],
                     "c0": bl[0], "ref_c0": ref_l.m[0], "gear_b": bg[2], "ref_gear_b": ref_g.m[2], "ref_gear_b_std": ref_g.param_std()[2],
                     "z_tc": z_tc, "z_c0": z_c0, "z_gear_b": z_bv, "updated": ok,
                     "p_change": 2 * norm.sf(abs(z_tc))})
        tracker.step(X, yl, yg, healthy=ok)
    return pd.DataFrame(rows).set_index("sample")

# ------------------------------------------------------------------ 1. tracking through the healthy batches
warm = order[:20]
tracker = make_tracker(warm)
track = run(tracker, order)
track.to_csv(os.path.join(args.out, f"tracking_axis{args.axis}.csv"))
after_warm = track.iloc[20:]
fa_rate = float((np.abs(after_warm.z_tc) > K).mean())
# how long the gear reference needs after a batch change: cycles until |z_gear_b| < 2 again
adapt = {}
for b in BATCHES[1:]:
    idx = np.where(track.variant.values == b)[0]
    if len(idx) == 0:
        continue
    z = np.abs(track.z_gear_b.values[idx])
    first_ok = next((i for i, v in enumerate(z) if v < 2), None)
    adapt[b] = {"first_z_gear": float(z[0]), "cycles_to_settle": first_ok, "mean_z_tc": float(np.abs(track.z_tc.values[idx]).mean())}
print(f"tracking: false-alarm rate on tau_c over {len(after_warm)} healthy cycles = {fa_rate:.4f}; batch adaptation {adapt}")

# ------------------------------------------------------------------ 2. scenarios
scen = {"settings": {"lam": args.lam, "k": K, "axis": args.axis}, "false_alarm_rate_healthy": fa_rate, "batch_adaptation": adapt}

# a frozen healthy reference (end of PRE_A) for the step scenarios
pre_a = [s for s in order if T.loc[s, "variant"] == "PRE_A"]
base = make_tracker(pre_a[:20]); run(base, pre_a[:300])
ref_state = (base.load.m.copy(), base.load.param_std().copy(), base.gear.m.copy(), base.gear.param_std().copy())

def step_scenario(name, fault_ids):
    tr = make_tracker(pre_a[:20]); run(tr, pre_a[:300])
    seq = pre_a[300:340] + list(fault_ids)
    r = run(tr, seq)
    z = r.z_tc.values; n0 = 40
    flagged = np.where(np.abs(z[n0:]) > K)[0]
    delay = int(flagged[0]) + 1 if len(flagged) else None
    return r, {"n_fault_cycles": len(fault_ids), "detection_delay_cycles": delay,
               "fraction_flagged": float((np.abs(z[n0:]) > K).mean()), "mean_z_fault": float(np.abs(z[n0:]).mean()),
               "mean_tc_fault": float(r.tc.values[n0:].mean()), "ref_tc": float(r.ref_tc.values[n0])}

steps = {}
own = T[(T.label == "friction")]
for lvl in sorted(own.variant.unique()):
    ids = own.index[own.variant == lvl].tolist()
    r, s = step_scenario(lvl, ids); steps[lvl] = s
    if lvl.endswith("_A"):
        step_A = r
# the real stepwise sequence: healthy -> level B -> level A (level A is the stronger one)
lv = sorted(own.variant.unique())
seqB = own.index[own.variant == lv[1]].tolist() if len(lv) > 1 else []
seqA = own.index[own.variant == lv[0]].tolist()
tr = make_tracker(pre_a[:20]); run(tr, pre_a[:300])
stair = run(tr, pre_a[300:340] + seqB + seqA)
stair.to_csv(os.path.join(args.out, f"staircase_axis{args.axis}.csv"))
scen["steps"] = steps

# faults on other axes must not move this joint's indicator
other = T.index[(T.category_name == "axis_friction") & (T.variant_axis != args.axis)].tolist()
tr = make_tracker(pre_a[:20]); run(tr, pre_a[:300])
r_other = run(tr, other, gate=False)
scen["other_axis_friction"] = {"n": len(other), "fraction_flagged": float((np.abs(r_other.z_tc) > K).mean()), "mean_z": float(np.abs(r_other.z_tc).mean())}

# a sensor offset: +0.5 N m on torque sensor A, injected into healthy cycles
tr = make_tracker(pre_a[:20]); run(tr, pre_a[:300])
r_off = run(tr, pre_a[300:340], gate=False, extra=0.5)
hc = DT.HealthCheck(healthy.loc[pre_a[:300]], k=K)
ab_flag = [(abs(T.loc[s, "sensor_ab_mean"] + 0.5 - hc.band["sensor_ab_mean"][0]) > K * hc.band["sensor_ab_mean"][1]) for s in pre_a[300:340]]
scen["sensor_offset_0p5Nm"] = {"n": 40, "tau_c_flagged": float((np.abs(r_off.z_tc) > K).mean()), "c0_flagged": float((np.abs(r_off.z_c0) > K).mean()),
                                "health_check_flagged": float(np.mean(ab_flag)), "note": "offset moves c0 (looks like payload) but not tau_c; the A-B check catches it"}

# anomalies the classifier never sees: does the residual test notice them?
rt = DT.ResidualTest(healthy.loc[pre_a[:300]], k=K)
unseen = T[T.label == "unseen"]
per_cat = {}
for c, g in unseen.groupby("category_name"):
    zs = np.array([rt.z(row) for _, row in g.iterrows()])
    per_cat[c] = {"n": len(g), "fraction_flagged": float((zs > K).mean()), "mean_z": float(zs.mean())}
zs_h = np.array([rt.z(row) for _, row in healthy.loc[pre_a[300:]].iterrows()])
scen["unseen_residual_test"] = {"healthy_false_alarm": float((zs_h > K).mean()), "per_category": per_cat}
# and the labelled fault types, same test
lab_rt = {}
for c, g in T[T.label.isin(["friction", "payload", "collision", "electrical"])].groupby("label"):
    zs = np.array([rt.z(row) for _, row in g.iterrows()])
    lab_rt[c] = {"n": len(g), "fraction_flagged": float((zs > K).mean()), "mean_z": float(zs.mean())}
scen["labelled_residual_test"] = lab_rt
json.dump(scen, open(os.path.join(args.out, "scenarios.json"), "w"), indent=1)
print("scenarios:", json.dumps({k: scen[k] for k in ["false_alarm_rate_healthy", "steps", "other_axis_friction", "sensor_offset_0p5Nm"]}, indent=0)[:900])

# ------------------------------------------------------------------ 3. classifier
lab = T[T.label.isin(DT.CLASSES)]
X = lab[DT.FEATURES]; y = lab["label"].to_numpy(dtype=object)   # plain numpy strings, newer pandas would hand sklearn an Arrow array
metrics = {}
probs = {}
for kind in ["nb", "knn"]:
    P = DT.cv_probabilities(X, y, kind=kind)
    probs[kind] = P
    m = DT.summarize(P, y)
    m["confusion"].to_csv(os.path.join(args.out, f"confusion_{kind}_axis{args.axis}.csv"))
    metrics[kind] = {"accuracy": m["accuracy"], "recall": m["recall"], "brier": m["brier"]}
    print(kind, "acc %.3f brier %.3f recall" % (m["accuracy"], m["brier"]), {c: round(v, 2) for c, v in m["recall"].items()})
    print(m["confusion"].to_string())

# leave-one-variant-out for friction: train on one injected level, test on the other
loo = {}
for kind in ["nb", "knn"]:
    loo[kind] = {}
    for lvl in sorted(lab.variant[lab.label == "friction"].unique()):
        test = lab.index[(lab.label == "friction") & (lab.variant == lvl)]
        train = lab.index.difference(test)
        clf = DT.make_classifier(kind).fit(X.loc[train], y[lab.index.get_indexer(train)])
        p = clf.predict(X.loc[test])
        loo[kind][lvl] = float((p == "friction").mean())
metrics["leave_one_level_out_friction_recall"] = loo

# unseen anomalies scored by a classifier trained on everything labelled: how confident is it, wrongly?
clf = DT.make_classifier("nb").fit(X, y)
Pu = pd.DataFrame(clf.predict_proba(unseen[DT.FEATURES]), columns=clf.classes_, index=unseen.index)
metrics["unseen_scored_by_nb"] = {"fraction_called_healthy": float((Pu.idxmax(axis=1) == "healthy").mean()),
                                  "mean_max_prob": float(Pu.max(axis=1).mean())}
# reliability of P(not healthy) for both models
rel = {}
for kind, P in probs.items():
    p_fault = 1 - P["healthy"].values
    truth = (y != "healthy").astype(float)
    r = DT.reliability(p_fault, truth)
    r.to_csv(os.path.join(args.out, f"reliability_{kind}_axis{args.axis}.csv"), index=False)
    rel[kind] = r
json.dump(metrics, open(os.path.join(args.out, f"classifier_axis{args.axis}.json"), "w"), indent=1)
print("leave-one-level-out:", loo, "| unseen:", metrics["unseen_scored_by_nb"])

# ------------------------------------------------------------------ figures
# tracking through the batches
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
x = np.arange(len(track))
ax[0].plot(x, track.tc, ".", ms=2, color="gray", label="per-cycle estimate")
ax[0].plot(x, track.ref_tc, "r", lw=1.2, label="reference (posterior mean)")
ax[0].fill_between(x, track.ref_tc - 2 * track.ref_tc_std, track.ref_tc + 2 * track.ref_tc_std, color="r", alpha=0.2, label="reference ± 2 s.d.")
ax[0].set_ylabel("tau_c [N m]"); ax[0].legend(fontsize=8, loc="upper right")
ax[1].plot(x, track.gear_b, ".", ms=2, color="gray"); ax[1].plot(x, track.ref_gear_b, "r", lw=1.2)
ax[1].fill_between(x, track.ref_gear_b - 2 * track.ref_gear_b_std, track.ref_gear_b + 2 * track.ref_gear_b_std, color="r", alpha=0.2)
ax[1].set_ylabel("gear viscous [N m s/rad]")
ax[2].plot(x, track.z_tc, lw=0.6); ax[2].axhline(K, color="k", ls="--", lw=0.8); ax[2].axhline(-K, color="k", ls="--", lw=0.8)
ax[2].set_ylabel("z of tau_c"); ax[2].set_xlabel("healthy cycles, batch after batch")
for b in BATCHES:
    idx = np.where(track.variant.values == b)[0]
    if len(idx):
        for a in ax: a.axvline(idx[0], color="gray", lw=0.6, ls=":")
        ax[0].text(idx[0] + 3, ax[0].get_ylim()[1] * 0.95, b, fontsize=7, va="top")
fig.suptitle(f"joint {args.axis}: the reference follows the machine (gear viscous friction) while tau_c stays put")
fig.tight_layout(); fig.savefig(os.path.join(args.out, f"tracking_axis{args.axis}.png"), dpi=130); plt.close(fig)

# staircase: healthy -> level B -> level A
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
x = np.arange(len(stair))
ax[0].plot(x, stair.tc, "o", ms=3, color="gray", label="per-cycle tau_c")
ax[0].plot(x, stair.ref_tc, "r", lw=1.2, label="reference"); ax[0].fill_between(x, stair.ref_tc - 2 * stair.ref_tc_std, stair.ref_tc + 2 * stair.ref_tc_std, color="r", alpha=0.2)
n_h = 40; nB = len(seqB)
ax[0].axvspan(n_h - 0.5, n_h + nB - 0.5, color="orange", alpha=0.15, label=f"friction {lv[1] if len(lv) > 1 else ''} injected")
ax[0].axvspan(n_h + nB - 0.5, len(stair) - 0.5, color="red", alpha=0.12, label=f"friction {lv[0]} injected")
ax[0].set_ylabel("tau_c [N m]"); ax[0].legend(fontsize=8)
ax[1].plot(x, stair.z_tc, "o-", ms=3, lw=0.8); ax[1].axhline(K, color="k", ls="--", lw=0.8); ax[1].set_ylabel("z-score vs reference"); ax[1].set_xlabel("cycle")
fig.suptitle(f"joint {args.axis}: stepwise degradation from recorded cycles - detection delay {steps[lv[1]]['detection_delay_cycles'] if len(lv) > 1 else steps[lv[0]]['detection_delay_cycles']} cycle(s)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, f"staircase_axis{args.axis}.png"), dpi=130); plt.close(fig)

# classifier: confusion matrices and reliability
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for a, kind in zip(axs[:2], ["nb", "knn"]):
    cm = pd.read_csv(os.path.join(args.out, f"confusion_{kind}_axis{args.axis}.csv"), index_col=0)
    cmn = cm.div(cm.sum(axis=1), axis=0)
    a.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            a.text(j, i, f"{cm.values[i, j]}", ha="center", va="center", fontsize=8, color="white" if cmn.values[i, j] > 0.6 else "black")
    a.set_xticks(range(len(cm))); a.set_xticklabels(cm.columns, rotation=40, fontsize=8); a.set_yticks(range(len(cm))); a.set_yticklabels(cm.index, fontsize=8)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"{kind.upper()}: accuracy {metrics[kind]['accuracy']:.2f}, Brier {metrics[kind]['brier']:.3f}")
a = axs[2]
for kind, r in rel.items():
    a.plot(r.predicted, r.observed, "o-", label=f"{kind.upper()}")
a.plot([0, 1], [0, 1], "k--", lw=0.8); a.set_xlabel("predicted P(not healthy)"); a.set_ylabel("observed fraction"); a.set_title("reliability (out-of-fold, calibrated)"); a.legend()
fig.tight_layout(); fig.savefig(os.path.join(args.out, f"classifier_axis{args.axis}.png"), dpi=130); plt.close(fig)

print("done ->", args.out, f"({time.time()-t0:.0f} s)")
