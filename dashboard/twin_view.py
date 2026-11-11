"""Helpers for the dashboard: read the replay log, build the figures, draw the arm.

Nothing here imports streamlit, so the functions can be tested on their own.
The arm is a schematic. Link lengths are roughly those of the Yu 5 (about 1.1 m reach), the joint
angles are the recorded ones, and the fixed offsets were picked by eye so the pick-and-place looks
right (tool tip touches the table at the pick). It shows where the robot is, it is not a CAD model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from twin.decide import DecisionRule, DEFAULT_COSTS

LOG_PATH = ROOT / "results" / "05_replay" / "replay_log.csv"
POSES_PATH = ROOT / "results" / "06_dashboard" / "poses.parquet"
PARQUET_PATH = ROOT / "data" / "voraus-ad-dataset-100hz.parquet"

ACTION_COLOR = {"continue": "#2e8b57", "slow": "#e0a500", "schedule": "#2f6fd6", "stop": "#c0392b", "abstain": "#8c8c8c"}
ACTION_TEXT = {
    "continue": "keep running",
    "slow": "slow down to 70 % speed",
    "schedule": "open a maintenance ticket",
    "stop": "stop the robot",
    "abstain": "sensor check failed, no decision",
}
CLASSES = ["healthy", "friction", "payload", "collision", "electrical"]
CLASS_COLOR = {"healthy": "#2e8b57", "friction": "#c0392b", "payload": "#e0a500", "collision": "#8e44ad", "electrical": "#2f6fd6"}

SCENARIOS = {
    "S1_healthy": "S1: healthy hold-out cycles",
    "S1_friction": "S1: friction added on joint 2 (recorded)",
    "S1_payload": "S1: extra payload (recorded)",
    "S1_collision": "S1: collisions (recorded)",
    "S1_electrical": "S1: mis-commutated motor (recorded)",
    "S3_payload": "S3: payload change, forecast must stay quiet",
    "S4_sensor_offset": "S4: torque sensor drifts by 0.5 N m",
    "S5_unseen_wobble": "S5: unseen fault: unstable platform",
}
for _k in range(10):
    SCENARIOS[f"S2_run{_k}"] = f"S2: simulated wear, run {_k}"

# schematic arm: link lengths in m and the angle offsets that make the recorded angles look right
LINKS = {"base": 0.16, "upper": 0.44, "fore": 0.40, "wrist": 0.12, "tool": 0.10}
OFFSETS = (0.0, -2.1, 2.1)
JOINT_NAMES = ["base", "shoulder (joint 2)", "elbow", "wrist", "flange", "tool"]


# ---------- the log ----------

def load_log(path=LOG_PATH):
    d = pd.read_csv(path)
    d["note"] = d["note"].fillna("").astype(str)
    for c in ["sensor_ok", "ticket_open", "b1_stop", "b2_stop", "unknown_flag"]:
        d[c] = d[c].map(lambda v: str(v).strip().lower() == "true")
    return d


def scenario_names(log):
    names = [s for s in SCENARIOS if s in set(log["scenario"])]
    return names + sorted(set(log["scenario"]) - set(names))


def scenario_frame(log, name):
    return log[log["scenario"] == name].reset_index(drop=True)


def true_crossing(sf, tau_max):
    """Cycle where the true friction crosses the limit (simulated runs only), else None."""
    if "true_tc" not in sf or sf["true_tc"].isna().all():
        return None
    over = sf.index[sf["true_tc"] > tau_max]
    return int(over[0]) if len(over) else None


def health_colour(z):
    z = abs(float(z)) if np.isfinite(z) else 0.0
    return "#2e8b57" if z < 2 else ("#e0a500" if z < 4 else "#c0392b")


def injected_from(sf):
    """First cycle that is not a healthy warm-up cycle (None when the whole scenario is healthy)."""
    lab = np.array([str(v) if isinstance(v, str) else "" for v in sf["label"]])
    if (lab != "").any():
        bad = sf.index[(lab != "") & (lab != "healthy")]
        return int(bad[0]) if len(bad) else None
    off = sf.index[~sf["sensor_ok"]]            # the synthetic sensor offset has no labels, the health check marks it
    return int(off[0]) if len(off) else None


def out_of_service(sf, i):
    """Where the robot left service by cycle i: the planned stop if it came first, else the first stop. None if still running."""
    t = ticket_info(sf, i)
    s = first_stop(sf, i)
    planned = t["planned_at"] if t is not None and t["planned_at"] <= int(sf["cycle"].iloc[i]) else None
    if planned is not None and (s is None or planned <= s):
        return {"cycle": planned, "planned": True}
    if s is not None:
        return {"cycle": s, "planned": False}
    return None


def ticket_info(sf, i):
    """When the ticket was opened and for which cycle the stop was planned, if there is one by cycle i."""
    d = sf.iloc[: i + 1]
    t = d.index[d["ticket_open"]]
    if not len(t):
        return None
    k = int(t[0])
    return {"opened": int(sf["cycle"].iloc[k]), "planned_at": int(sf["cycle"].iloc[k] + sf["t_plan"].iloc[k])}


def first_stop(sf, i):
    d = sf.iloc[: i + 1]
    s = d.index[d["action"] == "stop"]
    return int(sf["cycle"].iloc[int(s[0])]) if len(s) else None


def kpis(sf, i, tau_max):
    r = sf.iloc[i]
    cross = true_crossing(sf, tau_max)
    sample = r.get("sample", np.nan)
    return {
        "sample": int(sample) if sample is not None and np.isfinite(sample) else None,
        "variant": str(r["variant"]) if isinstance(r.get("variant"), str) else "",
        "injected_from": injected_from(sf),
        "ticket": ticket_info(sf, i),
        "out": out_of_service(sf, i),
        "cycle": int(r["cycle"]),
        "tau_c": float(r["tau_c"]),
        "ref_tc": float(r["ref_tc"]),
        "z_tc": float(r["z_tc"]),
        "p_fault": float(r["p_fault"]),
        "n_median": float(r["n_median"]),
        "n_lo5": float(r["n_lo5"]),
        "n_hi95": float(r["n_hi95"]),
        "action": str(r["action"]),
        "note": str(r["note"]),
        "speed": float(r["speed"]),
        "ticket_open": bool(r["ticket_open"]),
        "sensor_ok": bool(r["sensor_ok"]),
        "label": str(r["label"]),
        "true_remaining": (cross - int(r["cycle"])) if cross is not None else None,
    }


# ---------- poses of the arm ----------

def synthetic_cycle(n=110):
    """A made-up pick-and-place with the joint ranges of the data, used when the parquet is not there."""
    t = np.linspace(0, 1, n)
    lift = 0.5 - 0.5 * np.cos(2 * np.pi * t)       # up and back down
    swing = 0.5 - 0.5 * np.cos(np.pi * t)          # from the pick to the place
    return np.column_stack([
        -0.3 + 1.3 * swing,
        1.20 + 0.38 * lift,
        1.86 + 0.46 * lift,
        -2.58 + 0.50 * lift,
        -1.83 + 0.28 * lift,
        -0.80 + 1.40 * swing,
    ])


def build_pose_cache(parquet=PARQUET_PATH, cache=POSES_PATH, step=10):
    """Joint angles of every recorded cycle at 10 Hz, stored once so the big file is read only once."""
    import pyarrow.parquet as pq
    cols = [f"joint_position_{k}" for k in range(1, 7)]
    tab = pq.read_table(parquet, columns=["sample", "time", "category"] + cols).to_pandas()
    tab = tab.sort_values(["sample", "time"])
    tab["k"] = tab.groupby("sample").cumcount()
    tab = tab[tab["k"] % step == 0].drop(columns=["k", "time"])
    tab = tab.astype({c: "float32" for c in cols})
    cache.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(cache, index=False)
    return tab


def load_poses(cache=POSES_PATH, parquet=PARQUET_PATH):
    """Returns (dict sample -> (n, 6) angles, template angles, source string)."""
    cols = [f"joint_position_{k}" for k in range(1, 7)]
    if cache.exists():
        tab = pd.read_parquet(cache)
        source = "recorded trajectories (cached)"
    elif parquet.exists():
        tab = build_pose_cache(parquet, cache)
        source = "recorded trajectories"
    else:
        return {}, synthetic_cycle(), "synthetic trajectory (data file not found)"
    poses = {int(s): g[cols].to_numpy(dtype=float) for s, g in tab.groupby("sample")}
    healthy = tab.loc[tab["category"] == 12, "sample"]
    template = poses[int(healthy.iloc[0])] if len(healthy) else next(iter(poses.values()))
    return poses, template, source


def angles_for(row, poses, template):
    """Recorded angles of this cycle if we have them, otherwise the template (simulated cycles)."""
    s = row.get("sample", np.nan)
    if s is not None and np.isfinite(s) and int(s) in poses:
        return poses[int(s)]
    return template


def fk(q, links=LINKS, offsets=OFFSETS):
    """Joint angles -> xyz of base, shoulder, elbow, wrist, flange, tool tip (6 x 3)."""
    q1, q2, q3, q4 = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    def link(a, length):  # a link tilted by angle a from vertical, in the plane set by the base yaw
        return length * np.array([np.sin(a) * np.cos(q1), np.sin(a) * np.sin(q1), np.cos(a)])

    shoulder = np.array([0.0, 0.0, links["base"]])
    a2 = q2 + offsets[0]
    elbow = shoulder + link(a2, links["upper"])
    a3 = a2 + q3 + offsets[1]
    wrist = elbow + link(a3, links["fore"])
    a4 = a3 + q4 + offsets[2]
    flange = wrist + link(a4, links["wrist"])
    tip = flange + np.array([0.0, 0.0, -links["tool"]])
    return np.vstack([np.zeros(3), shoulder, elbow, wrist, flange, tip])


def arm_figure(q_cycle, z_tc, tau_c, tau_max, action, speed=1.0, cycle_seconds=11.0, max_frames=60):
    """3D view of the arm over one cycle, with play/pause. The monitored joint is coloured by its health."""
    q_cycle = np.asarray(q_cycle, dtype=float)
    if len(q_cycle) > max_frames:
        q_cycle = q_cycle[np.linspace(0, len(q_cycle) - 1, max_frames).astype(int)]
    P = np.array([fk(q) for q in q_cycle])          # (frames, 6, 3)
    colour = health_colour(z_tc)
    frame_ms = int(1000 * cycle_seconds / len(P) / max(speed, 0.1))
    tip = P[:, 5, :]

    def links_trace(p):
        return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers",
                            line=dict(color="#4a4a4a", width=14),
                            marker=dict(size=[9, 11, 9, 8, 6, 5], color="#9a9a9a"),
                            text=JOINT_NAMES, hoverinfo="text", name="arm", showlegend=False)

    def joint_trace(p):
        size = 14 + 12 * min(max(float(tau_c) / tau_max, 0.0), 1.5)      # grows with the friction
        return go.Scatter3d(x=[p[1, 0]], y=[p[1, 1]], z=[p[1, 2]], mode="markers",
                            marker=dict(size=size, color=colour, opacity=0.6, line=dict(color=colour, width=2)),
                            hoverinfo="text", text=[f"joint 2  tau_c = {tau_c:.2f} N m  z = {z_tc:.1f}"],
                            name="joint 2", showlegend=False)

    table = go.Mesh3d(x=[-0.35, 1.05, 1.05, -0.35], y=[-0.6, -0.6, 0.95, 0.95], z=[0, 0, 0, 0],
                      i=[0, 0], j=[1, 2], k=[2, 3], color="#d9d9d9", opacity=0.5, hoverinfo="skip", showlegend=False)
    path = go.Scatter3d(x=tip[:, 0], y=tip[:, 1], z=tip[:, 2], mode="lines", line=dict(color="#b0b0b0", width=3),
                        hoverinfo="skip", name="tool path", showlegend=False)
    base = go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, LINKS["base"]], mode="lines", line=dict(color="#2b2b2b", width=22),
                        hoverinfo="skip", showlegend=False)

    fig = go.Figure(data=[links_trace(P[0]), joint_trace(P[0]), table, path, base])
    fig.frames = [go.Frame(data=[links_trace(p), joint_trace(p)], traces=[0, 1], name=str(n)) for n, p in enumerate(P)]
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.35, 1.05], title="x [m]", showbackground=False),
            yaxis=dict(range=[-0.6, 0.95], title="y [m]", showbackground=False),
            zaxis=dict(range=[0, 0.8], title="z [m]", showbackground=False),
            aspectmode="manual", aspectratio=dict(x=1.4, y=1.55, z=0.8),
            camera=dict(eye=dict(x=1.8, y=-1.7, z=0.7)),
        ),
        margin=dict(l=0, r=0, t=30, b=0), height=430,
        title=dict(text=f"joint 2: tau_c = {tau_c:.2f} N m (limit {tau_max:.2f}), z = {z_tc:+.1f}  ->  {ACTION_TEXT.get(action, action)}",
                   font=dict(size=14, color=ACTION_COLOR.get(action, "#333"))),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                          buttons=[
                              dict(label="play", method="animate",
                                   args=[None, dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                              dict(label="pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                          ])],
    )
    return fig


# ---------- the panels ----------

SPREAD = 0.17   # cycle-to-cycle spread of tau_c in healthy operation (the Deliverable 2 budget), used by the z-score


def health_figure(sf, i, tau_max, k=4.0):
    """tau_c cycle by cycle up to now, the reference and its band, the KF level and the limit."""
    d = sf.iloc[: i + 1]
    fig = go.Figure()
    half = k * np.sqrt(SPREAD ** 2 + d["ref_tc_std"] ** 2)       # the same band the z-score uses
    lo = d["ref_tc"] - half
    hi = d["ref_tc"] + half
    fig.add_trace(go.Scatter(x=pd.concat([d["cycle"], d["cycle"][::-1]]), y=pd.concat([hi, lo[::-1]]), fill="toself",
                             fillcolor="rgba(46,139,87,0.10)", line=dict(width=0), hoverinfo="skip", name=f"healthy band (±{k:g} sigma)"))
    fig.add_trace(go.Scatter(x=d["cycle"], y=d["ref_tc"], mode="lines", name="tracked reference", line=dict(color="#2e8b57", width=1)))
    for a in ["continue", "slow", "schedule", "stop", "abstain"]:
        m = d["action"] == a
        if m.any():
            fig.add_trace(go.Scatter(x=d.loc[m, "cycle"], y=d.loc[m, "tau_c"], mode="markers", name=a,
                                     marker=dict(color=ACTION_COLOR[a], size=6)))
    if "true_tc" in d and d["true_tc"].notna().any():
        fig.add_trace(go.Scatter(x=d["cycle"], y=d["true_tc"], mode="lines", name="true tau_c (simulation)",
                                 line=dict(color="#555", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=d["cycle"], y=d["kf_level"], mode="lines", name="tracked level (KF)", line=dict(color="#1f3f8f", width=2)))
    fig.add_hline(y=tau_max, line=dict(color="#c0392b", dash="dash"), annotation_text="limit", annotation_position="top left")
    fig.update_layout(xaxis=dict(title="cycle", range=[-1, len(sf)]), yaxis=dict(title="Coulomb friction tau_c [N m]"),
                      margin=dict(l=10, r=10, t=30, b=10), height=330, legend=dict(orientation="h", y=1.12, font=dict(size=10)),
                      title=dict(text="health indicator", font=dict(size=14)))
    return fig


def forecast_figure(sf, i, tau_max, lead=30):
    """Cycles to the limit: median and 90 % interval of every forecast so far, against the truth when known."""
    d = sf.iloc[: i + 1]
    cross = true_crossing(sf, tau_max)
    fig = go.Figure()
    lo = d["n_lo5"].clip(lower=1)
    hi = d["n_hi95"].clip(lower=1)
    fig.add_trace(go.Scatter(x=pd.concat([d["cycle"], d["cycle"][::-1]]), y=pd.concat([hi, lo[::-1]]), fill="toself",
                             fillcolor="rgba(47,111,214,0.18)", line=dict(width=0), hoverinfo="skip", name="90 % interval"))
    fig.add_trace(go.Scatter(x=d["cycle"], y=d["n_median"].clip(lower=1), mode="lines", name="median N*", line=dict(color="#2f6fd6", width=2)))
    if cross is not None:
        x = sf["cycle"][: cross + 1]
        fig.add_trace(go.Scatter(x=x, y=(cross - x).clip(lower=1), mode="lines", name="true remaining", line=dict(color="#555", dash="dot")))
    t = d.loc[d["ticket_open"], "cycle"]
    if len(t):
        fig.add_vline(x=float(t.iloc[0]), line=dict(color="#2f6fd6", dash="dash"), annotation_text="ticket", annotation_position="top right")
    fig.add_hline(y=lead, line=dict(color="#e0a500", dash="dash"), annotation_text=f"lead time {lead}", annotation_position="bottom left")
    fig.update_layout(xaxis=dict(title="cycle", range=[-1, len(sf)]), yaxis=dict(title="cycles to limit N*", type="log", range=[0, 2.9]),
                      margin=dict(l=10, r=10, t=30, b=10), height=330, legend=dict(orientation="h", y=1.12, font=dict(size=10)),
                      title=dict(text="forecast", font=dict(size=14)))
    return fig


def class_figure(row):
    """What the classifier thinks this cycle is."""
    p = [float(row.get(f"p_{c}", 0.0)) for c in CLASSES]
    fig = go.Figure(go.Bar(x=p, y=CLASSES, orientation="h", marker_color=[CLASS_COLOR[c] for c in CLASSES],
                           text=[f"{v:.2f}" for v in p], textposition="outside", hoverinfo="skip"))
    note = "residual test: unseen pattern" if bool(row.get("unknown_flag", False)) else "residual test: fits the known classes"
    fig.update_layout(xaxis=dict(range=[0, 1.15], title="probability"), yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=30, b=10), height=260, title=dict(text=f"cycle class ({note})", font=dict(size=14)))
    return fig


def expected_costs(row, costs=None):
    """Expected cost of each action for this cycle with the given costs (defaults = the logged ones)."""
    rule = DecisionRule(costs)
    pf, pw, pp = float(row["p_fault"]), float(row["p_cross_window"]), float(row["p_cross_plan"])
    # the life given up by a planned stop is backed out of the logged schedule cost
    c = DEFAULT_COSTS
    unused = (float(row["cost_schedule"]) - c["sched"] - (pf + (1 - pf) * pp) * c["missed"]) / c["waste"]
    unused = max(unused, 0.0) if np.isfinite(unused) else 0.0
    return rule.expected_costs(pf, pw, pp, unused)


def cost_figure(ec, chosen):
    acts = ["continue", "slow", "schedule", "stop"]
    vals = [float(ec.get(a, np.nan)) for a in acts]
    best = acts[int(np.nanargmin(vals))] if np.isfinite(vals).any() else chosen
    colours = [ACTION_COLOR[a] if a == chosen else "#c8c8c8" for a in acts]
    fig = go.Figure(go.Bar(x=acts, y=vals, marker_color=colours, text=[f"{v:.1f}" for v in vals], textposition="outside", hoverinfo="skip"))
    fig.update_layout(yaxis=dict(title="expected cost [cycles]", range=[0, max(np.nanmax(vals) * 1.3, 1)]),
                      margin=dict(l=10, r=10, t=30, b=10), height=260,
                      title=dict(text=f"expected cost of each action (cheapest: {best})", font=dict(size=14)))
    return fig


def timeline_figure(sf, i):
    """One coloured bar per cycle: what the twin did, plus when the two baselines would have stopped."""
    fig = go.Figure()
    for a in ["continue", "slow", "schedule", "stop", "abstain"]:
        m = sf["action"] == a
        if m.any():
            fig.add_trace(go.Bar(x=sf.loc[m, "cycle"], y=np.ones(m.sum()), width=1, marker_color=ACTION_COLOR[a], name=a, hoverinfo="x+name"))
    for col, label, y in [("b1_stop", "baseline 1 (torque RMS)", 1.25), ("b2_stop", "baseline 2 (classifier)", 1.45)]:
        t = sf.loc[sf[col], "cycle"]
        if len(t):
            fig.add_trace(go.Scatter(x=t, y=np.full(len(t), y), mode="markers", marker=dict(symbol="triangle-down", size=7, color="#333"),
                                     name=label, hoverinfo="x+name"))
    warm = injected_from(sf)
    if warm:
        fig.add_vrect(x0=-0.5, x1=warm - 0.5, fillcolor="#000", opacity=0.05, line_width=0,
                      annotation_text="warm-up", annotation_position="top left", annotation_font_size=10)
    t = ticket_info(sf, len(sf) - 1)
    if t is not None and t["planned_at"] < len(sf):
        fig.add_vline(x=t["planned_at"], line=dict(color="#2f6fd6", dash="dash"), annotation_text="planned stop", annotation_position="top right", annotation_font_size=10)
    s = first_stop(sf, len(sf) - 1)
    if s is not None:
        fig.add_annotation(x=s, y=1.0, text="stopped", showarrow=True, arrowhead=2, ax=0, ay=-25, font=dict(size=10, color="#c0392b"))
    fig.add_vline(x=float(sf["cycle"].iloc[i]), line=dict(color="#111", width=2))
    fig.update_layout(barmode="overlay", bargap=0, xaxis=dict(title="cycle", range=[-1, len(sf)]),
                      yaxis=dict(visible=False, range=[0, 1.6]), margin=dict(l=10, r=10, t=30, b=10), height=190,
                      legend=dict(orientation="h", y=1.25, font=dict(size=10)), title=dict(text="actions over the scenario", font=dict(size=14)))
    return fig


def action_shares(sf, costs=None):
    """Share of each action: as logged, and as the cheapest acute action under the given costs (no hysteresis)."""
    logged = sf["action"].value_counts(normalize=True)
    best = []
    for _, r in sf.iterrows():
        if not bool(r["sensor_ok"]):
            best.append("abstain")
            continue
        ec = expected_costs(r, costs)
        acute = {a: ec[a] for a in ["continue", "slow", "stop"]}
        best.append(min(acute, key=acute.get))
    what_if = pd.Series(best).value_counts(normalize=True)
    acts = ["continue", "slow", "schedule", "stop", "abstain"]
    return pd.DataFrame({"logged": [logged.get(a, 0.0) for a in acts], "with these costs": [what_if.get(a, 0.0) for a in acts]}, index=acts)


def shares_figure(shares):
    fig = go.Figure()
    for col, alpha in [("logged", 1.0), ("with these costs", 0.5)]:
        fig.add_trace(go.Bar(x=shares.index, y=shares[col], name=col, marker_color=[ACTION_COLOR[a] for a in shares.index],
                             opacity=alpha, text=[f"{v:.0%}" for v in shares[col]], textposition="outside"))
    fig.update_layout(barmode="group", yaxis=dict(title="share of cycles", range=[0, 1.15], tickformat=".0%"),
                      margin=dict(l=10, r=10, t=30, b=10), height=260, legend=dict(orientation="h", y=1.15),
                      title=dict(text="what-if: cheapest acute action under your costs", font=dict(size=14)))
    return fig
