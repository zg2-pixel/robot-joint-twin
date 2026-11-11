"""Dashboard of the joint twin. Run from the repo root:  streamlit run dashboard/app.py

It replays results/05_replay/replay_log.csv cycle by cycle: what the twin observed, what it concluded,
what it decided and what it did, next to a 3D view of the arm. Nothing is recomputed here except the
what-if on the cost ratios; the numbers are the ones scripts/05_replay.py wrote.
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import twin_view as tv
from twin.decide import DEFAULT_COSTS

TAU_MAX = 1.8
LEAD = 30

st.set_page_config(page_title="Robot joint twin", layout="wide")


@st.cache_data(show_spinner="reading the replay log ...")
def get_log():
    return tv.load_log()


@st.cache_resource(show_spinner="reading the joint trajectories (first start reads the data file once) ...")
def get_poses():
    return tv.load_poses()


if not tv.LOG_PATH.exists():
    st.error("results/05_replay/replay_log.csv not found. Run `python scripts/05_replay.py` first.")
    st.stop()

log = get_log()
poses, template, pose_source = get_poses()
names = tv.scenario_names(log)

# ---------- state ----------
if "scenario" not in st.session_state:
    st.session_state.scenario = "S2_run0" if "S2_run0" in names else names[0]
if "cycle" not in st.session_state:
    st.session_state.cycle = 0
if "acked" not in st.session_state:
    st.session_state.acked = set()


def reset_cycle():
    st.session_state.cycle = 0


def jump_to(name):
    st.session_state.scenario = name
    st.session_state.cycle = 0
    st.session_state.play = False


# ---------- sidebar ----------
with st.sidebar:
    st.title("Robot joint twin")
    st.caption("joint 2 (shoulder) of the six-axis cobot, voraus-AD cycles")
    st.selectbox("scenario", names, key="scenario", format_func=lambda s: tv.SCENARIOS.get(s, s), on_change=reset_cycle)
    sf = tv.scenario_frame(log, st.session_state.scenario)
    n = len(sf)

    play = st.toggle("auto play", key="play")
    step = st.select_slider("cycles per tick", options=[1, 2, 5, 10], value=2)
    if play and st.session_state.cycle < n - 1:
        st.session_state.cycle = min(st.session_state.cycle + step, n - 1)
    st.session_state.cycle = min(st.session_state.cycle, n - 1)
    st.slider("cycle", 0, n - 1, key="cycle")

    st.markdown("**inject a fault** (jumps to the recorded cycles)")
    c1, c2 = st.columns(2)
    c1.button("friction", on_click=jump_to, args=("S1_friction",), width="stretch")
    c2.button("payload", on_click=jump_to, args=("S1_payload",), width="stretch")
    c1.button("collision", on_click=jump_to, args=("S1_collision",), width="stretch")
    c2.button("electrical", on_click=jump_to, args=("S1_electrical",), width="stretch")
    c1.button("sensor drift", on_click=jump_to, args=("S4_sensor_offset",), width="stretch")
    c2.button("unseen", on_click=jump_to, args=("S5_unseen_wobble",), width="stretch")
    c1.button("back to wear run", on_click=jump_to, args=("S2_run0",), width="stretch")
    c2.button("healthy", on_click=jump_to, args=("S1_healthy",), width="stretch")

    st.markdown("**cost ratios** (in cycles of production)")
    costs = {
        "missed": st.slider("missed failure", 10, 500, int(DEFAULT_COSTS["missed"]), 10),
        "stop": st.slider("unnecessary stop", 1, 100, int(DEFAULT_COSTS["stop"]), 1),
        "sched": st.slider("planned maintenance", 1, 30, int(DEFAULT_COSTS["sched"]), 1),
        "slow": st.slider("one slow cycle", 0.0, 1.0, float(DEFAULT_COSTS["slow"]), 0.05),
    }
    st.caption(f"arm motion: {pose_source}")

# ---------- header ----------
i = int(st.session_state.cycle)
row = sf.iloc[i]
k = tv.kpis(sf, i, TAU_MAX)
colour = tv.ACTION_COLOR.get(k["action"], "#333")

st.markdown(f"### {tv.SCENARIOS.get(st.session_state.scenario, st.session_state.scenario)}")
m = st.columns(6)
m[0].metric("cycle", f"{k['cycle']} / {n - 1}", f"speed {k['speed']:.0%}" if k["speed"] < 1 else "full speed", delta_color="off")
m[1].metric("Coulomb friction", f"{k['tau_c']:.2f} N m", f"{k['z_tc']:+.1f} sigma from healthy", delta_color="inverse")
m[2].metric("P(fault now)", f"{k['p_fault']:.0%}")
nstar = "> 600" if k["n_median"] >= 600 else f"{k['n_median']:.0f}"
m[3].metric("cycles to limit (median)", nstar, f"90 %: {k['n_lo5']:.0f} to {k['n_hi95']:.0f}", delta_color="off")
m[4].metric("sensor check", "ok" if k["sensor_ok"] else "failed", "ticket open" if k["ticket_open"] else "no ticket", delta_color="off")
m[5].markdown(
    f"<div style='border-radius:8px;padding:10px 14px;background:{colour};color:white;font-size:22px;font-weight:600;text-align:center'>"
    f"{k['action'].upper()}</div><div style='font-size:12px;color:#555;padding-top:4px'>{tv.ACTION_TEXT.get(k['action'], '')}</div>",
    unsafe_allow_html=True,
)
truth = []
if k["sample"] is not None:
    truth.append(f"recorded cycle #{k['sample']}, labelled {k['label']}" + (f" ({k['variant']})" if k["variant"] else ""))
else:
    truth.append("simulated cycle" + (f", true tau_c {row['true_tc']:.2f} N m" if row.get("true_tc") == row.get("true_tc") else ""))
if k["injected_from"] is not None:
    truth.append(f"cycles 0 to {k['injected_from'] - 1} are the healthy warm-up, the injected cycles start at {k['injected_from']}")
if k["note"]:
    truth.append(f"note: {k['note']}")
st.caption(" | ".join(truth))

# once the robot left service the log keeps going, to show what would have happened
if k["out"] is not None and k["out"]["cycle"] < k["cycle"]:
    tail = "Cycles after that show what the twin would have seen had the joint kept running."
    if k["out"]["planned"]:
        st.info(f"the planned maintenance took place at cycle {k['out']['cycle']}. {tail}")
    else:
        st.error(f"the robot was stopped at cycle {k['out']['cycle']} (unplanned). {tail}")

# a ticket needs a human to take it
if k["ticket"] is not None:
    tk = st.session_state.scenario
    t = k["ticket"]
    due = t["planned_at"] - k["cycle"]
    when = f"planned stop at cycle {t['planned_at']} " + (f"(in {due} cycles)" if due > 0 else "(due now)" if due == 0 else f"({-due} cycles ago)")
    if tk in st.session_state.acked:
        st.success(f"maintenance ticket opened at cycle {t['opened']}, acknowledged, {when}")
    else:
        a1, a2 = st.columns([5, 1])
        a1.warning(f"maintenance ticket opened at cycle {t['opened']}, {when}. The date is the lower bound of the 90 % interval, never inside the lead time")
        if a2.button("acknowledge", width="stretch"):
            st.session_state.acked.add(tk)
            st.rerun()

# ---------- main panels ----------
left, right = st.columns([1.05, 1])
with left:
    q = tv.angles_for(row, poses, template)
    st.plotly_chart(tv.arm_figure(q, k["z_tc"], k["tau_c"], TAU_MAX, k["action"], speed=k["speed"]), width="stretch",
                    key=f"arm_{st.session_state.scenario}_{i}")
with right:
    st.plotly_chart(tv.health_figure(sf, i, TAU_MAX), width="stretch")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.plotly_chart(tv.forecast_figure(sf, i, TAU_MAX, LEAD), width="stretch")
    if k["true_remaining"] is not None:
        st.caption(f"simulation truth: the limit is crossed in {k['true_remaining']} cycles" if k["true_remaining"] > 0 else "simulation truth: the limit has been crossed")
with c2:
    st.plotly_chart(tv.class_figure(row), width="stretch")
with c3:
    ec = tv.expected_costs(row, costs)
    st.plotly_chart(tv.cost_figure(ec, k["action"]), width="stretch")

st.plotly_chart(tv.timeline_figure(sf, i), width="stretch")

with st.expander("what-if: how often would the twin stop with these cost ratios?"):
    st.plotly_chart(tv.shares_figure(tv.action_shares(sf, costs)), width="stretch")
    st.caption("Only the acute rule (continue / slow / stop) is recomputed, without the two-vote hysteresis; the ticket rule depends on the forecast, not on the costs.")

with st.expander("log of the last cycles"):
    cols = ["cycle", "action", "note", "tau_c", "ref_tc", "z_tc", "p_fault", "n_median", "n_lo5", "n_hi95", "p_cross_window",
            "p_healthy", "p_friction", "p_payload", "p_collision", "p_electrical", "unknown_flag", "speed", "ticket_open", "label"]
    st.dataframe(sf.iloc[max(0, i - 11): i + 1][cols].iloc[::-1], width="stretch", hide_index=True)
    st.download_button("download this scenario as csv", sf.to_csv(index=False).encode(), f"{st.session_state.scenario}.csv", "text/csv")

with st.expander("how to read this"):
    st.markdown(
        """
- **Coulomb friction tau_c** is the health indicator: it is fitted from the torque sensor of joint 2 on every cycle.
  The green band is the healthy reference (±4 sigma); the blue line is the tracked level; the red dashed line is the limit (1.8 N m).
- **P(fault now)** combines the jump test on tau_c, the classifier (only the classes it was shown to be reliable on) and the residual test.
- **Cycles to limit N\\*** is the forecast of the trend model; the band is its 90 % interval. The 90 % interval is honest only
  once the wear has started (coverage 0.93 at mid-life, 0.48 very early).
- **Action**: the cheapest of continue / slow / stop by expected cost over the 30-cycle lead time; a ticket opens once the forecast
  gives at least 15 % probability of crossing within 50 cycles; a stop needs two consecutive votes; a failed sensor check means abstain.
- The arm's **joint 2 sphere** is green / amber / red for |z| < 2 / < 4 / larger; its size grows with tau_c.
  The motion is the recorded trajectory of that cycle (the template cycle for simulated ones); slow cycles play slower.
        """
    )

if play and st.session_state.cycle < n - 1:
    time.sleep(0.6)
    st.rerun()
