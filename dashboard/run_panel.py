"""Analysis controls rendered before any replay data are required."""
from pathlib import Path
import streamlit as st
from dashboard.jobs import JobManager, PROFILES
from dashboard.twin_view import PARQUET_PATH


@st.cache_resource
def get_manager():
    return JobManager()


def render():
    st.title("Run analysis")
    st.write("Start an analysis here, then return to Dashboard to explore its results. "
             "You can keep viewing saved results while the job runs.")
    st.caption("New results are saved separately. Closing this browser tab does not stop the job; "
               "closing the local server does.")
    manager = get_manager()
    dataset = st.text_input("100 Hz dataset file", value=str(PARQUET_PATH), key="dataset_path")
    exists = bool(dataset.strip()) and Path(dataset).expanduser().is_file()
    if not exists:
        st.info("The raw dataset is not installed at this path. Existing dashboard results remain available. "
                "Download the 100 Hz Parquet file, then enter its local path here.")
        st.link_button("Download 100 Hz dataset (about 1.1 GB)",
                       "https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet")
    profile = st.selectbox("Analysis", list(PROFILES))
    runs = st.number_input("Simulated degradation runs", min_value=1, max_value=50, value=10)
    st.caption("Detection and decision replay runs steps 3 and 5 for joint 2. "
               "Full analysis also runs the six-axis exploration, uncertainty analysis and forecasting. "
               "Simulation run count is not the number of train/test split seeds.")

    @st.fragment(run_every=1.0)
    def monitor():
        state = manager.snapshot()
        busy = state["status"] in ("running", "stopping")
        a, b = st.columns(2)
        if a.button("Run analysis", type="primary", disabled=busy or not exists):
            try:
                manager.start(profile, dataset, int(runs))
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            st.rerun()
        if b.button("Stop analysis", disabled=not busy or state["status"] == "stopping"):
            manager.stop()
            st.rerun()
        if state["status"] == "idle":
            st.caption("Ready. Analysis starts only when you click Run analysis.")
            return
        st.progress(state["done"] / max(state["total"], 1),
                    text=f'{state["status"]}: {state["step"]} ({state["done"]}/{state["total"]} steps)')
        st.caption("Output folder: " + state["output"])
        if state["status"] == "completed":
            st.success("Analysis complete. Switch to Dashboard and select this run under Results.")
        elif state["status"] == "failed":
            st.error(state["error"])
        elif state["status"] == "cancelled":
            st.warning("Analysis stopped. Partial output is retained but is not offered as a completed run.")
        if state["log"]:
            st.code(state["log"], language="text", height=280)
    monitor()
