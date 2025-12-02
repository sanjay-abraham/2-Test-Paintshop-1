
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, os, tempfile, time
from datetime import datetime, timedelta

st.set_page_config(page_title="PiRAG - PM4Py Integrated", layout="wide", initial_sidebar_state="expanded")

# Dark mode CSS
_dark_css = """
<style>
body, .stApp, .css-18e3th9 {background-color: #0e1117; color: #E6EEF3;}
[data-testid="stSidebar"] {background-color: #0b0f14;}
.stMarkdown {color: #E6EEF3;}
</style>
"""
st.markdown(_dark_css, unsafe_allow_html=True)

DEFAULT_CSV = "/mnt/data/PiRAG_Demo_PaintShop_Data.csv"
APP_DIR = os.path.abspath(os.path.dirname(__file__))

# Load data helper
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        try:
            df = pd.read_csv(DEFAULT_CSV)
        except Exception:
            df = pd.DataFrame(columns=[
                "Vehicle_ID","Process_Step","Paint_Used_Liters","Standard_Paint_Liters",
                "Defect_Type","Rework_Count","Cycle_Time_Min","Oven_Temp_Variance_C",
                "Shift","Operator_ID","Equipment_ID"
            ])
    else:
        df = pd.read_csv(uploaded_file)
    return df

# Sidebar
st.sidebar.title("PiRAG Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
process_step = st.sidebar.selectbox("Filter: Process Step", options=["All", "Primer", "Topcoat", "Clearcoat"], index=0)
miner_choice = st.sidebar.selectbox("Mining algorithm", options=["Inductive Miner (recommended)","Heuristics Miner","Alpha Miner"])
run_pm4py = st.sidebar.button("Generate Process Map")
show_sample = st.sidebar.checkbox("Show sample data (first 5 rows)", value=False)

df = load_data(uploaded_file)

if process_step != "All" and not df.empty:
    df = df[df["Process_Step"] == process_step]

st.title("PiRAG — Process Mining (PM4Py) Integrated")
st.subheader("Process Map & Insights — Full Integration (Dark Mode)")

tabs = st.tabs(["Overview", "Process Map & Insights", "Ask PiRAG (placeholder)"])

# ---------------- Overview ----------------
with tabs[0]:
    st.markdown("### Overview KPIs")
    if df.empty:
        st.info("No data available. Upload a CSV or place the demo CSV at: /mnt/data/PiRAG_Demo_PaintShop_Data.csv")
    else:
        avg_paint = df["Paint_Used_Liters"].mean()
        waste_percent = ((df["Paint_Used_Liters"] - df["Standard_Paint_Liters"]) / df["Standard_Paint_Liters"]).mean() * 100
        rework_rate = (df["Rework_Count"] > 0).mean() * 100
        defect_freq = (df["Defect_Type"] != "None").mean() * 100
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Avg Paint Use (L)", f"{avg_paint:.2f}")
        k2.metric("Avg Paint Waste (%)", f"{waste_percent:.2f}%")
        k3.metric("Rework Rate (%)", f"{rework_rate:.2f}%")
        k4.metric("Defect Frequency (%)", f"{defect_freq:.2f}%")

        if show_sample:
            st.dataframe(df.head())

# ---------------- Process Map & Insights ----------------
with tabs[1]:
    st.markdown("### Process Map & Insights (PM4Py)")
    if df.empty:
        st.write("No data to analyze.")
    else:
        st.info("This tab requires `pm4py` to be installed in the environment. If not present, the app will show guidance.")
        # Prepare event log for PM4Py
        # Create a copy and ensure required columns exist
        work_df = df.copy()
        # Create or ensure timestamp column exists: synthesize if not present
        if "timestamp" not in work_df.columns:
            # generate synthetic timestamps per Vehicle_ID by ordering and spacing
            base = datetime.now() - timedelta(days=7)
            timestamps = []
            group_counts = work_df.groupby("Vehicle_ID").cumcount()
            for idx, cnt in enumerate(group_counts):
                # Each step spaced by 1-5 minutes
                delta = timedelta(minutes=int((idx % 5) + 1))
                timestamps.append(base + timedelta(minutes=idx) + delta)
            work_df["timestamp"] = timestamps

        # Rename columns for PM4Py compatibility
        pm_df = work_df.rename(columns={
            "Vehicle_ID":"case:concept:name",
            "Process_Step":"concept:name",
            "timestamp":"time:timestamp"
        })[["case:concept:name","concept:name","time:timestamp"]].copy()

        # Try to import pm4py and run miners; if pm4py missing, show instructions
        try:
            import pm4py
            from pm4py.objects.conversion.log import converter as log_converter
            from pm4py.objects.log.util import dataframe_utils
            from pm4py.algo.discovery.inductive import algorithm as inductive_miner
            from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
            from pm4py.algo.discovery.alpha import algorithm as alpha_miner
            from pm4py.visualization.petri_net import visualizer as pn_visualizer
            from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
            from pm4py.statistics.traces.generic.log import case_statistics as cs
        except Exception as e:
            st.error("pm4py not installed in this environment. To enable process mining, install pm4py (and graphviz) by adding to requirements.txt: pm4py graphviz")
            st.caption("Ref: https://pm4py.fit.fraunhofer.de/")
            st.stop()

        # Ensure timestamps are datetime
        pm_df["time:timestamp"] = pd.to_datetime(pm_df["time:timestamp"])

        # Convert to pm4py event log
        pm_df = dataframe_utils.convert_timestamp_columns_in_df(pm_df)
        event_log = log_converter.apply(pm_df)

        # Variant statistics (from pandas original grouping for robustness)
        traces = df.groupby("Vehicle_ID")["Process_Step"].apply(list).reset_index()
        traces["variant"] = traces["Process_Step"].apply(lambda x: " > ".join(x))
        variant_counts = traces["variant"].value_counts().reset_index()
        variant_counts.columns = ["Variant","Frequency"]
        variant_counts["Proportion"] = variant_counts["Frequency"] / variant_counts["Frequency"].sum()

        st.markdown("#### Top process variants")
        st.table(variant_counts.head(10))

        # Rework detection: count traces where any step repeats
        def detect_rework(step_list):
            return any(step_list.count(s) > 1 for s in set(step_list))

        traces["has_rework"] = traces["Process_Step"].apply(detect_rework)
        rework_rate = traces["has_rework"].mean() * 100
        st.metric("Cases with rework (%)", f"{rework_rate:.2f}%")

        # Mining and visualization
        col1, col2 = st.columns([2,1])
        with col2:
            st.markdown("##### Mining options & export")
            st.write(f"Selected miner: **{miner_choice}**")
            if st.button("Download variants CSV"):
                tmp = io.BytesIO()
                variant_counts.to_csv(tmp, index=False)
                tmp.seek(0)
                st.download_button("Download variants.csv", data=tmp, file_name="piRAG_variants.csv", mime="text/csv")
        with col1:
            st.markdown("#### Process model visualization")
            # Run selected miner
            try:
                if "Inductive" in miner_choice:
                    net, im, fm = inductive_miner.apply(event_log)
                    gviz = pn_visualizer.apply(net, im, fm)
                    tmp_png = os.path.join(tempfile.gettempdir(), "pirag_process_map.png")
                    pn_visualizer.save(gviz, tmp_png)
                    st.image(tmp_png, use_column_width=True)
                elif "Heuristics" in miner_choice:
                    heu_net = heuristics_miner.apply_heu(event_log)
                    gviz = hn_visualizer.apply(heu_net)
                    tmp_png = os.path.join(tempfile.gettempdir(), "pirag_process_map_heuristics.png")
                    hn_visualizer.save(gviz, tmp_png)
                    st.image(tmp_png, use_column_width=True)
                else:
                    # Alpha miner
                    net, im, fm = alpha_miner.apply(event_log)
                    gviz = pn_visualizer.apply(net, im, fm)
                    tmp_png = os.path.join(tempfile.gettempdir(), "pirag_process_map_alpha.png")
                    pn_visualizer.save(gviz, tmp_png)
                    st.image(tmp_png, use_column_width=True)
            except Exception as e:
                st.error(f"Error generating process map: {e}")
                st.write("Note: PM4Py visualization requires Graphviz installed in the environment. On Streamlit Cloud add graphviz and pm4py in requirements.txt.")

        st.markdown("#### Rework insights")
        # Which steps are most commonly repeated?
        def most_repeated_steps(traces_df):
            repeats = {}
            for lst in traces_df["Process_Step"]:
                seen = set()
                for s in lst:
                    repeats[s] = repeats.get(s, 0) + (1 if lst.count(s) > 1 else 0)
            return sorted([(k,v) for k,v in repeats.items()], key=lambda x: -x[1])

        # build traces DataFrame for counting
        traces_expanded = df.groupby("Vehicle_ID")["Process_Step"].apply(list).reset_index()
        traces_expanded.columns = ["Vehicle_ID","Process_Step"]
        rep = most_repeated_steps(traces_expanded)
        rep_df = pd.DataFrame(rep, columns=["Step","Repeated_In_#Cases"])
        st.table(rep_df.head(10))

        st.markdown("#### Variant distribution chart")
        fig, ax = plt.subplots(figsize=(8,2))
        top_variants = variant_counts.head(8)
        ax.barh(top_variants["Variant"][::-1], top_variants["Frequency"][::-1])
        ax.set_xlabel("Frequency")
        st.pyplot(fig)

# ---------------- Ask PiRAG placeholder ----------------
with tabs[2]:
    st.markdown("### Ask PiRAG (placeholder)")
    st.write("LangChain + FAISS integration will power this chat in the next iteration. For now, use the PM4Py outputs to analyze process behavior.")

st.caption("PiRAG — PM4Py integrated app. Next: LangChain + FAISS for RAG answers.")
