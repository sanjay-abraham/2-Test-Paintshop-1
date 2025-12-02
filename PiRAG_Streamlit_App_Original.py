
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Page config
st.set_page_config(page_title="PiRAG - Paint Shop Demo", layout="wide", initial_sidebar_state="expanded")

# Dark mode CSS (simple)
_dark_css = """
<style>
/* Background and text */
body, .stApp, .css-18e3th9 {background-color: #0e1117; color: #E6EEF3;}
.css-1d391kg {background-color: #0e1117;}
/* Sidebar */
[data-testid="stSidebar"] {background-color: #0b0f14;}
/* Cards / metrics */
.css-1v0mbdj.e1fqkh3o3 {background-color: transparent;}
/* Tables */
.st-table thead th {color: #E6EEF3;}
</style>
"""
st.markdown(_dark_css, unsafe_allow_html=True)

# Helper: load example data from bundled csv path
DEFAULT_CSV = "/mnt/data/PiRAG_Demo_PaintShop_Data.csv"

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        try:
            df = pd.read_csv(DEFAULT_CSV)
        except Exception as e:
            # generate a tiny placeholder if default CSV not found
            df = pd.DataFrame({
                "Vehicle_ID": [], "Process_Step": [], "Paint_Used_Liters": [], "Standard_Paint_Liters": [],
                "Defect_Type": [], "Rework_Count": [], "Cycle_Time_Min": [], "Oven_Temp_Variance_C": [],
                "Shift": [], "Operator_ID": [], "Equipment_ID": []
            })
    else:
        df = pd.read_csv(uploaded_file)
    return df

# Sidebar controls
st.sidebar.title("PiRAG Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
process_step = st.sidebar.selectbox("Filter: Process Step", options=["All", "Primer", "Topcoat", "Clearcoat"], index=0)
show_sample = st.sidebar.checkbox("Show sample data (first 5 rows)", value=False)
run_analysis = st.sidebar.button("Run Analysis")

# Toggle for PiRAG Chat (placeholder)
show_chat = st.sidebar.checkbox("Show PiRAG Chat", value=True)

# Load data
df = load_data(uploaded_file)

if process_step != "All" and not df.empty:
    df = df[df["Process_Step"] == process_step]

# Top-level layout: header
st.title("PiRAG — Paint Shop Demo")
st.subheader("Process Intelligence + RAG — Dark Mode MVP Skeleton")

# Main container with tabs
tabs = st.tabs(["Paint Shop Overview", "Process Map & Insights", "Ask PiRAG", "Savings & ESG", "RCA Knowledge"])

# -------------------- Paint Shop Overview Tab --------------------
with tabs[0]:
    st.markdown("### Overview KPIs")
    # Compute simple KPIs (handle empty df)
    if df.empty:
        st.info("No data available. Upload a CSV or place the demo CSV at: /mnt/data/PiRAG_Demo_PaintShop_Data.csv")
    else:
        avg_paint = df["Paint_Used_Liters"].mean()
        # Paint waste % = (Paint_Used - Standard) / Standard
        waste_percent = ((df["Paint_Used_Liters"] - df["Standard_Paint_Liters"]) / df["Standard_Paint_Liters"]).mean() * 100
        rework_rate = (df["Rework_Count"] > 0).mean() * 100
        defect_freq = (df["Defect_Type"] != "None").mean() * 100
        # Simple cost impact (assume ₹3000 per liter as placeholder)
        paint_cost_per_liter = 3000
        avg_monthly_loss = max(0, (max(0, (avg_paint - df["Standard_Paint_Liters"].mean())) * paint_cost_per_liter * 1000)) # illustrative

        # KPI cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Avg Paint Use (L)", f"{avg_paint:.2f}")
        k2.metric("Avg Paint Waste (%)", f"{waste_percent:.2f}%")
        k3.metric("Rework Rate (%)", f"{rework_rate:.2f}%")
        k4.metric("Defect Frequency (%)", f"{defect_freq:.2f}%")
        k5.metric("Est. Monthly Impact (₹)", f"₹{avg_monthly_loss:,.0f}")

        st.markdown("---")
        # Charts area
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("#### Paint usage by Process Step")
            pivot = df.groupby("Process_Step")["Paint_Used_Liters"].mean().reset_index()
            fig1, ax1 = plt.subplots(figsize=(6,3))
            ax1.bar(pivot["Process_Step"], pivot["Paint_Used_Liters"])
            ax1.set_xlabel("Process Step")
            ax1.set_ylabel("Avg Paint Used (L)")
            st.pyplot(fig1)
        with c2:
            st.markdown("#### Defect Type Distribution")
            defect_counts = df["Defect_Type"].value_counts().reset_index()
            defect_counts.columns = ["Defect", "Count"]
            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.pie(defect_counts["Count"], labels=defect_counts["Defect"], autopct='%1.1f%%')
            st.pyplot(fig2)

        st.markdown("#### Heatmap: Paint Waste by Shift × Step")
        heat_df = df.copy()
        heat_df["Waste_L"] = heat_df["Paint_Used_Liters"] - heat_df["Standard_Paint_Liters"]
        pivot_heat = heat_df.pivot_table(index="Shift", columns="Process_Step", values="Waste_L", aggfunc="mean").fillna(0)
        fig3, ax3 = plt.subplots(figsize=(6,3))
        im = ax3.imshow(pivot_heat.values, aspect='auto')
        ax3.set_xticks(np.arange(len(pivot_heat.columns)))
        ax3.set_yticks(np.arange(len(pivot_heat.index)))
        ax3.set_xticklabels(pivot_heat.columns)
        ax3.set_yticklabels(pivot_heat.index)
        ax3.set_title("Avg Waste (L)")
        st.pyplot(fig3)

        if show_sample:
            st.markdown("#### Sample data (first 5 rows)")
            st.dataframe(df.head())

# -------------------- Process Map & Insights Tab --------------------
with tabs[1]:
    st.markdown("### Process Map & Insights (PM4Py placeholder)")
    st.info("PM4Py visuals will be rendered here in the integrated version. For now, this is a placeholder.")
    st.markdown("#### Variant Summary (Top 5)")
    if not df.empty:
        # Create a mock 'trace' by grouping Vehicle_ID and joining steps if multiple rows per vehicle exist.
        # For this skeleton we will simulate variants by random selection.
        variants = {
            "Primer > Topcoat > Clearcoat": 0.62,
            "Primer > Topcoat": 0.18,
            "Primer > Rework > Topcoat > Clearcoat": 0.12,
            "Primer > Topcoat > Rework": 0.08
        }
        var_df = pd.DataFrame(list(variants.items()), columns=["Variant", "Proportion"])
        st.table(var_df)
    else:
        st.write("No data to create variants.")

    st.markdown("#### Anomaly Highlights")
    st.write("- Shift B uses +14% more paint at Topcoat (example)")
    st.write("- Robot-3 misalignment reported in maintenance logs (example)")

# -------------------- Ask PiRAG Tab --------------------
with tabs[2]:
    st.markdown("### Ask PiRAG — Conversational Interface (placeholder)")
    st.write("This chat will connect to LangChain + FAISS + LLM in the next iteration. Currently shows placeholder responses.")
    if show_chat:
        user_query = st.text_input("Ask PiRAG anything...", value="Where are we losing the most paint?")
        if st.button("Send Query"):
            # Placeholder reasoning
            st.markdown("**PiRAG:** Based on recent process data, Topcoat in Shift B shows highest paint usage. Likely causes: robot misalignment, humidity variance, or operator technique. Suggested next step: inspect Robot-3 alignment and check Zone-2 humidity logs.")
    else:
        st.info("Enable 'Show PiRAG Chat' from the sidebar to interact.")

# -------------------- Savings & ESG Tab --------------------
with tabs[3]:
    st.markdown("### Savings & ESG (Placeholder calculations)")
    if df.empty:
        st.write("No data to calculate impact.")
    else:
        total_paint_used = df["Paint_Used_Liters"].sum()
        total_standard = df["Standard_Paint_Liters"].sum()
        total_waste = total_paint_used - total_standard
        paint_cost_per_liter = 3000  # placeholder cost
        est_loss = max(0, total_waste * paint_cost_per_liter)
        st.metric("Total Paint Used (L)", f"{total_paint_used:.0f}")
        st.metric("Total Waste (L)", f"{total_waste:.2f}")
        st.metric("Estimated Loss (₹)", f"₹{est_loss:,.0f}")

        st.markdown("#### Projected Improvement Scenarios")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.write("2% Reduction")
            st.write(f"Save ≈ ₹{est_loss*0.02:,.0f} per period")
        with s2:
            st.write("3% Reduction")
            st.write(f"Save ≈ ₹{est_loss*0.03:,.0f} per period")
        with s3:
            st.write("5% Reduction")
            st.write(f"Save ≈ ₹{est_loss*0.05:,.0f} per period")

# -------------------- RCA Knowledge Tab --------------------
with tabs[4]:
    st.markdown("### RCA Knowledge Base (Documents & SOPs)")
    st.write("Upload SOPs, maintenance notes, or RCA text files to enrich PiRAG's knowledge base (FAISS embeddings in next iteration).")
    uploaded_doc = st.file_uploader("Upload SOP / RCA (txt or pdf)", type=["txt","pdf"])
    if uploaded_doc is not None:
        st.success("Uploaded (placeholder) — document will be indexed in the RAG layer in the integrated version.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("PiRAG — Streamlit MVP Skeleton (Dark Mode). Next: integrate PM4Py visuals and RAG (LangChain + FAISS).")
