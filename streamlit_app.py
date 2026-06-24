import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="6G Imputation Dashboard", page_icon="📶", layout="wide")

RESULTS_DIR = "include/data/results"
MODEL_COLORS = {
    "BRITS": "#1f77b4",
    "CSDI": "#d62728",
    "TIMESNET": "#2ca02c",
    "KALMAN": "#9467bd",
    "NEAREST": "#ff7f0e",
}

# --- DYNAMIC DATASET DISCOVERY ---
try:
    available_datasets = [
        d
        for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ]
    available_datasets.sort()
except FileNotFoundError:
    st.error(f"Results directory not found at {RESULTS_DIR}. Please run the pipeline.")
    st.stop()

if not available_datasets:
    st.error("No dataset folders found inside the results directory.")
    st.stop()

st.sidebar.header("Dataset Selection")
selected_dataset = st.sidebar.selectbox("Choose Dataset", available_datasets)
st.sidebar.divider()


# --- DATA LOADING ---
@st.cache_data
def load_data(dataset_name):
    path = os.path.join(RESULTS_DIR, dataset_name, "streamlit_dataset.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Ensure model names match the uppercase dictionary keys
        if "Model" in df.columns:
            df["Model"] = df["Model"].str.upper()
        return df
    return pd.DataFrame()


df = load_data(selected_dataset)

if df.empty:
    st.warning(
        f"No 'streamlit_dataset.csv' found in the '{selected_dataset}' folder yet. Run the pipeline for this dataset!"
    )
    st.stop()

# Identify the exact column name for Gap Size
gap_col = "Gap_Size" if "Gap_Size" in df.columns else "gap_size"

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Settings")

# 1. Select the Target Metric
target_cols = df["Target_Column"].unique()
selected_col = st.sidebar.selectbox("Select Telemetry Metric", target_cols, index=0)

# 2. Select the Missing Ratio
ratios = df["Gap_Ratio"].dropna().unique()
ratios.sort()
selected_ratio = st.sidebar.selectbox("Select Missing Ratio", ["All"] + list(ratios))

# 3. Select the Gap Size
sizes = df[gap_col].dropna().unique()
sizes.sort()
sizes = [int(s) for s in sizes]
selected_size = st.sidebar.selectbox("Select Gap Size (Seconds)", ["All"] + list(sizes))

# --- FILTER LOGIC ---
filtered_df = df[df["Target_Column"] == selected_col]

if selected_ratio != "All":
    filtered_df = filtered_df[filtered_df["Gap_Ratio"] == selected_ratio]

if selected_size != "All":
    filtered_df = filtered_df[filtered_df[gap_col] == selected_size]

# --- MAIN DASHBOARD ---
st.title(f"📶 6G Telemetry Imputation ({selected_dataset.upper()})")
st.markdown(
    "Compare the accuracy and latency of deep learning and baseline models under various data loss scenarios."
)

# Handle empty data gracefully (just in case a filter combo doesn't exist)
if filtered_df.empty:
    st.warning(
        "No data matches this combination of filters. Please adjust the sidebar."
    )
    st.stop()

# --- KPI METRICS ---
st.subheader(f"Average Performance for: {selected_col}")
col1, col2, col3 = st.columns(3)

best_rmse = filtered_df["RMSE"].min()
best_rmse_model = filtered_df[filtered_df["RMSE"] == best_rmse]["Model"].iloc[0]

fastest_time = filtered_df["Latency_Seconds"].min()
fastest_model = filtered_df[filtered_df["Latency_Seconds"] == fastest_time][
    "Model"
].iloc[0]

col1.metric(
    "Lowest Average Error (RMSE)", f"{best_rmse:.2f}", f"Best: {best_rmse_model}"
)
col2.metric("Fastest Algorithm", f"{fastest_time:.3f}s", f"Best: {fastest_model}")
col3.metric(
    "Scenarios Evaluated",
    len(filtered_df["Scenario_Tag"].unique())
    if "Scenario_Tag" in filtered_df.columns
    else "N/A",
)

st.divider()

# --- CHARTS ---
# Top Row: Line Plot and Box Plot
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Predictive Degradation")
    st.markdown("RMSE vs. Gap Size")

    # Aggregate data for the line plot
    agg_line_df = filtered_df.groupby(["Model", gap_col])["RMSE"].mean().reset_index()

    fig_line = px.line(
        agg_line_df,
        x=gap_col,
        y="RMSE",
        color="Model",
        markers=True,
        color_discrete_map=MODEL_COLORS,
        log_y=True,
        labels={gap_col: "Simulated Gap Size (Seconds)", "RMSE": "RMSE (Log Scale)"},
    )
    fig_line.update_traces(line=dict(width=3.5), marker=dict(size=10))
    st.plotly_chart(fig_line, use_container_width=True)

with col_right:
    st.subheader("Error Variance")
    st.markdown("RMSE Distribution Across Scenarios")

    fig_box = px.box(
        filtered_df,
        x="Model",
        y="RMSE",
        color="Model",
        color_discrete_map=MODEL_COLORS,
        log_y=True,
        labels={"RMSE": "RMSE (Log Scale)"},
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# --- Bottom Row: Pareto Frontier ---
st.divider()
st.subheader("The Pareto Frontier")
st.markdown("Trade-off between Inference Latency and Imputation Accuracy.")

# Aggregate data for the Pareto scatter plot
agg_scatter_df = (
    filtered_df.groupby("Model")[["Latency_Seconds", "RMSE"]].mean().reset_index()
)

# Smart Y-Axis logic
is_log_y = agg_scatter_df["RMSE"].max() > 50

# --- CALCULATE PARETO FRONTIER ---
# 1. Sort by latency (fastest to slowest)
sorted_df = agg_scatter_df.sort_values("Latency_Seconds")

# 2. Iterate and keep only points that improve (lower) the RMSE
pareto_front = []
current_min_rmse = float("inf")

for _, row in sorted_df.iterrows():
    if row["RMSE"] < current_min_rmse:
        pareto_front.append(row)
        current_min_rmse = row["RMSE"]

pareto_df = pd.DataFrame(pareto_front)

# --- GENERATE SCATTER PLOT ---
fig_scatter = px.scatter(
    agg_scatter_df,
    x="Latency_Seconds",
    y="RMSE",
    color="Model",
    text="Model",
    color_discrete_map=MODEL_COLORS,
    log_x=True,
    log_y=is_log_y,
    labels={
        "Latency_Seconds": "Algorithmic Latency in Seconds (Log Scale)",
        "RMSE": f"Root Mean Square Error{' (Log Scale)' if is_log_y else ''}",
    },
)

# Style the scatter markers and text position
fig_scatter.update_traces(
    marker=dict(size=20, line=dict(width=1, color="black")),
    textposition="middle right",
    textfont=dict(size=14, color="black"),
)

# --- ADD PARETO LINE TRACE ---
fig_scatter.add_trace(
    go.Scatter(
        x=pareto_df["Latency_Seconds"],
        y=pareto_df["RMSE"],
        mode="lines",
        line=dict(color="rgba(100, 100, 100, 0.5)", width=3, dash="dash"),
        name="Pareto Frontier",
        showlegend=False,
        hoverinfo="skip",
    )
)

max_lat = agg_scatter_df["Latency_Seconds"].max()
if max_lat > 0:
    padded_max_log = np.log10(max_lat * 2.5)
    fig_scatter.update_xaxes(range=[None, padded_max_log])

st.plotly_chart(fig_scatter, use_container_width=True)

# --- RAW DATA VIEWER ---
st.divider()
st.subheader("Raw Data Inspector")
with st.expander("Click to view the raw tabular data"):
    st.dataframe(filtered_df, use_container_width=True)
