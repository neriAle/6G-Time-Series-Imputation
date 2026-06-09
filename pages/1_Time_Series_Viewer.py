import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

st.set_page_config(page_title="Time Series Viewer", page_icon="📈", layout="wide")
st.title("📈 Time Series Reconstruction Viewer")

# Point strictly to the safe results backup folder
RESULTS_ROOT = "include/data/results"

# --- DYNAMIC DATASET DISCOVERY ---
try:
    available_datasets = [
        d
        for d in os.listdir(RESULTS_ROOT)
        if os.path.isdir(os.path.join(RESULTS_ROOT, d))
    ]
    available_datasets.sort()
except FileNotFoundError:
    st.error(f"Results directory not found at {RESULTS_ROOT}.")
    st.stop()

if not available_datasets:
    st.error("No dataset folders found inside the results directory.")
    st.stop()

st.sidebar.header("Dataset Selection")
selected_dataset = st.sidebar.selectbox("Choose Dataset", available_datasets)
st.sidebar.divider()

# Set dynamic paths based on the selected dataset inside the RESULTS folder
IMPUTED_DIR = os.path.join(RESULTS_ROOT, selected_dataset, "imputed")
GT_PATH = os.path.join(RESULTS_ROOT, selected_dataset, "test_gt.csv")

st.sidebar.header("Waveform Settings")

# Find available scenarios
try:
    files = [f for f in os.listdir(IMPUTED_DIR) if f.endswith("_output.parquet")]
    scenarios = set()
    for f in files:
        match = re.search(r"(r\d+\.\d+_s\d+)", f)
        if match:
            scenarios.add(match.group(1))

    scenarios = sorted(list(scenarios))

    if not scenarios:
        st.warning(
            f"No valid scenarios found in {IMPUTED_DIR}. Have you moved your imputed data here?"
        )
        st.stop()

except FileNotFoundError:
    st.warning(
        f"Could not find the imputed directory for '{selected_dataset}'. Expected: {IMPUTED_DIR}"
    )
    st.stop()

selected_scenario = st.sidebar.selectbox("Select Scenario (Ratio_Size)", scenarios)

TARGET_COLUMNS = [
    "cpu_limit",
    "cpu_usage",
    "n",
    "c",
    "ram_limit_mb",
    "ram_usage_mb",
    "lat50_ms",
    "lat66_ms",
    "lat75_ms",
    "lat80_ms",
    "lat90_ms",
    "lat95_ms",
    "lat98_ms",
    "lat99_ms",
    "lat100_ms",
]
selected_col = st.sidebar.selectbox("Select Telemetry Metric", TARGET_COLUMNS)


# --- DATA LOADING (CACHED) ---
@st.cache_data
def load_timeseries_data(dataset_name, scenario, col):
    if not os.path.exists(GT_PATH):
        st.error(f"Ground Truth file missing: {GT_PATH}")
        return pd.DataFrame()

    df_gt = pd.read_csv(GT_PATH)
    df_gt["time"] = pd.to_datetime(df_gt["time"], unit="s")
    combined_data = []

    # Add Ground Truth
    gt_subset = df_gt[["time", col]].copy()
    gt_subset["Model"] = "Ground Truth"
    combined_data.append(gt_subset)

    # Load Models dynamically
    models = ["timesnet", "brits", "csdi", "nearest", "kalman"]
    for model in models:
        path = os.path.join(IMPUTED_DIR, f"{model}_{scenario}_output.parquet")
        if os.path.exists(path):
            df_model = pd.read_parquet(path)
            df_model["time"] = pd.to_datetime(df_model["time"], unit="s")

            # We only care about the gap points for the models
            gaps_only = df_model[df_model["is_gap"] == 1][["time", col]].copy()
            gaps_only["Model"] = model.upper()
            combined_data.append(gaps_only)

    return pd.concat(combined_data, ignore_index=True)


# --- PLOTTING ---
with st.spinner("Loading parquet files..."):
    df_plot = load_timeseries_data(selected_dataset, selected_scenario, selected_col)

    if not df_plot.empty:
        gt_data = df_plot[df_plot["Model"] == "Ground Truth"]
        model_data = df_plot[df_plot["Model"] != "Ground Truth"]

        fig = px.line(
            gt_data,
            x="time",
            y=selected_col,
            color_discrete_sequence=["gray"],
            title=f"Imputation for {selected_col} ({selected_scenario}) - {selected_dataset.upper()}",
        )

        if not model_data.empty:
            fig_scatter = px.scatter(
                model_data, x="time", y=selected_col, color="Model"
            )
            for trace in fig_scatter.data:
                fig.add_trace(trace)

        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
