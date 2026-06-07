import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

st.set_page_config(page_title="Time Series Viewer", page_icon="📈", layout="wide")
st.title("📈 Time Series Reconstruction Viewer")
st.markdown(
    "Inspect how each model physically reconstructed the waveforms during network outages."
)

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Waveform Settings")

IMPUTED_DIR = "include/data/intermediate/imputed"
GT_PATH = "include/data/intermediate/test_gt.csv"

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
        st.error("No valid scenarios found in filenames. Check your output files.")
        st.stop()

except FileNotFoundError:
    st.error(
        f"Could not find the directory: {IMPUTED_DIR}. Have you run the pipeline yet?"
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
def load_timeseries_data(scenario, col):
    # Load Ground Truth
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
    df_plot = load_timeseries_data(selected_scenario, selected_col)

    # Isolate Ground Truth for the line plot
    gt_data = df_plot[df_plot["Model"] == "Ground Truth"]
    model_data = df_plot[df_plot["Model"] != "Ground Truth"]

    fig = px.line(
        gt_data,
        x="time",
        y=selected_col,
        color_discrete_sequence=["gray"],
        title=f"Imputation for {selected_col} ({selected_scenario})",
    )

    # Add the model predictions as scatter dots on top
    if not model_data.empty:
        fig_scatter = px.scatter(model_data, x="time", y=selected_col, color="Model")
        # Merge the scatter traces into the main figure
        for trace in fig_scatter.data:
            fig.add_trace(trace)

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
