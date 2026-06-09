import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="6G Imputation Dashboard", page_icon="📶", layout="wide")

RESULTS_DIR = "include/data/results"

# --- DYNAMIC DATASET DISCOVERY ---
# Find all subdirectories inside the results folder
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
        return pd.read_csv(path)
    return pd.DataFrame()


df = load_data(selected_dataset)

if df.empty:
    st.warning(
        f"No 'streamlit_dataset.csv' found in the '{selected_dataset}' folder yet. Run the pipeline for this dataset!"
    )
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Settings")

# 1. Select the Target Metric (e.g., cpu_usage, lat99_ms)
target_cols = df["Target_Column"].unique()
selected_col = st.sidebar.selectbox("Select Telemetry Metric", target_cols, index=0)

# 2. Select the Missing Ratio
ratios = df["Gap_Ratio"].dropna().unique()
ratios.sort()
selected_ratio = st.sidebar.selectbox("Select Missing Ratio", ["All"] + list(ratios))

# 3. Select the Gap Size
sizes = df["Gap_Size"].dropna().unique()
sizes.sort()
sizes = [int(s) for s in sizes]
selected_size = st.sidebar.selectbox("Select Gap Size (Seconds)", ["All"] + list(sizes))

# --- FILTER LOGIC ---
filtered_df = df[df["Target_Column"] == selected_col]

if selected_ratio != "All":
    filtered_df = filtered_df[filtered_df["Gap_Ratio"] == selected_ratio]

if selected_size != "All":
    filtered_df = filtered_df[filtered_df["Gap_Size"] == selected_size]

# --- MAIN DASHBOARD ---
st.title(f"📶 6G Telemetry Imputation ({selected_dataset.upper()})")
st.markdown(
    "Compare the accuracy and latency of deep learning and baseline models under various data loss scenarios."
)

# Top Row: KPI Metrics
st.subheader(f"Average Performance for: {selected_col}")
col1, col2, col3 = st.columns(3)

# Handle empty data gracefully (just in case a filter combo doesn't exist)
if filtered_df.empty:
    st.warning(
        "No data matches this combination of filters. Please adjust the sidebar."
    )
else:
    best_mape = filtered_df["MAPE"].min()
    best_mape_model = filtered_df[filtered_df["MAPE"] == best_mape]["Model"].iloc[0]

    fastest_time = filtered_df["Latency_Seconds"].min()
    fastest_model = filtered_df[filtered_df["Latency_Seconds"] == fastest_time][
        "Model"
    ].iloc[0]

    col1.metric(
        "Lowest Average Error (MAPE)", f"{best_mape:.2f}%", f"Best: {best_mape_model}"
    )
    col2.metric("Fastest Algorithm", f"{fastest_time:.3f}s", f"Best: {fastest_model}")
    col3.metric("Scenarios Evaluated", len(filtered_df["Scenario_Tag"].unique()))

    st.divider()

    # --- CHARTS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Error vs. Gap Size")
        st.markdown(
            "How does model accuracy degrade as the network outage lasts longer?"
        )

        chart_df = filtered_df.sort_values("Gap_Size")

        fig_bar = px.bar(
            chart_df,
            x="Gap_Size",
            y="MAPE",
            color="Model",
            barmode="group",
            title=f"MAPE by Gap Size ({selected_col})",
            labels={"Gap_Size": "Gap Duration (Seconds)", "MAPE": "MAPE (%)"},
        )

        # Force the X-axis to treat Gap Size as discrete categories, not a continuous timeline
        fig_bar.update_xaxes(type="category")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("The Pareto Frontier")
        st.markdown("Trade-off between Inference Latency and Imputation Accuracy.")

        fig_scatter = px.scatter(
            filtered_df,
            x="Latency_Seconds",
            y="MAPE",
            color="Model",
            size="Gap_Size" if selected_size == "All" else None,
            hover_data=["Scenario_Tag"],
            title=f"Latency vs. Accuracy ({selected_col})",
            log_x=True,
            log_y=True,
            labels={"Latency_Seconds": "Algorithmic Latency (s)", "MAPE": "MAPE (%)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- RAW DATA VIEWER ---
    st.divider()
    st.subheader("Raw Data Inspector")
    with st.expander("Click to view the raw tabular data"):
        st.dataframe(filtered_df, use_container_width=True)
