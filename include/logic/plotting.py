import os
import json
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
MODELS = ["brits", "csdi", "timesnet", "kalman", "nearest"]
COLORS = {
    "brits": "#E63946",
    "csdi": "#F4A261",
    "timesnet": "#2A9D8F",
    "kalman": "#457B9D",
    "nearest": "#1D3557",
}
SUBSET_COLUMNS = ["cpu_limit", "ram_usage_mb", "lat100_ms"]


def get_scenario_tags(target_dir: str) -> set:
    """Finds all unique rX.X_sX tags from any files in the directory."""
    tags = set()
    for file in glob.glob(os.path.join(target_dir, "*")):
        match = re.search(r"(r\d+\.\d+_s\d+)", file)
        if match:
            tags.add(match.group(1))
    return tags if tags else {"static"}


def extract_title_from_tag(tag: str) -> str:
    """Converts 'r0.25_s10' into 'Missing Ratio: 0.25 - Gap Size: 10'."""
    if tag == "static":
        return "Static Baseline"

    # Extract the ratio (decimals) and size (integers)
    match = re.search(r"r(\d+\.\d+)_s(\d+)", tag)
    if match:
        ratio = match.group(1)
        size = match.group(2)
        return f"Missing Ratio: {ratio} - Gap Size: {size}"

    # Fallback
    return tag


def plot_time_series(imputed_dir: str, gt_path: str, output_dir: str):
    print("Generating Time-Series Plots...")
    os.makedirs(output_dir, exist_ok=True)
    df_gt = pd.read_csv(gt_path)

    # 1. Convert Unix timestamps to human-readable datetimes BEFORE merging
    df_gt["time"] = pd.to_datetime(df_gt["time"], unit="s")

    # 2. Mathematically isolate the 6 to 10 hour window
    start_time = df_gt["time"].min()
    block_start_dt = start_time + pd.Timedelta(hours=6)
    block_end_dt = start_time + pd.Timedelta(hours=10)

    scenarios = get_scenario_tags(imputed_dir)

    for tag in scenarios:
        reference_file = glob.glob(os.path.join(imputed_dir, f"*_{tag}_output.parquet"))
        if not reference_file:
            continue

        df_ref = pd.read_parquet(reference_file[0])
        df_ref["time"] = pd.to_datetime(df_ref["time"], unit="s")

        # Merge to get the full timeline with the is_gap markers
        df_merged = pd.merge(
            df_gt[["time"] + SUBSET_COLUMNS], df_ref[["time", "is_gap"]], on="time"
        )

        # 3. Slice the merged dataframe to exactly the 4-hour representative window
        zoom_window = df_merged[
            (df_merged["time"] >= block_start_dt) & (df_merged["time"] <= block_end_dt)
        ].reset_index(drop=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        title = extract_title_from_tag(tag)
        fig.suptitle(f"Imputation Reconstruction ({title})", fontweight="bold")

        for i, col in enumerate(SUBSET_COLUMNS):
            ax = axes[i]

            # Plot Ground Truth (Solid Black Line)
            ax.plot(
                zoom_window["time"],
                zoom_window[col],
                color="black",
                linewidth=1.5,
                label="Ground Truth",
                zorder=1,
            )

            gap_mask = zoom_window["is_gap"] == 1
            gap_times = zoom_window.loc[gap_mask, "time"]

            # Plot Imputed Models (Scatter Dots)
            for model in MODELS:
                model_path = os.path.join(imputed_dir, f"{model}_{tag}_output.parquet")
                if os.path.exists(model_path):
                    df_model = pd.read_parquet(model_path)
                    df_model["time"] = pd.to_datetime(df_model["time"], unit="s")

                    df_model_merged = pd.merge(
                        zoom_window[["time"]],
                        df_model[["time", col]],
                        on="time",
                        how="left",
                    )
                    imputed_values = df_model_merged.loc[gap_mask, col]

                    ax.scatter(
                        gap_times,
                        imputed_values,
                        color=COLORS[model],
                        label=model.upper(),
                        s=20,
                        alpha=0.8,
                        edgecolors="white",
                        zorder=5,
                    )

            ax.set_ylabel(col)

            # Format X-axis as (Hours:Minutes)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

            if i == 0:
                ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        plot_filename = f"timeseries_{tag}.png"
        plt.savefig(
            os.path.join(output_dir, plot_filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved Time-Series plot: {plot_filename}")


def plot_accuracy_bars(results_dir: str, output_dir: str):
    print("Generating Accuracy Grouped Bar Charts...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Fetch scenarios
    scenarios = get_scenario_tags(results_dir)

    # 2. Sorting helper
    def scenario_sort_key(tag):
        if tag == "static":
            return (-1.0, -1)

        # Extract ratio as float and size as integer
        match = re.search(r"r(\d+(?:\.\d+)?)_s(\d+)", tag)
        if match:
            ratio = float(match.group(1))
            size = int(match.group(2))
            # Sorts first by ratio, then by size
            return (ratio, size)

        return (0.0, 0)

    # 3. Use the computed order for the columns
    sorted_scenarios = sorted(list(scenarios), key=scenario_sort_key)

    data = []
    for tag in sorted_scenarios:
        for model in MODELS:
            metric_file = os.path.join(results_dir, f"{model}_{tag}_metrics.json")
            if os.path.exists(metric_file):
                with open(metric_file, "r") as f:
                    metrics = json.load(f)
                    for col in SUBSET_COLUMNS:
                        data.append(
                            {
                                "Scenario": tag,
                                "Model": model.upper(),
                                "Column": col,
                                "RMSE": metrics[col]["RMSE"],
                            }
                        )

    if not data:
        return
    df_plot = pd.DataFrame(data)

    for col in SUBSET_COLUMNS:
        plt.figure(figsize=(10, 6))
        col_data = df_plot[df_plot["Column"] == col]

        # Grouped bar chart
        ax = sns.barplot(
            data=col_data,
            x="Scenario",
            y="RMSE",
            hue="Model",
            palette=[COLORS[m.lower()] for m in MODELS],
            order=sorted_scenarios,
        )

        # Log Scale Y-Axis
        ax.set_yscale("log")
        plt.title(f"RMSE by Gap Scenario (Log Scale) - {col}", fontweight="bold")
        plt.ylabel("RMSE (Log Scale)")
        plt.xlabel("Gap Scenario (Ratio_Size)")

        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"accuracy_bars_{col}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved Accuracy plot for {col}")


def get_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Calculates the true mathematical Pareto frontier (minimizing both X and Y).
    Returns the points that form the bottom-left boundary of the dataset.
    """
    # Sort by X (Latency) ascending
    sorted_df = df.sort_values(x_col)

    pareto_front = []
    min_y = float("inf")

    for _, row in sorted_df.iterrows():
        # A point is on the Pareto front if its Y (Error) is strictly less
        # than the smallest Y seen so far as we sweep from left (fastest) to right (slowest).
        if row[y_col] < min_y:
            pareto_front.append(row)
            min_y = row[y_col]

    pareto_df = pd.DataFrame(pareto_front)
    return pareto_df


def plot_pareto_frontier(results_dir: str, imputed_dir: str, output_dir: str):
    print("Generating Robust Pareto Frontier...")
    os.makedirs(output_dir, exist_ok=True)
    scenarios = get_scenario_tags(results_dir)

    data = []
    for tag in scenarios:
        for model in MODELS:
            metric_file = os.path.join(results_dir, f"{model}_{tag}_metrics.json")
            timing_file = os.path.join(imputed_dir, f"{model}_{tag}_timing.json")

            if os.path.exists(metric_file) and os.path.exists(timing_file):
                with open(metric_file, "r") as mf, open(timing_file, "r") as tf:
                    metrics = json.load(mf)
                    timing = json.load(tf)

                    # Extract all 15 individual column MAPEs
                    column_mapes = [
                        metrics[col]["MAPE"]
                        for col in metrics
                        if col != "GLOBAL_AVERAGE_MAPE"
                        and isinstance(metrics[col], dict)
                    ]

                    # Use Median to ignore outlier columns
                    robust_mape = np.median(column_mapes) if column_mapes else 0

                    data.append(
                        {
                            "Model": model.upper(),
                            "Latency": max(timing["total_algorithmic_time"], 0.001),
                            "Robust_Median_MAPE": robust_mape,
                        }
                    )

    if not data:
        return

    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=df_plot,
        x="Latency",
        y="Robust_Median_MAPE",
        hue="Model",
        palette=[COLORS[m.lower()] for m in MODELS],
        s=120,
        alpha=0.7,
        edgecolor="white",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.title("Pareto Frontier: Latency vs. Accuracy", fontweight="bold")
    plt.xlabel("Algorithmic Latency (Seconds)")
    plt.ylabel("Median MAPE Across Columns (%)")

    # Calculate and plot the Pareto Frontier
    pareto_df = get_pareto_frontier(df_plot, "Latency", "Robust_Median_MAPE")

    # Plot the frontier line connecting the optimal points
    plt.plot(
        pareto_df["Latency"],
        pareto_df["Robust_Median_MAPE"],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        zorder=0,
        label="Optimal Pareto Frontier",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "pareto_frontier.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved Pareto plot.")
