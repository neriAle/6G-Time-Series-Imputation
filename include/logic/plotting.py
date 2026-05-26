import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Thesis-grade styling
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


def plot_time_series(imputed_dir: str, gt_path: str, output_dir: str):
    print("Generating Time-Series Plots...")
    os.makedirs(output_dir, exist_ok=True)
    df_gt = pd.read_csv(gt_path)

    # 1. Convert Unix timestamps to human-readable datetimes BEFORE merging
    df_gt["time"] = pd.to_datetime(df_gt["time"], unit="s")

    scenarios = get_scenario_tags(imputed_dir)

    for tag in scenarios:
        # sharex=True cleanly ties the 3 subplots together so the time axis is only on the bottom
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Imputation Reconstruction (Scenario: {tag})", fontweight="bold")

        reference_file = glob.glob(os.path.join(imputed_dir, f"*_{tag}_output.parquet"))
        if not reference_file:
            continue

        df_ref = pd.read_parquet(reference_file[0])
        # Convert the reference time as well to match
        df_ref["time"] = pd.to_datetime(df_ref["time"], unit="s")

        df_merged = pd.merge(
            df_gt[["time"] + SUBSET_COLUMNS], df_ref[["time", "is_gap"]], on="time"
        )
        gap_indices = df_merged.index[df_merged["is_gap"] == 1].tolist()

        if not gap_indices:
            continue

        # 2. Smart Dynamic Windowing: Find the exact bounds of the FIRST gap chunk
        first_gap_start = gap_indices[0]
        first_gap_end = first_gap_start
        for idx in gap_indices[1:]:
            if idx == first_gap_end + 1:
                first_gap_end = idx
            else:
                break  # We hit the end of the first contiguous missing block

        # Add exactly 60 steps (1 minute) of visible context before and after the gap
        start_idx = max(0, first_gap_start - 60)
        end_idx = min(len(df_merged), first_gap_end + 60)

        zoom_window = df_merged.iloc[start_idx:end_idx].reset_index(drop=True)

        for i, col in enumerate(SUBSET_COLUMNS):
            ax = axes[i]

            # Plot Ground Truth (Solid Black Line)
            ax.plot(
                zoom_window["time"],
                zoom_window[col],
                color="black",
                linewidth=2,
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
                        s=40,
                        alpha=0.9,
                        edgecolors="white",
                        linewidth=0.5,
                        zorder=5,
                    )

            ax.set_ylabel(col)

            # Format X-axis as (Hours:Minutes:Seconds)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            if i == 0:
                ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

        # Rotate x-axis labels slightly so they don't overlap
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"timeseries_{tag}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved Time-Series plot for {tag}")


def plot_accuracy_bars(results_dir: str, output_dir: str):
    print("Generating Accuracy Grouped Bar Charts...")
    os.makedirs(output_dir, exist_ok=True)
    scenarios = get_scenario_tags(results_dir)

    data = []
    for tag in scenarios:
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


def plot_pareto_frontier(results_dir: str, imputed_dir: str, output_dir: str):
    print("Generating Pareto Frontier...")
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

                    data.append(
                        {
                            "Scenario": tag,
                            "Model": model.upper(),
                            "Latency": max(timing["total_algorithmic_time"], 0.001),
                            "Global_MAPE": metrics["GLOBAL_AVERAGE_MAPE"],
                        }
                    )

    if not data:
        return
    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_plot,
        x="Latency",
        y="Global_MAPE",
        hue="Model",
        style="Scenario",
        palette=[COLORS[m.lower()] for m in MODELS],
        s=150,
        alpha=0.9,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Pareto Frontier: Latency vs. Accuracy (Log-Log)", fontweight="bold")
    plt.xlabel("Algorithmic Latency (Seconds, Log Scale)")
    plt.ylabel("Global Average MAPE (%, Log Scale)")

    # Draw the optimal Pareto Curve (Bottom-Left is best)
    # Group by model to find average position to draw a rough line between Nearest and BRITS
    avg_perf = df_plot.groupby("Model")[["Latency", "Global_MAPE"]].mean().reset_index()
    optimal_models = avg_perf[avg_perf["Model"].isin(["NEAREST", "BRITS"])]
    optimal_models = optimal_models.sort_values("Latency")
    plt.plot(
        optimal_models["Latency"],
        optimal_models["Global_MAPE"],
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Pareto Frontier",
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "pareto_frontier.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved Pareto plot.")
