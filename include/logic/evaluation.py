import os
import json
import re
import pandas as pd
import numpy as np

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


def evaluate_model(
    predictions_paths: list, gt_path: str, model_name: str, output_dir: str
) -> str:
    """
    Evaluates a list of model predictions against the ground truth strictly on the masked gaps.
    """
    print(f"Loading Ground Truth for {model_name} evaluation...")
    df_gt = pd.read_csv(gt_path)

    os.makedirs(output_dir, exist_ok=True)
    results_files = []

    for pred_path in predictions_paths:
        # Extract the gap parameters from the filename (e.g., "r0.4_s10")
        match = re.search(r"(r\d+\.\d+_s\d+)", os.path.basename(pred_path))
        param_tag = match.group(1) if match else "static"

        print(f"Evaluating {model_name} on scenario {param_tag}...")
        df_pred = pd.read_parquet(pred_path)

        # Merging on 'time' drops the artificial 1-second grid rows from the discrete datasets.
        # In this way we can evaluate the predictions only on the masked gaps
        df_merged = pd.merge(
            df_gt[["time"] + TARGET_COLUMNS],
            df_pred[["time", "is_gap"] + TARGET_COLUMNS],
            on="time",
            suffixes=("_gt", "_pred"),
        )

        # 3. Isolate the Masked Rows (The Evaluation Space)
        df_gaps = df_merged[df_merged["is_gap"] == 1]

        if len(df_gaps) == 0:
            print(f"Warning: No masked rows found for {pred_path}. Skipping.")
            continue

        print(f"Found {len(df_gaps)} masked rows for evaluation.")

        # 4. Calculate Metrics
        results = {}
        for col in TARGET_COLUMNS:
            y_true = df_gaps[f"{col}_gt"]
            y_pred = df_gaps[f"{col}_pred"]

            # RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

            # MAPE (Mean Absolute Percentage Error)
            # We bound the denominator strictly away from zero, to prevent Division by Zero errors
            denominator = np.maximum(np.abs(y_true), 1e-8)
            mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100

            results[col] = {"RMSE": float(rmse), "MAPE": float(mape)}

        # 5. Calculate Global Score (Average MAPE across all 15 columns)
        global_mape = np.mean([res["MAPE"] for res in results.values()])
        results["GLOBAL_AVERAGE_MAPE"] = float(global_mape)

        # 6. Save to JSON
        res_file = os.path.join(
            output_dir, f"{model_name.lower()}_{param_tag}_metrics.json"
        )
        with open(res_file, "w") as f:
            json.dump(results, f, indent=4)

        results_files.append(res_file)
        print(f"[{model_name} - {param_tag}] Global Average MAPE: {global_mape:.2f}%")

    return results_files
