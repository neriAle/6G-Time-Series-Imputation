import os
import json
import time
import re
import pandas as pd
from darts import TimeSeries
from darts.models import KalmanFilter
from include.logic.helper import TARGET_COLUMNS


def impute_kalman_filter(
    discrete_train_path: str, discrete_test_paths: list, output_dir: str
):
    """
    Fits a Kalman Filter on the training set, then imputes missing values on the test set.
    """
    print(f"Loading train data: {discrete_train_path}")
    df_train = pd.read_parquet(discrete_train_path)

    # Temporarily interpolate training data, to remove NaNs so the Darts N4SID algorithm doesn't crash.
    df_train_clean = df_train.interpolate(method="linear").ffill().bfill()

    trained_models = {}
    total_fit_time = 0.0

    # Train a model on each column
    for col in TARGET_COLUMNS:
        train_ts = TimeSeries.from_dataframe(df_train_clean, value_cols=[col])

        start_fit = time.perf_counter()
        kf = KalmanFilter(dim_x=1)
        kf.fit(train_ts[-1000:])  # Preserve your trailing 1000-row logic
        total_fit_time += time.perf_counter() - start_fit

        trained_models[col] = kf

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    # Imputation
    for test_path in discrete_test_paths:
        print(f"\nProcessing Kalman test file: {test_path}")

        match = re.search(r"(r\d+\.\d+_s\d+)", os.path.basename(test_path))
        param_tag = match.group(1) if match else "static"

        df_test = pd.read_parquet(test_path)
        df_imputed = df_test.copy()

        total_predict_time = 0.0

        for col in TARGET_COLUMNS:
            test_ts = TimeSeries.from_dataframe(df_test, value_cols=[col])

            # Fetch the pre-trained model for this specific column
            kf = trained_models[col]

            start_predict = time.perf_counter()
            filtered_ts = kf.filter(test_ts)
            total_predict_time += time.perf_counter() - start_predict

            df_imputed[col] = filtered_ts.values().flatten()

        out_parquet = os.path.join(output_dir, f"kalman_{param_tag}_output.parquet")
        df_imputed.to_parquet(out_parquet)
        output_files.append(out_parquet)

        timing_data = {
            "fit_time_seconds": total_fit_time,
            "predict_time_seconds": total_predict_time,
            "total_algorithmic_time": total_fit_time + total_predict_time,
        }

        timing_path = os.path.join(output_dir, f"kalman_{param_tag}_timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing_data, f, indent=4)

        print(f"Saved: {out_parquet}")

    return output_files


def impute_nearest(discrete_test_paths: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for test_path in discrete_test_paths:
        print(f"\nProcessing Nearest test file: {test_path}")

        match = re.search(r"(r\d+\.\d+_s\d+)", os.path.basename(test_path))
        param_tag = match.group(1) if match else "static"

        df_test = pd.read_parquet(test_path)
        df_imputed = df_test.copy()

        total_fit_time = 0.0
        total_predict_time = 0.0

        for col in TARGET_COLUMNS:
            start_predict = time.perf_counter()
            df_imputed[col] = (
                df_imputed[col].interpolate(method="nearest").ffill().bfill()
            )
            total_predict_time += time.perf_counter() - start_predict

        out_parquet = os.path.join(output_dir, f"nearest_{param_tag}_output.parquet")
        df_imputed.to_parquet(out_parquet)
        output_files.append(out_parquet)

        timing_data = {
            "fit_time_seconds": total_fit_time,
            "predict_time_seconds": total_predict_time,
            "total_algorithmic_time": total_fit_time + total_predict_time,
        }

        timing_path = os.path.join(output_dir, f"nearest_{param_tag}_timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing_data, f, indent=4)

        print(f"Saved: {out_parquet}")

    return output_files
