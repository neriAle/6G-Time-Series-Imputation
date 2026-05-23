import os
import json
import time
import pandas as pd
from darts import TimeSeries
from darts.models import KalmanFilter

# Columns that have to be imputed
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


def impute_kalman_filter(
    discrete_train_path: str, discrete_test_path: str, output_dir: str
) -> str:
    """
    Fits a Kalman Filter on the training set, then imputes missing values on the test set.
    """
    print(f"Loading train data: {discrete_train_path}")
    df_train = pd.read_parquet(discrete_train_path)

    print(f"Loading test data: {discrete_test_path}")
    df_test = pd.read_parquet(discrete_test_path)

    # Create a copy to store the predictions without altering the intact columns
    df_imputed = df_test.copy()

    # Temporarily interpolate training data, to remove NaNs so the Darts N4SID algorithm doesn't crash.
    df_train_clean = df_train.interpolate(method="linear").ffill().bfill()

    total_fit_time = 0.0
    total_predict_time = 0.0

    for col in TARGET_COLUMNS:
        print(f"Imputing column: {col}")

        # Convert into Darts TimeSeries objects
        train_ts = TimeSeries.from_dataframe(df_train_clean, value_cols=[col])
        test_ts = TimeSeries.from_dataframe(df_test, value_cols=[col])

        # Fit (Train) the model to learn the properties of the column
        start_fit = time.perf_counter()
        kf = KalmanFilter(dim_x=1)
        kf.fit(train_ts[-1000:])
        total_fit_time += time.perf_counter() - start_fit

        # Run the filter on the test set
        start_predict = time.perf_counter()
        filtered_ts = kf.filter(test_ts)
        total_predict_time += time.perf_counter() - start_predict

        # Extract the imputed values back into the dataframe
        df_imputed[col] = filtered_ts.values().flatten()

    # Save the fully imputed dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "kalman_output.parquet")
    df_imputed.to_parquet(output_path)

    timing_data = {
        "fit_time_seconds": total_fit_time,
        "predict_time_seconds": total_predict_time,
        "total_algorithmic_time": total_fit_time + total_predict_time,
    }

    timing_path = os.path.join(output_dir, "kalman_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=4)

    print(f"Kalman imputation complete. Saved to: {output_path}")
    return output_path


def impute_nearest(discrete_test_path: str, output_dir: str) -> str:
    """
    Imputes missing values using Nearest-Neighbor interpolation (0-order hold).
    Requires the train_path in the signature to match Airflow DAG templates, but does not use it.
    """
    print(f"Loading test data: {discrete_test_path}")
    df_test = pd.read_parquet(discrete_test_path)

    df_imputed = df_test.copy()

    # Initialize Cumulative Timers
    total_fit_time = 0.0
    total_predict_time = 0.0

    for col in TARGET_COLUMNS:
        print(f"Imputing column: {col}")

        start_predict = time.perf_counter()

        # 1. Apply nearest neighbor interpolation
        # 2. Chain ffill() and bfill() to catch any NaNs that exist at the absolute first or last row
        df_imputed[col] = df_imputed[col].interpolate(method="nearest").ffill().bfill()

        total_predict_time += time.perf_counter() - start_predict

    # Save the fully imputed dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nearest_output.parquet")
    df_imputed.to_parquet(output_path)

    # Save the timing data
    timing_data = {
        "fit_time_seconds": total_fit_time,
        "predict_time_seconds": total_predict_time,
        "total_algorithmic_time": total_fit_time + total_predict_time,
    }

    timing_path = os.path.join(output_dir, "nearest_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=4)

    print(f"Nearest imputation complete. Saved to: {output_path}")
    print(f"Total Algorithmic Time: {total_predict_time:.6f}s")

    return output_path
