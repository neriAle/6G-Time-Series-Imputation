import pandas as pd
import os

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


def ingest_raw_csvs(csv_paths_dict, intermediate_dir):
    """
    Reads the raw CSV files, converts them to Parquet for faster I/O,
    saves them in the intermediate directory, and returns their path.
    """
    os.makedirs(intermediate_dir, exist_ok=True)
    parquet_paths = {}

    for name, path in csv_paths_dict.items():
        df = pd.read_csv(path)
        if name == "test_input":
            df["is_gap"] = df[TARGET_COLUMNS].isna().any(axis=1).astype(int)
        out_path = os.path.join(intermediate_dir, f"{name}.parquet")
        df.to_parquet(out_path)
        parquet_paths[name] = out_path

    return parquet_paths


def apply_discrete_adapter(parquet_path, intermediate_dir):
    """
    Snaps irregular time-series data to a strict 1-second grid for TimesNet and Kalman models.
    Missing seconds are left as NaN, to preserve the temporal spacing without introducing new values.
    """
    df = pd.read_parquet(parquet_path)

    # Index the Dataframe on the timestamp column
    df = df.sort_values("time")
    df = df.set_index("time")

    # Create a mathematically perfect grid from min time to max time (step=1 second)
    full_time_grid = range(df.index.min(), df.index.max() + 1)

    # Align existing data to the grid and put NaNs everywhere else
    df_discrete = df.reindex(full_time_grid)
    df_discrete = df_discrete.reset_index(names=["time"])

    # Save the discrete version
    base_name = os.path.basename(parquet_path).replace(".parquet", "_discrete.parquet")
    out_path = os.path.join(intermediate_dir, base_name)
    df_discrete.to_parquet(out_path)

    print(
        f"Discrete Adapter applied: Expanded from {len(df)} to {len(df_discrete)} rows."
    )
    return out_path
