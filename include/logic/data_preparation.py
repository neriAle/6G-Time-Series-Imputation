import pandas as pd
import os
import numpy as np
from scripts.optuna.helper import TARGET_COLUMNS


def ingest_raw_csvs(
    csv_paths_dict, intermediate_dir, mode="static", gap_parameters=None
):
    """
    Reads the raw CSV files, converts them to Parquet for faster I/O,
    saves them in the intermediate directory, and returns their paths.

    If mode="dynamic", skips the original test_input and creates multiple
    test_inputs by masking the ground truth using a list of `gap_parameters`.
    Example of gap_parameters: [(missing_ratio=0.2, block_size=5), (0.4, 10)]
    """
    if gap_parameters is None:
        gap_parameters = [(0.2, 5)]

    os.makedirs(intermediate_dir, exist_ok=True)
    parquet_paths = {}

    if mode == "static":
        for name, path in csv_paths_dict.items():
            df = pd.read_csv(path)
            if name == "test_input":
                df["is_gap"] = df[TARGET_COLUMNS].isna().any(axis=1).astype(int)
                out_path = os.path.join(intermediate_dir, f"{name}.parquet")
                # Store as a list so downstream tasks handle static/dynamic uniformly
                parquet_paths["test_inputs"] = [out_path]
            else:
                out_path = os.path.join(intermediate_dir, f"{name}.parquet")
                parquet_paths[name] = out_path
            df.to_parquet(out_path)
    else:
        parquet_paths["test_inputs"] = []

        for name, path in csv_paths_dict.items():
            if name == "test_input":
                continue
            df = pd.read_csv(path)

            if name == "test_gt":
                # Loop through every parameter combination
                for scenario in gap_parameters:
                    ratio = float(scenario[0])
                    size = int(scenario[1])

                    test_df = inject_gaps_dynamically(df, TARGET_COLUMNS, ratio, size)

                    # Create a unique filename based on the parameters
                    file_name = f"test_input_r{ratio}_s{size}.parquet"
                    out_path = os.path.join(intermediate_dir, file_name)

                    test_df.to_parquet(out_path)
                    parquet_paths["test_inputs"].append(out_path)

            # Save the original files (train_data and test_gt) normally
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
    df = df.drop_duplicates(subset=["time"], keep="last")
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


def inject_gaps_dynamically(df_gt, target_columns, missing_ratio=0.2, block_size=5):
    """
    Injects non-overlapping NaN gaps in a dataframe for all `target_columns`.
    """
    print(
        f"Injecting non-overlapping gaps: {missing_ratio * 100}% missingness, block size {block_size}"
    )

    # Reset index
    df_injected = df_gt.copy().reset_index(drop=True)
    df_injected["is_gap"] = 0

    total_rows = len(df_injected)
    n_missing_target = int(total_rows * missing_ratio)
    n_blocks = n_missing_target // block_size

    # Initialize valid starting positions
    valid_indices = list(range(0, total_rows - block_size + 1))
    start_indices = []

    np.random.seed(42)

    # Non-Overlapping Selection of blocks to mask
    for _ in range(n_blocks):
        if not valid_indices:
            print(
                "Warning: Ran out of valid space for non-overlapping blocks before reaching target ratio."
            )
            break

        start = np.random.choice(valid_indices)
        start_indices.append(start)

        # Remove this chunk and its surroundings to prevent overlapping.
        valid_indices = [
            idx
            for idx in valid_indices
            if idx <= start - block_size or idx >= start + block_size
        ]

    # Get the integer positions of the columns for .iloc
    target_col_indices = df_injected.columns.get_indexer(target_columns)
    is_gap_index = df_injected.columns.get_loc("is_gap")

    # Apply the masks
    for start_idx in start_indices:
        end_idx = start_idx + block_size

        # Inject NaNs into the target metrics
        df_injected.iloc[start_idx:end_idx, target_col_indices] = np.nan

        # Flag the rows as missing
        df_injected.iloc[start_idx:end_idx, is_gap_index] = 1

    actual_missing = df_injected["is_gap"].sum()
    print(
        f"Successfully injected {actual_missing} masked rows (Target was {n_missing_target})."
    )

    return df_injected
