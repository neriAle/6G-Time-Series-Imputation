import pandas as pd
import os
import shutil
import numpy as np
from include.logic.helper import TARGET_COLUMNS


def stage_and_prepare_dataset(dataset_folder, intermediate_dir, is_pre_split):
    """
    Stages the dataset into the intermediate directory.
    If the data is not pre-split, it conforms the schema to match the pipeline's temporal
    features and telemetry targets, then performs an 80/20 temporal split.
    """
    source_dir = f"include/data/datasets/{dataset_folder}"

    # 1. If pre-split, just copy the files
    if is_pre_split:
        print(f"Loading pre-split data from {source_dir}...")
        shutil.copy(
            os.path.join(source_dir, "train.csv"),
            os.path.join(intermediate_dir, "train.csv"),
        )
        shutil.copy(
            os.path.join(source_dir, "test_input.csv"),
            os.path.join(intermediate_dir, "test_input.csv"),
        )
        shutil.copy(
            os.path.join(source_dir, "test_gt.csv"),
            os.path.join(intermediate_dir, "test_gt.csv"),
        )

        return {
            "train": os.path.join(intermediate_dir, "train.csv"),
            "test_input": os.path.join(intermediate_dir, "test_input.csv"),
            "test_gt": os.path.join(intermediate_dir, "test_gt.csv"),
        }

    # 2. If raw, align the schema, generate time features, and split it.
    print(f"Formatting and splitting raw data from {source_dir}/raw.csv...")
    df = pd.read_csv(os.path.join(source_dir, "raw.csv"))

    # Schema Alignment: AMF to target units
    if "ram_limit" in df.columns and "M" in str(df["ram_limit"].iloc[0]):
        df["ram_limit_mb"] = df["ram_limit"].str.replace("M", "").astype(float)
        df["ram_usage_mb"] = df["ram_usage"] / (1024 * 1024)

        for col in [
            "lat50",
            "lat75",
            "lat80",
            "lat90",
            "lat95",
            "lat98",
            "lat99",
            "lat100",
        ]:
            if col in df.columns:
                df[f"{col}_ms"] = df[col] / 1000.0

        # Handle missing columns expected by the pipeline
        df["c"] = 1.0
        df["lat66_ms"] = df["lat75_ms"]

    # Generate the unmasked temporal features
    df["t_norm"] = (df["time"] - df["time"].min()) / (
        df["time"].max() - df["time"].min()
    )
    seconds_in_day = 24 * 60 * 60
    time_of_day = df["time"] % seconds_in_day
    df["sin_day"] = np.sin(2 * np.pi * time_of_day / seconds_in_day)
    df["cos_day"] = np.cos(2 * np.pi * time_of_day / seconds_in_day)

    # Initialize pipeline masking columns
    df["is_gap"] = 0.0
    df["time_since_last_obs"] = 0.0
    df["time_to_next_obs"] = 0.0

    # Enforce strict column selection to match the professor's original dataset exactly
    final_columns = [
        "time",
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
        "t_norm",
        "sin_day",
        "cos_day",
        "is_gap",
        "time_since_last_obs",
        "time_to_next_obs",
    ]

    df = df[final_columns].sort_values(by="time").reset_index(drop=True)

    # Temporal 80/20 Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    if len(train_df) > 5000:
        print(
            f"Capping training data from {len(train_df)} to 5000 rows to prevent OOM."
        )
        train_df = train_df.tail(5000)
    test_gt_df = df.iloc[split_idx:]

    train_df.to_csv(os.path.join(intermediate_dir, "train.csv"), index=False)
    test_gt_df.to_csv(os.path.join(intermediate_dir, "test_gt.csv"), index=False)
    test_gt_df.copy().to_csv(
        os.path.join(intermediate_dir, "test_input.csv"), index=False
    )

    return {
        "train": os.path.join(intermediate_dir, "train.csv"),
        "test_input": os.path.join(intermediate_dir, "test_input.csv"),
        "test_gt": os.path.join(intermediate_dir, "test_gt.csv"),
    }


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


def apply_discrete_adapter(parquet_path, intermediate_dir, max_pad_seconds=60):
    """
    Snaps irregular time-series data to a strict 1-second grid for TimesNet and Kalman models.
    Prevents the 'NaN Explosion' by refusing to pad macroscopic gaps.
    """
    df = pd.read_parquet(parquet_path)

    # Sort and remove exact duplicates
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")

    times = df["time"].astype(int).values
    valid_times = []

    # Iterate through timestamps and only pad small "operational" gaps
    for i in range(len(times) - 1):
        valid_times.append(times[i])
        gap = times[i + 1] - times[i]

        # If the gap is larger than 1 second, but smaller than the 1-minute threshold, pad it with NaNs
        if 1 < gap <= max_pad_seconds:
            valid_times.extend(range(times[i] + 1, times[i + 1]))

    # Add the final timestamp
    valid_times.append(times[-1])

    # Reindex the dataframe using our smart grid
    df_discrete = df.set_index("time").reindex(valid_times).reset_index(names=["time"])

    # Save the discrete version
    base_name = os.path.basename(parquet_path).replace(".parquet", "_discrete.parquet")
    out_path = os.path.join(intermediate_dir, base_name)
    df_discrete.to_parquet(out_path)

    print(
        f"Discrete Adapter applied to {os.path.basename(parquet_path)}: Expanded from {len(df)} to {len(df_discrete)} rows."
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
