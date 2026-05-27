import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


def calculate_median_mape(
    df_imputed: pd.DataFrame, df_gt: pd.DataFrame, target_columns: list
) -> float:
    """
    Evaluates the imputed dataframe against the ground truth on the masked gaps,
    returning the robust median MAPE across all target columns.
    """
    # Merge on 'time' to align rows and drop artificial discrete grids
    df_merged = pd.merge(
        df_gt[["time"] + target_columns],
        df_imputed[["time", "is_gap"] + target_columns],
        on="time",
        suffixes=("_gt", "_pred"),
    )

    # Isolate the evaluation space to only evaluate the masked rows
    df_gaps = df_merged[df_merged["is_gap"] == 1]

    # If something breaks and there are no gaps, return a terrible score
    if len(df_gaps) == 0:
        return float("inf")

    mapes = []
    for col in target_columns:
        y_true = df_gaps[f"{col}_gt"]
        y_pred = df_gaps[f"{col}_pred"]

        # Prevent Division by Zero
        denominator = np.maximum(np.abs(y_true), 1e-8)
        mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
        mapes.append(mape)

    # Return the median across all 15 columns
    return float(np.median(mapes))


def preprocess_and_reshape(
    df: pd.DataFrame, scaler: StandardScaler, n_steps: int, is_train: bool
):
    """Chunking logic preserved from your original pipeline."""
    if is_train:
        data_scaled = scaler.fit_transform(df[TARGET_COLUMNS].values)
    else:
        data_scaled = scaler.transform(df[TARGET_COLUMNS].values)

    total_len = len(data_scaled)
    remainder = total_len % n_steps
    pad_len = 0
    if remainder != 0:
        pad_len = n_steps - remainder
        padding = np.full((pad_len, len(TARGET_COLUMNS)), np.nan)
        data_scaled = np.vstack([data_scaled, padding])

    n_samples = len(data_scaled) // n_steps
    data_reshaped = data_scaled.reshape((n_samples, n_steps, len(TARGET_COLUMNS)))

    return data_reshaped, pad_len
