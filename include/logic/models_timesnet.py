import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.imputation import TimesNet

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


def create_sliding_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Transforms 2D tabular data into 3D tensors (samples, sequence_length, features)."""
    samples = []
    for i in range(len(data) - seq_len + 1):
        samples.append(data[i : i + seq_len])
    return np.array(samples)


def reconstruct_2d_from_windows(
    windows: np.ndarray, original_length: int, seq_len: int
) -> np.ndarray:
    """Reconstructs the continuous 2D array from PyPOTS 3D overlapping window outputs."""
    res_2d = np.zeros((original_length, windows.shape[2]))
    # Take the entire first window
    res_2d[:seq_len] = windows[0]
    # For all subsequent windows, append the final step to reconstruct the timeline
    for i in range(1, len(windows)):
        res_2d[seq_len + i - 1] = windows[i, -1, :]
    return res_2d


def impute_timesnet(
    discrete_train_path: str, discrete_test_path: str, output_dir: str
) -> str:
    print(f"Loading training data: {discrete_train_path}")
    df_train = pd.read_parquet(discrete_train_path)

    print(f"Loading testing data: {discrete_test_path}")
    df_test = pd.read_parquet(discrete_test_path)

    # 1. Scale the data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train[TARGET_COLUMNS])
    test_scaled = scaler.transform(df_test[TARGET_COLUMNS])

    # 2. Convert to 3D for Convolution processing
    SEQ_LEN = 30
    X_train = create_sliding_windows(train_scaled, SEQ_LEN)
    X_test = create_sliding_windows(test_scaled, SEQ_LEN)

    dataset_for_training = {"X": X_train}
    dataset_for_testing = {"X": X_test}

    # 3. Initialize TimesNet Architecture
    timesnet = TimesNet(
        n_steps=SEQ_LEN,
        n_features=len(TARGET_COLUMNS),
        n_layers=2,
        top_k=3,
        d_model=64,
        d_ffn=64,
        n_kernels=3,
        dropout=0.1,
        epochs=15,
        batch_size=32,
    )

    # 4. Training
    print("Training TimesNet...")
    start_fit = time.perf_counter()
    timesnet.fit(dataset_for_training)
    total_fit_time = time.perf_counter() - start_fit

    # 5. Imputation
    print("Executing TimesNet Imputation...")
    start_predict = time.perf_counter()
    imputation_outputs = timesnet.impute(dataset_for_testing)
    total_predict_time = time.perf_counter() - start_predict

    # 6. Reconstruct and unscale
    imputed_2d_scaled = reconstruct_2d_from_windows(
        imputation_outputs, len(df_test), SEQ_LEN
    )
    imputed_2d_final = scaler.inverse_transform(imputed_2d_scaled)

    df_imputed = df_test.copy()
    df_imputed[TARGET_COLUMNS] = imputed_2d_final

    # 7. Save outputs and timing
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "timesnet_output.parquet")
    df_imputed.to_parquet(output_path)

    timing_data = {
        "fit_time_seconds": total_fit_time,
        "predict_time_seconds": total_predict_time,
        "total_algorithmic_time": total_fit_time + total_predict_time,
    }

    timing_path = os.path.join(output_dir, "timesnet_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=4)

    print(f"TimesNet output saved to: {output_path}")
    print(f"Algorithmic Predict Time: {total_predict_time:.6f}s")

    return output_path
