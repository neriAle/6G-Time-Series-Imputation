import os
import json
import time
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.imputation import TimesNet
from pypots.optim import Adam
from include.logic.helper import TARGET_COLUMNS

CONFIG_PATH = "include/model_configs.json"
SAVED_MODELS_DIR = "include/saved_models"


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
    discrete_train_path: str, discrete_test_paths: list, output_dir: str
) -> list:
    print(f"Loading training data: {discrete_train_path}")
    df_train = pd.read_parquet(discrete_train_path)

    # 1. Fit the Scaler strictly on the Training Data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train[TARGET_COLUMNS])

    # 2. Convert Training data to 3D for Convolution processing
    SEQ_LEN = 30
    X_train = create_sliding_windows(train_scaled, SEQ_LEN)
    dataset_for_training = {"X": X_train}

    # 3. Load Configurations (if they exist)
    configs = {}
    if os.path.exists(CONFIG_PATH):
        print(f"Loading hyperparameters from {CONFIG_PATH}")
        with open(CONFIG_PATH, "r") as f:
            configs = json.load(f)

    model_cfg = configs.get("TIMESNET", {})

    # 4. Initialize TimesNet Architecture with Configs (or Defaults)
    timesnet = TimesNet(
        n_steps=SEQ_LEN,
        n_features=len(TARGET_COLUMNS),
        n_layers=model_cfg.get("n_layers", 2),
        top_k=model_cfg.get("top_k", 3),
        d_model=model_cfg.get("d_model", 64),
        d_ffn=model_cfg.get("d_ffn", 64),
        n_kernels=model_cfg.get("n_kernels", 3),
        dropout=model_cfg.get("dropout", 0.1),
        optimizer=Adam(lr=model_cfg.get("learning_rate", 0.001)),
        epochs=15,
        batch_size=32,
    )

    # 5. Check for Cached Model & Timing
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(SAVED_MODELS_DIR, "timesnet_trained.pypots")
    timing_save_path = os.path.join(SAVED_MODELS_DIR, "timesnet_train_timing.json")

    if os.path.exists(model_save_path) and os.path.exists(timing_save_path):
        print("Found pre-trained TimesNet model! Loading from cache...")

        # Load the physical model state into memory
        timesnet.load(model_save_path)

        # Load the historical training time
        with open(timing_save_path, "r") as f:
            total_fit_time = json.load(f)["train_time_seconds"]

    else:
        print(
            "No cached model found. Training TimesNet from scratch (This may take a while)..."
        )
        start_fit = time.perf_counter()
        timesnet.fit(dataset_for_training)
        total_fit_time = time.perf_counter() - start_fit

        # Save the model state and the training time for future runs
        timesnet.save(model_save_path)
        with open(timing_save_path, "w") as f:
            json.dump({"train_time_seconds": total_fit_time}, f, indent=4)
        print(f"Saved trained TimesNet model to {model_save_path}")

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    # 6. Imputation
    for test_path in discrete_test_paths:
        print(f"\nProcessing TimesNet test file: {test_path}")

        match = re.search(r"(r\d+\.\d+_s\d+)", os.path.basename(test_path))
        param_tag = match.group(1) if match else "static"

        df_test = pd.read_parquet(test_path)

        # Scale and reshape test data using the pre-fitted scaler
        test_scaled = scaler.transform(df_test[TARGET_COLUMNS])
        X_test = create_sliding_windows(test_scaled, SEQ_LEN)
        dataset_for_testing = {"X": X_test}

        # Predict
        start_predict = time.perf_counter()

        # Support both .impute() and .predict() depending on your PyPOTS version
        if hasattr(timesnet, "predict"):
            imputed_output = timesnet.predict(dataset_for_testing)
            imputation_outputs = (
                imputed_output["imputation"]
                if isinstance(imputed_output, dict)
                else imputed_output
            )
        else:
            imputation_outputs = timesnet.impute(dataset_for_testing)

        total_predict_time = time.perf_counter() - start_predict

        # Reconstruct and unscale
        imputed_2d_scaled = reconstruct_2d_from_windows(
            imputation_outputs, len(df_test), SEQ_LEN
        )
        imputed_2d_final = scaler.inverse_transform(imputed_2d_scaled)

        df_imputed = df_test.copy()
        df_imputed[TARGET_COLUMNS] = imputed_2d_final

        # Save unique outputs and timing
        output_path = os.path.join(output_dir, f"timesnet_{param_tag}_output.parquet")
        df_imputed.to_parquet(output_path)
        output_files.append(output_path)

        timing_data = {
            "fit_time_seconds": total_fit_time,
            "predict_time_seconds": total_predict_time,
            "total_algorithmic_time": total_fit_time + total_predict_time,
        }

        timing_path = os.path.join(output_dir, f"timesnet_{param_tag}_timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing_data, f, indent=4)

        print(f"TimesNet output saved to: {output_path}")

    return output_files
