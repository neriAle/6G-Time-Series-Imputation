import os
import json
import time
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.imputation import BRITS, CSDI
from pypots.optim import Adam
from include.logic.helper import TARGET_COLUMNS

CONFIG_PATH = "include/model_configs.json"
SAVED_MODELS_DIR = "include/saved_models"


def preprocess_and_reshape(
    df: pd.DataFrame, scaler: StandardScaler, n_steps: int, is_train: bool
):
    """Scales data and pads it into strict 3D tensors for PyTorch."""
    if is_train:
        data_scaled = scaler.fit_transform(df[TARGET_COLUMNS].values)
    else:
        data_scaled = scaler.transform(df[TARGET_COLUMNS].values)

    # Pad the sequence with NaNs so it divides perfectly by n_steps
    total_len = len(data_scaled)
    remainder = total_len % n_steps
    pad_len = 0
    if remainder != 0:
        pad_len = n_steps - remainder
        padding = np.full((pad_len, len(TARGET_COLUMNS)), np.nan)
        data_scaled = np.vstack([data_scaled, padding])

    # Reshape to [n_samples, n_steps, n_features]
    n_samples = len(data_scaled) // n_steps
    data_reshaped = data_scaled.reshape((n_samples, n_steps, len(TARGET_COLUMNS)))

    return data_reshaped, pad_len


def impute_pypots_model(
    continuous_train_path: str,
    continuous_test_paths: list,
    model_type: str,
    output_dir: str,
) -> str:
    print(f"Loading continuous train: {continuous_train_path}")
    df_train = pd.read_parquet(continuous_train_path)

    scaler = StandardScaler()
    n_steps = 300
    n_features = len(TARGET_COLUMNS)

    # 1. Prepare Tensors
    X_train, _ = preprocess_and_reshape(df_train, scaler, n_steps, is_train=True)

    # 2. Load Configurations (if they exist)
    configs = {}
    if os.path.exists(CONFIG_PATH):
        print(f"Loading hyperparameters from {CONFIG_PATH}")
        with open(CONFIG_PATH, "r") as f:
            configs = json.load(f)

    model_cfg = configs.get(model_type.upper(), {})

    # 3. Initialize Model with Configs
    if model_type == "BRITS":
        model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=model_cfg.get("rnn_hidden_size", 256),
            optimizer=Adam(lr=model_cfg.get("learning_rate", 0.001)),
            epochs=100,
        )
    elif model_type == "CSDI":
        model = CSDI(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=model_cfg.get("n_layers", 4),
            n_heads=model_cfg.get("n_heads", 4),
            n_channels=model_cfg.get("n_channels", 16),
            d_time_embedding=64,
            d_feature_embedding=64,
            d_diffusion_embedding=64,
            optimizer=Adam(lr=model_cfg.get("learning_rate", 0.001)),
            batch_size=8,
            epochs=100,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 4. Check for Cached pre-trained Model
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(
        SAVED_MODELS_DIR, f"{model_type.lower()}_trained.pypots"
    )
    timing_save_path = os.path.join(
        SAVED_MODELS_DIR, f"{model_type.lower()}_train_timing.json"
    )

    if os.path.exists(model_save_path) and os.path.exists(timing_save_path):
        print(f"Found pre-trained {model_type} model! Loading from cache...")

        # Load the physical model state into memory
        model.load(model_save_path)

        # Load the historical training time
        with open(timing_save_path, "r") as f:
            train_time = json.load(f)["train_time_seconds"]

    else:
        # If no pre-trained model is found in cache, train it and save it
        print(f"No cached model found. Training {model_type} from scratch...")
        start_train = time.perf_counter()
        model.fit({"X": X_train})
        train_time = time.perf_counter() - start_train

        # Save the model state and the training time for future runs
        model.save(model_save_path)
        with open(timing_save_path, "w") as f:
            json.dump({"train_time_seconds": train_time}, f, indent=4)
        print(f"Saved trained {model_type} model to {model_save_path}")

    # 5. Imputation
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for test_path in continuous_test_paths:
        print(f"\nProcessing {model_type} test file: {test_path}")

        # Extract tag (e.g., r0.4_s10)
        match = re.search(r"(r\d+\.\d+_s\d+)", os.path.basename(test_path))
        param_tag = match.group(1) if match else "static"

        df_test = pd.read_parquet(test_path)

        # Prepare specific testing tensor
        X_test, test_pad_len = preprocess_and_reshape(
            df_test, scaler, n_steps, is_train=False
        )

        # Predict
        start_predict = time.perf_counter()
        imputed_output = model.predict({"X": X_test})
        predict_time = time.perf_counter() - start_predict

        # Extract the numpy array
        if isinstance(imputed_output, dict):
            imputed_data = imputed_output["imputation"]
        else:
            imputed_data = imputed_output

        # 6. Flatten, Truncate Padding, and Inverse Scale
        imputed_flat = imputed_data.reshape(-1, n_features)
        if test_pad_len > 0:
            imputed_flat = imputed_flat[:-test_pad_len]

        imputed_final = scaler.inverse_transform(imputed_flat)

        # 7. Save Outputs uniquely based on the gap parameters
        df_imputed = df_test.copy()
        df_imputed[TARGET_COLUMNS] = imputed_final

        out_parquet = os.path.join(
            output_dir, f"{model_type.lower()}_{param_tag}_output.parquet"
        )
        df_imputed.to_parquet(out_parquet)
        output_files.append(out_parquet)

        timing_data = {
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
            "total_algorithmic_time": train_time + predict_time,
        }

        timing_path = os.path.join(
            output_dir, f"{model_type.lower()}_{param_tag}_timing.json"
        )
        with open(timing_path, "w") as f:
            json.dump(timing_data, f, indent=4)

        print(f"Saved: {out_parquet}")

    return output_files
