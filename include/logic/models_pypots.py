import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.imputation import BRITS, CSDI

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
    continuous_test_path: str,
    model_type: str,
    output_dir: str,
) -> str:
    print(f"Loading continuous train: {continuous_train_path}")
    df_train = pd.read_parquet(continuous_train_path)

    print(f"Loading continuous test: {continuous_test_path}")
    df_test = pd.read_parquet(continuous_test_path)

    scaler = StandardScaler()
    n_steps = 100
    n_features = len(TARGET_COLUMNS)

    # 1. Prepare Tensors
    X_train, _ = preprocess_and_reshape(df_train, scaler, n_steps, is_train=True)
    X_test, test_pad_len = preprocess_and_reshape(
        df_test, scaler, n_steps, is_train=False
    )

    # 2. Initialize Model
    # Note: epochs=10 is set for pipeline testing. Increase to 50-100 for better accuracy.
    if model_type == "BRITS":
        model = BRITS(
            n_steps=n_steps, n_features=n_features, rnn_hidden_size=64, epochs=10
        )
    elif model_type == "CSDI":
        model = CSDI(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=2,
            n_heads=2,
            n_channels=16,
            epochs=10,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 3. Train
    print(f"Training {model_type}...")
    model.fit({"X": X_train})

    # 4. Predict
    print(f"Predicting with {model_type}...")
    imputed_output = model.predict({"X": X_test})

    # Extract the numpy array
    if isinstance(imputed_output, dict):
        imputed_data = imputed_output["imputation"]
    else:
        imputed_data = imputed_output

    # 5. Flatten, Truncate Padding, and Inverse Scale
    imputed_flat = imputed_data.reshape(-1, n_features)
    if test_pad_len > 0:
        imputed_flat = imputed_flat[:-test_pad_len]

    imputed_final = scaler.inverse_transform(imputed_flat)

    # 6. Save Outputs
    df_imputed = df_test.copy()
    df_imputed[TARGET_COLUMNS] = imputed_final

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_type.lower()}_output.parquet")
    df_imputed.to_parquet(output_path)

    print(f"{model_type} imputation complete. Saved to: {output_path}")
    return output_path
