import optuna
import pandas as pd
from helper import calculate_median_mape, preprocess_and_reshape, TARGET_COLUMNS
from sklearn.preprocessing import StandardScaler
from pypots.imputation import BRITS, CSDI
from pypots.optim import Adam


def objective_brits(trial):
    print(f"\n--- Starting BRITS Trial {trial.number} ---")

    # 1. Suggest Hyperparameters
    rnn_hidden_size = trial.suggest_categorical("rnn_hidden_size", [64, 128, 256, 512])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # 2. Load and Prepare Data
    df_train = pd.read_parquet("include/data/intermediate/train.parquet")
    df_val = pd.read_parquet("include/data/intermediate/test_input_r0.25_s5.parquet")
    df_gt = pd.read_csv("include/data/1/test_gt.csv")

    scaler = StandardScaler()
    n_steps = 300
    X_train, _ = preprocess_and_reshape(df_train, scaler, n_steps, is_train=True)
    X_val, test_pad_len = preprocess_and_reshape(
        df_val, scaler, n_steps, is_train=False
    )

    # 3. Initialize and Train
    model = BRITS(
        n_steps=n_steps,
        n_features=len(TARGET_COLUMNS),
        rnn_hidden_size=rnn_hidden_size,
        optimizer=Adam(lr=lr),
        epochs=15,
        batch_size=32,
    )

    model.fit({"X": X_train})

    # 4. Predict and Score
    imputed_output = model.predict({"X": X_val})
    imputed_data = (
        imputed_output["imputation"]
        if isinstance(imputed_output, dict)
        else imputed_output
    )

    imputed_flat = imputed_data.reshape(-1, len(TARGET_COLUMNS))
    if test_pad_len > 0:
        imputed_flat = imputed_flat[:-test_pad_len]

    imputed_final = scaler.inverse_transform(imputed_flat)
    df_imputed = df_val.copy()
    df_imputed[TARGET_COLUMNS] = imputed_final

    return calculate_median_mape(df_imputed, df_gt, TARGET_COLUMNS)


def objective_csdi(trial):
    print(f"\n--- Starting CSDI Trial {trial.number} ---")

    # 1. Suggest Hyperparameters
    n_layers = trial.suggest_int("n_layers", 2, 6)
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    n_channels = trial.suggest_categorical("n_channels", [16, 32, 64])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # 2. Load and Prepare Data
    df_train = pd.read_parquet("include/data/intermediate/train.parquet")
    df_val = pd.read_parquet("include/data/intermediate/test_input_r0.25_s5.parquet")
    df_gt = pd.read_csv("include/data/1/test_gt.csv")

    scaler = StandardScaler()
    n_steps = 300
    X_train, _ = preprocess_and_reshape(df_train, scaler, n_steps, is_train=True)
    X_val, test_pad_len = preprocess_and_reshape(
        df_val, scaler, n_steps, is_train=False
    )

    # 3. Initialize and Train
    model = CSDI(
        n_steps=n_steps,
        n_features=len(TARGET_COLUMNS),
        n_layers=n_layers,
        n_heads=n_heads,
        n_channels=n_channels,
        d_time_embedding=64,
        d_feature_embedding=64,
        d_diffusion_embedding=64,
        optimizer=Adam(lr=lr),
        target_strategy="random",
        batch_size=8,
        epochs=15,
    )

    model.fit({"X": X_train})

    # 4. Predict and Score
    imputed_output = model.predict({"X": X_val})
    imputed_data = (
        imputed_output["imputation"]
        if isinstance(imputed_output, dict)
        else imputed_output
    )

    imputed_flat = imputed_data.reshape(-1, len(TARGET_COLUMNS))
    if test_pad_len > 0:
        imputed_flat = imputed_flat[:-test_pad_len]

    imputed_final = scaler.inverse_transform(imputed_flat)
    df_imputed = df_val.copy()
    df_imputed[TARGET_COLUMNS] = imputed_final

    return calculate_median_mape(df_imputed, df_gt, TARGET_COLUMNS)


if __name__ == "__main__":
    print("Starting PyPOTS Optimization...")

    # Run BRITS Study
    study_brits = optuna.create_study(direction="minimize", study_name="BRITS_Tuning")
    study_brits.optimize(objective_brits, n_trials=15)

    # Run CSDI Study
    study_csdi = optuna.create_study(direction="minimize", study_name="CSDI_Tuning")
    study_csdi.optimize(objective_csdi, n_trials=15)

    print("\n==================================")
    print("BEST HYPERPARAMETERS FOUND:")
    print("==================================")
    print(f"BRITS (MAPE: {study_brits.best_value:.2f}%):")
    print(study_brits.best_params)
    print(f"\nCSDI (MAPE: {study_csdi.best_value:.2f}%):")
    print(study_csdi.best_params)
