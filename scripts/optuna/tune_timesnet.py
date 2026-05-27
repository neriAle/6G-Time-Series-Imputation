import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pypots.imputation import TimesNet
from pypots.optim import Adam
from helper import (
    create_sliding_windows,
    reconstruct_2d_from_windows,
    calculate_median_mape,
    TARGET_COLUMNS,
)


# Define the Objective Function
def objective(trial):
    print(f"--- Starting Trial {trial.number} ---")

    # 1. Let Optuna "suggest" hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    top_k = trial.suggest_int("top_k", 1, 5)
    n_kernels = trial.suggest_int("n_kernels", 1, 5)
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    d_ffn = trial.suggest_categorical("d_ffn", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # 2. Load Training Data
    df_train = pd.read_parquet("include/data/intermediate/train_discrete.parquet")

    # Only consider the first 30% of data, to speed up optimization
    split_index = int(len(df_train) * 0.3)
    df_train = df_train.iloc[:split_index]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train[TARGET_COLUMNS])
    X_train = create_sliding_windows(train_scaled, 30)

    # 3. Initialize Model with Optuna's suggested params
    timesnet = TimesNet(
        n_steps=30,
        n_features=15,
        n_layers=n_layers,
        top_k=top_k,
        n_kernels=n_kernels,
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=dropout,
        optimizer=Adam(lr=lr),
        epochs=10,
        batch_size=32,
    )

    # 4. Train the Model
    timesnet.fit({"X": X_train})

    # 5. Evaluate the Model on a Validation Set
    df_val = pd.read_parquet("include/data/intermediate/test_input_r0.25_s5.parquet")
    df_gt = pd.read_csv("include/data/1/test_gt.csv")

    val_scaled = scaler.transform(df_val[TARGET_COLUMNS])
    X_val = create_sliding_windows(val_scaled, 30)

    imputed_outputs = timesnet.impute({"X": X_val})

    # Reconstruct, unscale, and calculate the MAPE
    imputed_2d = reconstruct_2d_from_windows(imputed_outputs, len(df_val), 30)
    imputed_final = scaler.inverse_transform(imputed_2d)
    df_imputed = df_val.copy()
    df_imputed[TARGET_COLUMNS] = imputed_final
    robust_mape = calculate_median_mape(imputed_final, df_gt)

    # 6. Return the score
    return robust_mape


# Run the Optimization Study
if __name__ == "__main__":
    # "minimize" because we want the lowest possible MAPE
    study = optuna.create_study(direction="minimize", study_name="TimesNet_Tuning")

    # Run for 20 trials
    study.optimize(objective, n_trials=20)

    print("\nBest hyperparameters found by Optuna:")
    print(study.best_params)
    print(f"Achieved Validation MAPE: {study.best_value}%")
