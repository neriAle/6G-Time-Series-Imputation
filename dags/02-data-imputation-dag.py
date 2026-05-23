from airflow.sdk import dag, task, Asset
from pendulum import datetime
import os
from include.logic.models_darts import impute_kalman_filter, impute_nearest
from include.logic.models_pypots import impute_pypots_model

prepared_data_asset = Asset("file://include/intermediate/prepared_data")
INTERMEDIATE_DIR = "include/data/intermediate"
IMPUTED_DIR = "include/data/intermediate/imputed"


@dag(
    start_date=datetime(2024, 4, 20),
    schedule=[prepared_data_asset],  # Triggers when data-preparation finishes
    catchup=False,
)
def data_imputation():

    # 1. Discrete Models (Using the discrete .parquet)
    @task
    def run_kalman_filter(discrete_train_path, discrete_test_path):
        return impute_kalman_filter(
            discrete_train_path, discrete_test_path, IMPUTED_DIR
        )

    @task
    def run_nearest(discrete_test_path):
        return impute_nearest(discrete_test_path, IMPUTED_DIR)

    @task
    def run_timesnet(discrete_train_path, discrete_test_path):
        # Placeholder
        return ""

    # 2. Continuous Models (Using the raw Partially Observed Time Series (POTS) .parquet)
    @task
    def run_brits(continuous_train_path, continuous_test_path):
        return impute_pypots_model(
            continuous_train_path, continuous_test_path, "BRITS", IMPUTED_DIR
        )

    @task
    def run_csdi(continuous_train_path, continuous_test_path):
        return impute_pypots_model(
            continuous_train_path, continuous_test_path, "CSDI", IMPUTED_DIR
        )

    continuous_train = os.path.join(INTERMEDIATE_DIR, "train.parquet")
    continuous_test = os.path.join(INTERMEDIATE_DIR, "test_input.parquet")
    discrete_train = os.path.join(INTERMEDIATE_DIR, "train_discrete.parquet")
    discrete_test = os.path.join(INTERMEDIATE_DIR, "test_input_discrete.parquet")

    run_kalman_filter(discrete_train, discrete_test)
    run_nearest(discrete_test)
    run_timesnet(discrete_train, discrete_test)
    run_brits(continuous_train, continuous_test)
    run_csdi(continuous_train, continuous_test)


imputation_dag = data_imputation()
