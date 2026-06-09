from airflow.sdk import dag, task, Asset
from pendulum import datetime
import os
import glob
from include.logic.models_darts import impute_kalman_filter, impute_nearest
from include.logic.models_pypots import impute_pypots_model
from include.logic.models_timesnet import impute_timesnet

prepared_data_asset = Asset("file://include/intermediate/prepared_data")
INTERMEDIATE_DIR = "include/data/intermediate"
IMPUTED_DIR = "include/data/intermediate/imputed"


def get_test_paths(intermediate_dir, is_discrete=False):
    """Helper to dynamically find test files at runtime."""
    all_files = glob.glob(os.path.join(intermediate_dir, "test_input*.parquet"))
    if is_discrete:
        # Keep only files that have 'discrete' in the name
        return [f for f in all_files if "discrete" in f]
    else:
        # Keep only continuous files (exclude 'discrete')
        return [f for f in all_files if "discrete" not in f]


@dag(
    start_date=datetime(2024, 4, 20),
    schedule=[prepared_data_asset],  # Triggers when data-preparation finishes
    catchup=False,
    max_active_tasks=2,
)
def data_imputation():

    # 1. Discrete Models (Using the discrete .parquet)
    @task
    def run_kalman_filter(train_path, intermediate_dir):
        test_paths = get_test_paths(intermediate_dir, is_discrete=True)
        return impute_kalman_filter(train_path, test_paths, IMPUTED_DIR)

    @task
    def run_nearest(intermediate_dir):
        test_paths = get_test_paths(intermediate_dir, is_discrete=True)
        return impute_nearest(test_paths, IMPUTED_DIR)

    @task
    def run_timesnet(train_path, intermediate_dir):
        test_paths = get_test_paths(intermediate_dir, is_discrete=True)
        return impute_timesnet(train_path, test_paths, IMPUTED_DIR)

    # 2. Continuous Models (Using the raw Partially Observed Time Series (POTS) .parquet)
    @task
    def run_brits(train_path, intermediate_dir):
        test_paths = get_test_paths(intermediate_dir, is_discrete=False)
        return impute_pypots_model(train_path, test_paths, "BRITS", IMPUTED_DIR)

    @task
    def run_csdi(train_path, intermediate_dir):
        test_paths = get_test_paths(intermediate_dir, is_discrete=False)
        return impute_pypots_model(train_path, test_paths, "CSDI", IMPUTED_DIR)

    continuous_train = os.path.join(INTERMEDIATE_DIR, "train.parquet")
    discrete_train = os.path.join(INTERMEDIATE_DIR, "train_discrete.parquet")

    run_kalman_filter(discrete_train, INTERMEDIATE_DIR)
    run_nearest(INTERMEDIATE_DIR)
    run_timesnet(discrete_train, INTERMEDIATE_DIR)
    run_brits(continuous_train, INTERMEDIATE_DIR)
    run_csdi(continuous_train, INTERMEDIATE_DIR)


imputation_dag = data_imputation()
