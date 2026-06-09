import os
import glob
from airflow.sdk import Asset, dag, task
from airflow.sdk import Param
from pendulum import datetime
from include.logic.data_preparation import (
    ingest_raw_csvs,
    apply_discrete_adapter,
    stage_and_prepare_dataset,
)

INTERMEDIATE_DIR = "include/data/intermediate"
prepared_data_asset = Asset("file://include/intermediate/prepared_data")


@dag(
    start_date=datetime(2024, 4, 20),
    schedule=None,
    catchup=False,
    params={
        "dataset_folder": Param(
            default="python",
            type="string",
            description="The folder name inside include/data/datasets/ (e.g., 'python', 'amf')",
        ),
        "is_pre_split": Param(
            default=False,
            type="boolean",
            description="If True, loads train/test/gt csvs. If False, expects a raw.csv to split.",
        ),
        "mode": Param(
            default="dynamic",
            type="string",
            enum=["static", "dynamic"],
            description="Use static test_input.csv or dynamically generate gaps?",
        ),
        "gap_scenarios": Param(
            default=[
                [0.1, 1],
                [0.1, 5],
                [0.1, 10],
                [0.1, 20],
                [0.1, 30],
                [0.1, 60],
                [0.25, 1],
                [0.25, 5],
                [0.25, 10],
                [0.25, 20],
                [0.25, 30],
                [0.25, 60],
                [0.4, 1],
                [0.4, 5],
                [0.4, 10],
                [0.4, 20],
                [0.4, 30],
                [0.4, 60],
                [0.5, 1],
                [0.5, 5],
                [0.5, 10],
                [0.5, 20],
                [0.5, 30],
                [0.5, 60],
            ],
            type="array",
            description="List of [missing_ratio, block_size] scenarios.",
            items={
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        ),
    },
)
def data_preparation():
    @task
    def clean_intermediate_directory(out_dir):
        """Ensures a clean state by deleting all old files."""
        imputed_dir = os.path.join(out_dir, "imputed")

        # Make sure the directories exist before trying to clean them
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(imputed_dir, exist_ok=True)

        # 1. Clean the main intermediate directory
        for file_path in glob.glob(os.path.join(out_dir, "*.parquet")) + glob.glob(
            os.path.join(out_dir, "*.csv")
        ):
            os.remove(file_path)

        # 2. Clean the imputed directory
        for file_path in glob.glob(os.path.join(imputed_dir, "*")):
            os.remove(file_path)

        print("Clean state achieved: Old intermediate and imputed files purged.")
        return True

    @task
    def stage_dataset(intermediate_dir, **kwargs):
        """Pulls raw data, formats it to target specifications, and splits it."""
        ui_params = kwargs["params"]
        dataset_folder = ui_params["dataset_folder"]
        is_pre_split = ui_params["is_pre_split"]

        return stage_and_prepare_dataset(dataset_folder, intermediate_dir, is_pre_split)

    @task(multiple_outputs=True)
    def ingest_datasets(csv_dict, out_dir, **kwargs):
        """Reads staged CSVs, generates gap scenarios, and saves as Parquet."""
        ui_params = kwargs["params"]

        mode = ui_params["mode"]
        gap_scenarios = ui_params["gap_scenarios"]
        return ingest_raw_csvs(csv_dict, out_dir, mode, gap_scenarios)

    @task(outlets=[prepared_data_asset])
    def prepare_discrete_versions(continuous_paths, out_dir):
        """Snaps continuous time-series data to a discrete 1-second grid."""
        discrete_train = apply_discrete_adapter(continuous_paths["train"], out_dir)
        discrete_test_list = []
        for test_path in continuous_paths["test_inputs"]:
            discrete_test_path = apply_discrete_adapter(test_path, out_dir)
            discrete_test_list.append(discrete_test_path)

        return {
            "discrete_train": discrete_train,
            "discrete_test_inputs": discrete_test_list,
        }

    clean_task = clean_intermediate_directory(INTERMEDIATE_DIR)
    staged_csvs = stage_dataset(INTERMEDIATE_DIR)
    continuous_data = ingest_datasets(staged_csvs, INTERMEDIATE_DIR)
    discrete_data = prepare_discrete_versions(continuous_data, INTERMEDIATE_DIR)

    # Dependencies
    clean_task >> staged_csvs >> continuous_data >> discrete_data


data_prep_dag = data_preparation()
