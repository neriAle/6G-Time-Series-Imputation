from airflow.sdk import Asset, dag, task
from airflow.sdk import Param
from pendulum import datetime
from include.logic.data_preparation import ingest_raw_csvs, apply_discrete_adapter

INPUT_CSVS = {
    "train": "include/data/1/train.csv",
    "test_input": "include/data/1/test_input.csv",
    "test_gt": "include/data/1/test_gt.csv",
}
INTERMEDIATE_DIR = "include/data/intermediate"

# The asset that the Imputation DAG will listen for
prepared_data_asset = Asset("file://include/intermediate/prepared_data")


@dag(
    start_date=datetime(2024, 4, 20),
    schedule=None,
    catchup=False,
    params={
        "mode": Param(
            default="static",
            type="string",
            enum=["static", "dynamic"],
            description="Use static test_input.csv or dynamically generate gaps?",
        ),
        "gap_scenarios": Param(
            default=[[0.2, 5], [0.4, 10]],
            type="array",
            description="List of [missing_ratio, block_size] scenarios to test (if mode=dynamic).",
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

    @task(multiple_outputs=True)
    def ingest_datasets(csv_dict, out_dir, **kwargs):
        ui_params = kwargs["params"]

        mode = ui_params["mode"]
        gap_scenarios = ui_params["gap_scenarios"]
        print(gap_scenarios)
        return ingest_raw_csvs(csv_dict, out_dir, mode, gap_scenarios)

    @task(outlets=[prepared_data_asset])
    def prepare_discrete_versions(continuous_paths, out_dir):
        discrete_train = apply_discrete_adapter(continuous_paths["train"], out_dir)
        discrete_test_list = []
        for test_path in continuous_paths["test_inputs"]:
            discrete_test_path = apply_discrete_adapter(test_path, out_dir)
            discrete_test_list.append(discrete_test_path)

        return {
            "discrete_train": discrete_train,
            "discrete_test_inputs": discrete_test_list,
        }

    continuous_data = ingest_datasets(INPUT_CSVS, INTERMEDIATE_DIR)
    prepare_discrete_versions(continuous_data, INTERMEDIATE_DIR)


data_prep_dag = data_preparation()
