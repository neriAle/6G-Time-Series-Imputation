from airflow.sdk import Asset, dag, task
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


@dag(start_date=datetime(2024, 4, 20), schedule=None, catchup=False)
def data_preparation():

    @task(multiple_outputs=True)
    def ingest_datasets(csv_dict, out_dir):
        return ingest_raw_csvs(csv_dict, out_dir)

    @task(outlets=[prepared_data_asset])
    def prepare_discrete_versions(continuous_paths, out_dir):
        discrete_train = apply_discrete_adapter(continuous_paths["train"], out_dir)
        discrete_test = apply_discrete_adapter(continuous_paths["test_input"], out_dir)

        return {"discrete_train": discrete_train, "discrete_test_input": discrete_test}

    continuous_data = ingest_datasets(INPUT_CSVS, INTERMEDIATE_DIR)
    prepare_discrete_versions(continuous_data, INTERMEDIATE_DIR)


data_prep_dag = data_preparation()
