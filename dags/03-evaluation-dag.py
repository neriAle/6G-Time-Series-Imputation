from airflow.sdk import dag, task
from pendulum import datetime
import os
from include.logic.evaluation import evaluate_model

IMPUTED_DIR = "include/data/intermediate/imputed"
GT_PATH = "include/data/1/test_gt.csv"
RESULTS_DIR = "include/data/results"


@dag(
    start_date=datetime(2024, 4, 20),
    schedule=None,
    catchup=False,
)
def model_evaluation():

    @task
    def evaluate_kalman():
        pred_path = os.path.join(IMPUTED_DIR, "kalman_output.parquet")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"Missing predictions at {pred_path}. Run imputation DAG first!"
            )

        return evaluate_model(pred_path, GT_PATH, "Kalman", RESULTS_DIR)

    @task
    def evaluate_timesnet():
        pred_path = os.path.join(IMPUTED_DIR, "timesnet_output.parquet")
        if not os.path.exists(pred_path):
            # raise FileNotFoundError(
            #     f"Missing predictions at {pred_path}. Run imputation DAG first!"
            # )
            return "placeholder"

        return evaluate_model(pred_path, GT_PATH, "TimesNet", RESULTS_DIR)

    @task
    def evaluate_brits():
        pred_path = os.path.join(IMPUTED_DIR, "brits_output.parquet")
        if not os.path.exists(pred_path):
            # raise FileNotFoundError(
            #     f"Missing predictions at {pred_path}. Run imputation DAG first!"
            # )
            return "placeholder"

        return evaluate_model(pred_path, GT_PATH, "BRITS", RESULTS_DIR)

    @task
    def evaluate_csdi():
        pred_path = os.path.join(IMPUTED_DIR, "csdi_output.parquet")
        if not os.path.exists(pred_path):
            # raise FileNotFoundError(
            #     f"Missing predictions at {pred_path}. Run imputation DAG first!"
            # )
            return "placeholder"

        return evaluate_model(pred_path, GT_PATH, "CSDI", RESULTS_DIR)

    # Execution
    evaluate_kalman()
    evaluate_timesnet()
    evaluate_brits()
    evaluate_csdi()


eval_dag = model_evaluation()
