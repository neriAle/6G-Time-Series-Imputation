from airflow.sdk import dag, task
from pendulum import datetime
import os
import glob
from include.logic.evaluation import evaluate_model
from include.logic.aggregation import aggregate_results_for_streamlit

IMPUTED_DIR = "include/data/intermediate/imputed"
GT_PATH = "include/data/intermediate/test_gt.csv"
RESULTS_DIR = "include/data/results"


def get_prediction_paths(model_prefix: str) -> list:
    """Helper to dynamically find all output parquets for a specific model."""
    search_pattern = os.path.join(IMPUTED_DIR, f"{model_prefix}_*_output.parquet")
    paths = glob.glob(search_pattern)
    # Fallback for static mode where the tag is just {model}_output.parquet
    if not paths:
        paths = glob.glob(os.path.join(IMPUTED_DIR, f"{model_prefix}_output.parquet"))
    return paths


@dag(
    start_date=datetime(2024, 4, 20),
    schedule=None,
    catchup=False,
)
def model_evaluation():

    @task
    def evaluate_kalman():
        pred_paths = get_prediction_paths("kalman")
        if not pred_paths:
            raise FileNotFoundError(
                "Missing Kalman predictions. Run imputation DAG first!"
            )
        return evaluate_model(pred_paths, GT_PATH, "Kalman", RESULTS_DIR)

    @task
    def evaluate_nearest():
        pred_paths = get_prediction_paths("nearest")
        if not pred_paths:
            raise FileNotFoundError(
                "Missing Nearest predictions. Run imputation DAG first!"
            )
        return evaluate_model(pred_paths, GT_PATH, "Nearest", RESULTS_DIR)

    @task
    def evaluate_timesnet():
        pred_paths = get_prediction_paths("timesnet")
        if not pred_paths:
            raise FileNotFoundError(
                "Missing TimesNet predictions. Run imputation DAG first!"
            )
        return evaluate_model(pred_paths, GT_PATH, "TimesNet", RESULTS_DIR)

    @task
    def evaluate_brits():
        pred_paths = get_prediction_paths("brits")
        if not pred_paths:
            raise FileNotFoundError(
                "Missing BRITS predictions. Run imputation DAG first!"
            )
        return evaluate_model(pred_paths, GT_PATH, "BRITS", RESULTS_DIR)

    @task
    def evaluate_csdi():
        pred_paths = get_prediction_paths("csdi")
        if not pred_paths:
            raise FileNotFoundError(
                "Missing CSDI predictions. Run imputation DAG first!"
            )
        return evaluate_model(pred_paths, GT_PATH, "CSDI", RESULTS_DIR)

    @task
    def build_streamlit_dataset():
        RESULTS_DIR = "include/data/results"
        IMPUTED_DIR = "include/data/intermediate/imputed"
        OUTPUT_CSV = "include/data/results/streamlit_dataset.csv"

        return aggregate_results_for_streamlit(RESULTS_DIR, IMPUTED_DIR, OUTPUT_CSV)

    # Execution
    kalman = evaluate_kalman()
    nearest = evaluate_nearest()
    timesnet = evaluate_timesnet()
    brits = evaluate_brits()
    csdi = evaluate_csdi()
    streamlit = build_streamlit_dataset()

    [
        kalman,
        nearest,
        timesnet,
        brits,
        csdi,
    ] >> streamlit


eval_dag = model_evaluation()
