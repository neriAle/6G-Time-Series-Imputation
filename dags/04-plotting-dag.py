from airflow.sdk import dag, task
from pendulum import datetime
from include.logic.plotting import (
    plot_time_series,
    plot_accuracy_bars,
    plot_pareto_frontier,
)

IMPUTED_DIR = "include/data/intermediate/imputed"
GT_PATH = "include/data/intermediate/test_gt.csv"
RESULTS_DIR = "include/data/results"
PLOTS_DIR = "include/data/plots"


@dag(
    start_date=datetime(2024, 4, 20),
    schedule=None,
    catchup=False,
)
def generate_plots():

    @task
    def create_timeseries():
        plot_time_series(IMPUTED_DIR, GT_PATH, PLOTS_DIR)

    @task
    def create_accuracy_bars():
        plot_accuracy_bars(RESULTS_DIR, PLOTS_DIR)

    @task
    def create_pareto():
        plot_pareto_frontier(RESULTS_DIR, IMPUTED_DIR, PLOTS_DIR)

    create_timeseries()
    create_accuracy_bars()
    create_pareto()


plotting_dag = generate_plots()
