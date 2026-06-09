# 6G Multivariate Time-Series Imputation Pipeline

<p align="center">
  <a href="https://gitmoji.dev">
    <img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square" alt="Gitmoji">
  </a>
  <a href="https://www.python.org/downloads/release/python-3120/">
    <img src="https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+">
  </a>
  <a href="https://docs.astral.sh/ruff/">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://docs.pytest.org/">
    <img src="https://img.shields.io/badge/pytest-Testing-0A9EDC.svg?style=flat-square&logo=pytest&logoColor=white" alt="Pytest">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Containerized-2496ED.svg?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  </a>
  <a href="https://airflow.apache.org/">
    <img src="https://img.shields.io/badge/Apache_Airflow-Orchestration-017CEE.svg?style=flat-square&logo=apache-airflow&logoColor=white" alt="Apache Airflow">
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg?style=flat-square" alt="License: Apache 2.0">
  </a>
</p>

This repository contains the end-to-end MLOps evaluation framework for benchmarking multivariate time-series imputation architectures under simulated 6G edge telemetry outages and connection drops. 

This pipeline was developed to evaluate the Pareto-optimal trade-off between reconstruction accuracy (RMSE/MAPE) and algorithmic latency across traditional statistical baselines and modern deep learning paradigms.

## Evaluated Architectures
* **Statistical Baselines:** Nearest Neighbor, Kalman Filter ([`Darts`](https://github.com/unit8co/darts))
* **Deep Learning (Recurrent):** BRITS ([`PyPOTS`](https://github.com/WenjieDu/PyPOTS))
* **Deep Learning (Generative):** CSDI ([`PyPOTS`](https://github.com/WenjieDu/PyPOTS))
* **Deep Learning (Multi-Periodic):** TimesNet ([`PyPOTS`](https://github.com/WenjieDu/PyPOTS))

*Note on Hyperparameter Tuning:* Model optimization (specifically for TimesNet) was conducted offline using the [Optuna](https://optuna.org/) framework with GPU acceleration (`device="cuda"`). However, to ensure a strictly fair and hardware-agnostic comparison for the final Pareto latency frontier, all model inferences within the Airflow evaluation pipeline are forced to execute on the CPU.

## Prerequisites
To ensure absolute reproducibility, regardless of OS, the Airflow evaluation framework is containerized. To run the pipeline and the local dashboard, you will need:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* [Astronomer CLI](https://docs.astronomer.io/astro/cli/install-cli) (`astro`) installed.
* **Python 3.12+** (Required locally for the interactive [Streamlit](https://streamlit.io/) dashboard).
* **Recommended Hardware:** 16GB RAM and a multi-core CPU. *(Note: Running the full pipeline on 8GB of RAM is possible, but you may need to increase your operating system's Swap/Pagefile memory to prevent Docker Out-Of-Memory container crashes during the heavy deep learning inferences).*

## Setup & Execution Instructions

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/neriAle/6G-Time-Series-Imputation.git
cd 6G-Time-Series-Imputation
```

### 2. Launch the Pipeline
Start the Airflow environment using the Astronomer CLI. This will spin up the necessary Postgres, Scheduler, and Webserver containers:
```bash
astro dev start
```

### 3. Access the Airflow UI
Because this project utilizes dynamic port mapping to prevent local collisions, the Airflow Webserver port changes on each initialization. To find the correct UI link:
1. Open **Docker Desktop**.
2. Locate the running container named `api-server-1`.
3. Under the **"Ports"** column, click the dynamically generated localhost link (e.g., `http://127.0.0.1:11371/`).

### 4. Run the Experiments
The pipeline is modularized into four distinct DAGs with a dynamic staging area. To execute the full evaluation:

1. **Start the Pipeline:** In the Airflow dashboard, click the **"Trigger"** (Play) button on the `data_preparation` DAG. Here you can specify the `dataset_folder` (e.g., `amf` or `python`), indicate if the data `is_pre_split`, and adjust the 24-scenario simulation grid. Click **"Trigger"** at the bottom of the page.
2. **Wait for Imputation:** The `data_preparation` DAG will automatically trigger the `data_imputation` DAG upon completion. The pipeline utilizes a throttled concurrency limit (`max_active_tasks=2`) to protect local RAM while maintaining parallel branching. Wait for `data_imputation` to fully succeed for all models. *(Note: Training deep learning architectures from scratch on new datasets requires significant compute time)*.
3. **Evaluate Metrics:** Once imputation is complete, manually click the **"Trigger"** button on `model_evaluation` to calculate the RMSE and MAPE scores.
4. **Stage Results for Visualization:** The pipeline outputs results to a volatile staging area. To visualize the data and allow the Streamlit dashboard to dynamically discover your dataset, you must move the outputs into a dedicated folder. Create a new folder named after your dataset (e.g., `include/data/results/amf/`) and move the following items into it:
   - `include/data/results/streamlit_dataset.csv`
   - `include/data/intermediate/test_gt.csv`
   - The entire `include/data/intermediate/imputed/` folder
5. **Interactive Dashboard:** With the evaluation complete and files safely staged, you can interactively explore the results across any generated datasets. Open a terminal in the root directory of the project and launch the web dashboard:
   ```bash
   pip install streamlit pandas
   streamlit run streamlit_app.py
   ```
6. **Generate Static Visuals:** Alternatively, you can manually trigger the `generate_plots` DAG to output the aggregate tables, line charts, and Pareto frontiers as static files.

### 5. Shut Down
To safely spin down the containers run:
```bash
astro dev stop
```

## Dataset Information
This repository utilizes 5G/6G Core Edge Telemetry (specifically 5G AMF request configurations and Python Web Server metrics) sourced from a public [Zenodo repository](https://zenodo.org/records/6907619) designed for microservice benchmarking. The datasets can be placed into the `/include/data/datasets` directory for dynamic pipeline ingestion.

## 📄 License
This project is licensed under the [Apache-2.0 License](LICENSE).