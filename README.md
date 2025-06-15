# P7 - Implement a Scoring Model

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [MLflow & Model Registry](#mlflow--model-registry)
- [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
- [Start experiment](#start-experiment)
- [API Usage (FastAPI)](#api-usage-fastapi)
- [Running Tests](#running-tests)
- [License](#license)

## Overview

This project demonstrates the full lifecycle of a credit scoring model:
- **Data Preprocessing** and model training using scikit-learn and several models (LogisticRegression, RandomForestClassifier, LGBMClassifier and XGBClassifier).
- **Experiment tracking** with MLflow, including logging parameters, metrics, and registering the trained model in the MLflow Model Registry.
- **Hyperparameter optimization** using Optuna, with automatic logging of each trial.
- **Serving** the registered model via a FastAPI REST API for inference, including probability thresholds for decision making.
- **Automated testing** of the API endpoints using pytest.


## Installation

### Clone the repository:
  ```bash
    mkdir foo
    git clone https://github.com/jjbochard/ocr_data_science_p7 foo
    cd foo
  ```

### Create and activate a virtual environment:
First, install [Python 3.13+](https://www.python.org/downloads/).
  ```bash
    python3 -m venv venv
    source venv/bin/activate    # macOS/Linux
    .\venv\Scripts\activate   # Windows PowerShell
  ```

Install dependencies:
  ```bash
    pip3 install -r requirements.txt
  ```

 Install the JupyterLab extension (or Jupyter):
  ```bash
    pip install jupyterlab
  ```

To deactivate your venv:
  ```bash
    deactivate
  ```

### Optionnal : configure your git repository with pre-commit (if you want to fork this project)

You can install pre-commit with python

    pip install pre-commit

You can install the configured pre commit hook with

    pre-commit install

## Configuration

Copy `.env.example` to `.env` and edit as needed:

This configures MLflow to point at your tracking server and selects which model version to load.

## MLflow & Model Registry

- **Training runs** are tracked with MLflow. All parameters, metrics, and artifacts are logged.
- After training, the  best model is registered under the name `home_credit/<num_version>` in the **Model Registry**.
- The FastAPI app loads the live model from the registry (using `models:/` URI) and reads the saved `threshold` parameter.
- You can view runs and model versions in the MLflow UI:
  ```bash
  mlflow ui --host 0.0.0.0 --port 5000
  ```

## Hyperparameter Tuning with Optuna

- The Optuna script performs cross-validated searches over parameters (e.g. `threshold`, model hyperparams).
- Each trial is logged as a nested MLflow run, capturing trial parameters and evaluation metrics (cost, ROC AUC and more).
- The best parameters and optimal threshold are automatically registered in MLflow.
![Mlflow UI](mlflow_ui.png)

## Start experiment

You just have to follow this template

```python
df_experiment = data_feat_engineering_bureau_previous_pos_cash_ins.copy()

df_experiment.drop(
    columns=["SK_ID_CURR", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    inplace=True,
)
model_name = "lightgbm"
run_experiment(
    X=df_experiment.drop(columns=["TARGET"]),
    y=df_experiment["TARGET"],
    n_trials=30,
    n_splits=5,
    fn_cost=1000,
    fp_cost=100,
    model_name=model_name,
    dataset_name="train-fe-b-p-pc-i",
    experiment_name="home_credit_" + model_name,
    test_size=0.2,
    random_state=42,
)

del df_experiment
gc.collect()
```

Here, delete an ID column and column with data from an unknown source.

For the dataset_name, use abbreviation to get a smaller name.

The model is trained adding files one by one.

train is the raw dataset. train_fe is the raw dataset with columns from features engineering. And so on.

b -> data from bureau.csv and bureau_balance.csv file.

p -> data from previous_application.csv file.

pc -> data from POS_CASH_balance.csv file.

i -> data from installments_payment.csv file.

ccb -> data from credit_card_balance.csv files.


## API Usage (FastAPI)

- The API serves predictions via POST `/predict`.
- The API expects a JSON object with a `features` key containing a flat dictionary
where each key is a feature name and each value is the corresponding input.

**Example:**

```json
{
  "features": {
    "EXT_SOURCE_2": 0.5943,
    "EXT_SOURCE_3": 0.4276,
    "EXT_SOURCE_1": null,
    "AMT_ANNUITY": 30676.5,
    "CODE_GENDER": "M",
    "AGE": 37,
    "AMT_INCOME_TOTAL": 157500.0
  }
}
```
- Response includes:

  ```json
  {
    "threshold": 0.42,
    "predictions": [0],
    "predict_proba": [[0.8, 0.2]]
  }
  ```
- Run the API locally:
  ```bash
  make run_api_dev
  ```
- Example curl request:
  ```bash
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       --data-binary @payload.json
  ```

## Running Tests

Automated tests ensure the API returns expected outputs and handles errors:
```bash
pytest tests/test_api.py
```
Tests include:
- Successful inference with a dummy model.
- 503 when model or threshold is missing.
- 422 for invalid payload schemas.
