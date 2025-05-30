import os
from contextlib import asynccontextmanager
from typing import Any, List

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

# Load environment variables
load_dotenv(override=True)
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
TRACKING_URI = os.getenv("TRACKING_URI")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow_client = MlflowClient()


class DataFrameSplit(BaseModel):
    """
    Schema for MLflow `dataframe_split` payload format.

    Attributes:
        columns: list of column names
        data: list of rows, each a list of values
    """

    columns: List[str]
    data: List[List[Any]]


class PredictPayload(BaseModel):
    """
    Request schema for prediction endpoint.
    Accepts only `dataframe_split` format for simplicity.
    """

    dataframe_split: DataFrameSplit


class PredictResponse(BaseModel):
    """
    Response schema for prediction endpoint.

    Attributes:
        threshold: decision threshold used to binarize probabilities
        predictions: list of model predictions (0 or 1)
        predict_proba: list of [prob_neg, prob_pos] for each sample
    """

    threshold: float
    predictions: List[int]
    predict_proba: List[List[float]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context: runs before the application starts.
    - Parses MODEL_URI, fetches model version info.
    - Loads sklearn model from MLflow Model Registry.
    - Reads the `threshold` parameter from the training run.
    - Stores model and threshold in app.state for use in endpoints.
    """
    try:
        model_version = mlflow_client.get_model_version_by_alias(
            name=MODEL_NAME, alias=MODEL_ALIAS
        )
        run_id = model_version.run_id
    except MlflowException:
        raise RuntimeError(
            f"[Error] There is no model with alias '{MODEL_ALIAS}'"
            f" for model '{MODEL_NAME}'"
        )
    # Load the sklearn model
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    sklearn_model = mlflow.sklearn.load_model(model_uri)

    # Retrieve threshold parameter from the run metadata
    run = mlflow_client.get_run(run_id)
    params = run.data.params
    if "threshold" not in params:
        raise RuntimeError(
            "Missing 'threshold' parameter in MLflow run params"
        )
    try:
        threshold_value = float(params["threshold"])
    except ValueError as e:
        raise RuntimeError("Invalid threshold value in MLflow params") from e

    # Store loaded model and threshold in application state
    app.state.sk_model = sklearn_model
    app.state.threshold = threshold_value

    yield


app = FastAPI(
    title="Home Credit API",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """
    Check if API is up
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    request: Request, payload: PredictPayload
) -> PredictResponse:
    """
    Performs inference using loaded model and threshold.
    """
    sklearn_model = getattr(request.app.state, "sk_model", None)
    threshold_value = getattr(request.app.state, "threshold", None)

    if sklearn_model is None or threshold_value is None:
        raise HTTPException(
            status_code=503, detail="Model or threshold not loaded"
        )

    # Convert payload to pandas DataFrame
    df = pd.DataFrame(**payload.dataframe_split.model_dump())

    # Predict classes and probabilities
    preds = sklearn_model.predict(df)
    probas = sklearn_model.predict_proba(df)

    return PredictResponse(
        threshold=threshold_value,
        predictions=preds,
        predict_proba=probas,
    )
