import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app  # assuming your FastAPI app is in app2.py


class DummyModel:
    """Simple dummy model for testing predict and predict_proba."""

    def __init__(self):
        # store last input for inspection if needed
        self.last_df = None

    def predict(self, df: pd.DataFrame):
        # return 1 for all rows
        self.last_df = df
        return [1] * len(df)

    def predict_proba(self, df: pd.DataFrame):
        # return fixed probabilities [[0.1, 0.9], ...]
        self.last_df = df
        return [[0.1, 0.9] for _ in range(len(df))]


@pytest.fixture(autouse=True)
def test_client_and_model(monkeypatch):
    # Create test client
    client = TestClient(app)

    # Inject dummy model and threshold into app state
    dummy = DummyModel()
    app.state.sk_model = dummy
    app.state.threshold = 0.42

    return client


def make_payload():
    # Simple 2-row payload matching DataFrameSplit schema
    payload = {
        "dataframe_split": {"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}
    }
    return payload


def test_predict_success(test_client_and_model):
    client = test_client_and_model
    payload = make_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    # Check threshold
    assert body["threshold"] == 0.42

    # Check predictions
    assert body["predictions"] == [1, 1]

    # Check predict_proba
    assert body["predict_proba"] == [[0.1, 0.9], [0.1, 0.9]]


def test_missing_model(monkeypatch):
    # Simulate missing model
    client = TestClient(app)
    if hasattr(app.state, "sk_model"):
        del app.state.sk_model
    if hasattr(app.state, "threshold"):
        del app.state.threshold

    payload = make_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model or threshold not loaded"


def test_missing_threshold(monkeypatch):
    # Simulate missing model
    client = TestClient(app)
    if hasattr(app.state, "threshold"):
        del app.state.threshold

    payload = make_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model or threshold not loaded"


def test_invalid_payload(test_client_and_model):
    client = test_client_and_model
    # Missing dataframe_split
    response = client.post("/predict", json={})
    # Pydantic validation error -> 422 Unprocessable Entity
    assert response.status_code == 422
