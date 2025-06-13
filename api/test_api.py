import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pandas.testing import assert_frame_equal

from api.api import app


def post_predict(client, payload):
    """
    Helper to send a predict request and return the response.
    """
    return client.post("/predict", json=payload)


def assert_df_received(model, payload):
    """
    Assert that the dummy model received the correct DataFrame.
    """
    expected = pd.DataFrame([payload["features"]])
    actual = model.last_df
    assert actual is not None, "Model did not receive a DataFrame"
    assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


def assert_response_body(body, threshold, predictions, proba):
    """
    Assert the response JSON body matches expected values.
    """
    assert body["threshold"] == threshold
    assert body["predictions"] == predictions
    assert body["predict_proba"] == proba


class DummyModel:
    """Simple dummy model for testing predict and predict_proba."""

    def __init__(self):
        self.last_df = None

    def predict(self, df: pd.DataFrame):
        self.last_df = df
        return [1] * len(df)

    def predict_proba(self, df: pd.DataFrame):
        self.last_df = df
        return [[0.1, 0.9] for _ in range(len(df))]


@pytest.fixture
def client_and_model():
    """Provide a TestClient with DummyModel injected into app state."""
    client = TestClient(app)
    dummy = DummyModel()
    app.state.sk_model = dummy
    app.state.threshold = 0.42
    return client, dummy


@pytest.fixture
def payload():
    """Return a standard 2-row payload matching DataFrameSplit schema."""
    return {"features": {"a": 1, "b": 2}}


def test_predict_success(client_and_model, payload):
    client, dummy = client_and_model
    response = post_predict(client, payload)
    assert response.status_code == 200
    body = response.json()

    # verify DataFrame passed to model
    assert_df_received(dummy, payload)

    # verify response contents
    assert_response_body(
        body,
        threshold=0.42,
        predictions=[1],
        proba=[[0.1, 0.9]],
    )


@pytest.mark.parametrize(
    "remove_model, remove_threshold, detail",
    [
        (True, True, "Model or threshold not loaded"),
        (True, False, "Model or threshold not loaded"),
        (False, True, "Model or threshold not loaded"),
    ],
)
def test_missing_model_or_threshold(
    remove_model, remove_threshold, detail, payload
):
    client = TestClient(app)
    if remove_model and hasattr(app.state, "sk_model"):
        del app.state.sk_model
    if remove_threshold and hasattr(app.state, "threshold"):
        del app.state.threshold

    response = post_predict(client, payload)
    assert response.status_code == 503
    assert response.json()["detail"] == detail


def test_invalid_payload(client_and_model):
    client, _ = client_and_model
    response = post_predict(client, {})
    assert response.status_code == 422
