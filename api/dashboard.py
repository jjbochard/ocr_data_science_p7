import base64
import io
import json
import os

import dash
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import shap
import streamlit as st
from dash import dcc, html
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
from scipy.special import expit

matplotlib.use("Agg")  # "Agg" = Anti-Grain Geometry = non-GUI backend
load_dotenv(override=True)
TRACKING_URI = os.getenv("TRACKING_URI")
mlflow.set_tracking_uri(TRACKING_URI)
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
FINAL_RUN = os.getenv("FINAL_RUN")


@st.cache_resource
def load_model_and_explainer():
    df = pd.read_csv(
        "data/home_credit_selected_features.csv.gz", compression="gzip"
    )
    # TODO: Remove this line when app in production
    # Filter only the first 10 rows for performance
    df = df.copy().iloc[:10, :]

    # Load model from MLflow
    model_uri = f"runs:/{FINAL_RUN}/final_model"
    pipeline = mlflow.sklearn.load_model(model_uri)
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["processor"]

    df_experiment = df.copy()
    df_experiment_no_id = df_experiment.drop(columns=["SK_ID_CURR"])

    X_transformed_full = preprocessor.transform(df_experiment_no_id)
    X_transformed_full = pd.DataFrame(X_transformed_full)
    X_transformed_full.columns = X_transformed_full.columns.map(
        lambda col: col.split("__")[-1]
    )

    # Explainer global
    explainer = shap.TreeExplainer(model)

    # SHAP global (log-odds)
    shap_values_full = explainer(X_transformed_full)

    # Probas globales
    Y_pred_proba_full = model.predict_proba(X_transformed_full)[:, 1]

    print("L'ets go !")
    return shap_values_full, Y_pred_proba_full, df_experiment, df


shap_values_full, Y_pred_proba_full, df_experiment, df = (
    load_model_and_explainer()
)


def shap_transform_scale(original_shap_values, Y_pred_proba, which_obs):
    """
    Transform SHAP values from log-odds to approximate probability scale.
    Only rescales .values, leaves .data intact.
    """
    untransformed_base_value = original_shap_values.base_values[which_obs]
    base_value_proba = expit(untransformed_base_value)

    distance_to_explain = Y_pred_proba[which_obs] - base_value_proba
    original_explanation_distance = np.sum(
        original_shap_values.values[which_obs]
    )
    distance_coefficient = original_explanation_distance / distance_to_explain

    new_values = original_shap_values.values[which_obs] / distance_coefficient

    shap_values_transformed = shap.Explanation(
        values=new_values,
        base_values=base_value_proba,
        data=original_shap_values.data[which_obs],
        feature_names=original_shap_values.feature_names,
    )

    return shap_values_transformed


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Dropdown(
            id="client-id",
            options=[{"label": str(i), "value": i} for i in df["SK_ID_CURR"]],
            placeholder="Choose a client ID",
        ),
        html.Div(id="client-summary"),
        dcc.Graph(id="proba-vs-threshold"),
        html.Div(id="feature-importance-local"),
    ],
    style={
        "maxWidth": "900px",
        "margin": "0 auto",
        "padding": "20px",
        "backgroundColor": "#f9f9f9",
    },
)


@app.callback(
    [
        dash.Output("client-summary", "children"),
        dash.Output("proba-vs-threshold", "figure"),
        dash.Output("feature-importance-local", "children"),
    ],
    [dash.Input("client-id", "value")],
)
def update_dashboard(client_id):
    if client_id is None:
        raise PreventUpdate

    try:
        client_row = df[df["SK_ID_CURR"] == client_id].drop(
            columns=["SK_ID_CURR"]
        )

        # Manage NaN values
        features = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in client_row.squeeze().to_dict().items()
        }

        payload = {"features": features}

        payload_text = json.dumps(payload, indent=2, ensure_ascii=False)

        api_url = "http://localhost:8000/predict"
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()

        print(result.get("predict_proba", 0.0))

        positive_proba = result.get("predict_proba")[0][1]

        threshold = result.get("threshold", 0.5)

        # Graph for predicted score vs threshold
        fig = px.bar(
            x=["Client score"],
            y=[positive_proba],
            title="Predicted Score vs Threshold",
            text=[f"{positive_proba:.2f}"],
            labels={"x": "", "y": "Score"},
        )

        # Add horizontal line for threshold
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=0.5,
            y0=threshold,
            y1=threshold,
            line=dict(color="red", width=2, dash="dash"),
            name="Threshold",
        )

        fig.add_annotation(
            x=0,
            y=threshold,
            text=f"Threshold = {threshold:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(color="red"),
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(showlegend=False)

        decision = (
            "❌ Credit refused"
            if positive_proba >= threshold
            else "✅ Credit accepted"
        )
        summary = (
            f"Score client predicted : {positive_proba:.2f} | "
            f"Threshold : {threshold:.2f} → {decision}"
        )

        # Retrieve client index
        obs_index = df_experiment[
            df_experiment["SK_ID_CURR"] == client_id
        ].index[0]

        # Extract SHAP values for the specific observation
        shap_values_proba_approx = shap_transform_scale(
            shap_values_full, Y_pred_proba_full, obs_index
        )

        # Create a waterfall plot
        buffer = io.BytesIO()
        shap.plots.waterfall(shap_values_proba_approx, show=False)
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode()
        img_html = html.Img(
            src=f"data:image/png;base64,{encoded_image}",
            style={"width": "100%"},
        )
        return (
            html.Div(
                [
                    html.P(summary),
                    html.Pre(
                        payload_text,
                        style={
                            "whiteSpace": "pre-wrap",
                            "fontFamily": "monospace",
                        },
                    ),
                ]
            ),
            fig,
            img_html,
        )
    except Exception as e:
        print(f"[ERROR update_dashboard] {e}")
        return (
            html.Div(f"API Error: {e}"),
            {},
            html.Div("Erreur lors de la génération de l'explication locale."),
        )


if __name__ == "__main__":
    app.run(debug=True)
