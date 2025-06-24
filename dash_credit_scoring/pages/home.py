from typing import Tuple

import numpy as np
from components.graphs import shap_waterfall_plot
from dash import Input, Output, callback, dcc, html, register_page
from dash.exceptions import PreventUpdate
from utils import (
    call_prediction_api,
    get_row_and_index,
    load_model_and_data,
    make_score_figure,
    transform_shap_to_proba,
)

register_page(__name__, path="/home", name="Home")
shap_values_full, Y_pred_proba_full, df_full, df_transformed, explainer, _ = (
    load_model_and_data()
)

client_ids = df_full["SK_ID_CURR"].tolist()

layout = html.Div(
    [
        dcc.Dropdown(
            id="client-id",
            options=[{"label": str(i), "value": i} for i in client_ids],
            placeholder="Choose a client ID",
        ),
        html.Div(id="client-summary"),
        dcc.Graph(id="proba-vs-threshold"),
        dcc.Dropdown(
            id="max-feature-count",
            options=[
                {"label": f"{n} features", "value": n}
                for n in [5, 10, 15, 20, 25, 50]
            ],
            value=5,
            clearable=False,
            style={"marginTop": "10px"},
        ),
        html.Div(id="feature-importance-local"),
    ],
    style={
        "margin": "0 auto",
        "padding": "20px",
        "backgroundColor": "#f9f9f9",
    },
)


@callback(
    [
        Output("client-summary", "children"),
        Output("proba-vs-threshold", "figure"),
        Output("feature-importance-local", "children"),
    ],
    [Input("client-id", "value"), Input("max-feature-count", "value")],
)
def update_dashboard(
    client_id: int, max_features: int
) -> Tuple[html.Div, dict, html.Div]:
    """
    Update the dashboard with prediction summary, score vs threshold chart,
    and SHAP local explanation for a given client.

    Args:
        client_id (int): ID of the selected client.
        max_features (int): Number of top SHAP features to display.

    Returns:
        Tuple:
            - html.Div: summary with prediction and decision,
            - dict: Plotly figure for the probability bar chart,
            - html.Div: local SHAP explanation graph.
    """
    if client_id is None:
        raise PreventUpdate

    try:
        # Get the client row and its index
        row, obs_index = get_row_and_index(df_full, "SK_ID_CURR", client_id)

        # Clean values
        features = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in row.squeeze().to_dict().items()
        }
        # API call
        result = call_prediction_api(features)

        # Results
        positive_proba = result.get("predict_proba")[0][1]
        threshold = result.get("threshold", 0.5)
        decision = (
            "Credit refused"
            if positive_proba >= threshold
            else "Credit validated"
        )

        summary_text = (
            f"Predicted score : {positive_proba:.2f} | "
            + f"Threshold : {threshold:.2f} → {decision}"
        )

        # Graph score vs threshold
        figure = make_score_figure(
            positive_proba, "Score VS Threshold", threshold
        )

        # SHAP local
        shap_proba_expl = transform_shap_to_proba(
            shap_values_full, Y_pred_proba_full, obs_index
        )
        fig_shap = shap_waterfall_plot(
            shap_proba_expl, max_display=max_features
        )
        fig_shap.update_yaxes(autorange="reversed")
        return (
            html.Div([html.P(summary_text)]),
            figure,
            dcc.Graph(figure=fig_shap),
        )

    except Exception as e:
        print(f"[ERROR update_dashboard] {e}")
        return (
            html.Div(f"Erreur API: {e}"),
            {},
            html.Div("Erreur lors de la génération de l’explication locale."),
        )
