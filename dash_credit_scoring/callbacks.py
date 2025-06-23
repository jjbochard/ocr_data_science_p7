import json

import numpy as np
import requests
from components.graphs import shap_waterfall_plot
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate
from utils.model_loader import load_model_and_data
from utils.shap_utils import transform_shap_to_proba

shap_values_full, Y_pred_proba_full, df_full, df_transformed, explainer = (
    load_model_and_data()
)


def register_callbacks(app):
    @app.callback(
        [
            Output("client-summary", "children"),
            Output("proba-vs-threshold", "figure"),
            Output("feature-importance-local", "children"),
        ],
        [Input("client-id", "value")],
    )
    def update_dashboard(client_id):
        if client_id is None:
            raise PreventUpdate

        try:
            row = df_full[df_full["SK_ID_CURR"] == client_id].drop(
                columns=["SK_ID_CURR"]
            )

            # Manage NaN values
            features = {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in row.squeeze().to_dict().items()
            }

            payload = {"features": features}
            payload_text = json.dumps(payload, indent=2, ensure_ascii=False)

            response = requests.post(
                "http://localhost:8000/predict", json=payload
            )
            response.raise_for_status()
            result = response.json()

            positive_proba = result.get("predict_proba")[0][1]
            threshold = result.get("threshold", 0.5)

            decision = (
                "❌ Credit refused"
                if positive_proba >= threshold
                else "✅ Credit validate"
            )
            summary_text = (
                f"Predicted score : {positive_proba:.2f} | "
                + f"Threshold : {threshold:.2f} → {decision}"
            )

            # Graph score vs threshold
            figure = {
                "data": [
                    {
                        "x": ["Client score"],
                        "y": [positive_proba],
                        "type": "bar",
                        "text": [f"{positive_proba:.2f}"],
                        "textposition": "auto",
                    }
                ],
                "layout": {
                    "title": "Score vs Threshold",
                    "yaxis": {"range": [0, 1]},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": -0.5,
                            "x1": 0.5,
                            "y0": threshold,
                            "y1": threshold,
                            "line": {
                                "color": "red",
                                "width": 2,
                                "dash": "dash",
                            },
                        }
                    ],
                    "annotations": [
                        {
                            "x": 0,
                            "y": threshold,
                            "text": f"Seuil = {threshold:.2f}",
                            "showarrow": False,
                            "font": {"color": "red"},
                            "yshift": 10,
                        }
                    ],
                    "showlegend": False,
                },
            }

            # SHAP local
            obs_index = df_full[df_full["SK_ID_CURR"] == client_id].index[0]
            shap_proba_expl = transform_shap_to_proba(
                shap_values_full, Y_pred_proba_full, obs_index
            )
            fig_shap = shap_waterfall_plot(shap_proba_expl, max_display=10)
            fig_shap.update_yaxes(autorange="reversed")
            return (
                html.Div(
                    [
                        html.P(summary_text),
                        html.Pre(
                            payload_text,
                            style={
                                "whiteSpace": "pre-wrap",
                                "fontFamily": "monospace",
                            },
                        ),
                    ]
                ),
                figure,
                dcc.Graph(figure=fig_shap),
            )

        except Exception as e:
            print(f"[ERROR update_dashboard] {e}")
            return (
                html.Div(f"Erreur API: {e}"),
                {},
                html.Div(
                    "Erreur lors de la génération de l’explication locale."
                ),
            )
