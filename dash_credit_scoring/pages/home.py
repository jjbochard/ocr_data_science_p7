from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
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
(
    shap_values_full,
    Y_pred_proba_full,
    df_full,
    df_transformed,
    explainer,
    groupable_columns,
    _,
) = load_model_and_data()

LOW_UNIQUE_THRESHOLD = 10
WEEKDAY_ORDER = [
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
]
palette = px.colors.qualitative.Plotly


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
        dcc.Dropdown(
            id="group-column",
            options=[{"label": "All clients", "value": ""}]
            + [{"label": col, "value": col} for col in groupable_columns],
            value="",
            placeholder="Choose a column to groupby",
            style={"marginTop": "10px"},
        ),
        html.Div(id="violin-plots"),
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


@callback(
    Output("violin-plots", "children"),
    Input("client-id", "value"),
    Input("max-feature-count", "value"),
    Input("group-column", "value"),
)
def update_violin_plots(client_id, max_features, group_column):
    if client_id is None or group_column is None:
        raise PreventUpdate

    try:
        row, obs_index = get_row_and_index(df_full, "SK_ID_CURR", client_id)
        shap_vals = shap_values_full.values[obs_index]
        feature_names = shap_values_full.feature_names

        # Top sorted N SHAP features
        top_features = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_feature_names = [f[0] for f in top_features[:max_features]]

        # Keep only numerical features
        numeric_cols = df_full.select_dtypes(include=[np.number]).columns
        top_numeric_features = [
            f for f in top_feature_names if f in numeric_cols
        ]

        figures = []
        for feature in top_numeric_features:
            fig = go.Figure()
            unique_vals = df_full[feature].nunique()
            use_bar_plot = unique_vals < LOW_UNIQUE_THRESHOLD
            client_val = row[feature]

            if group_column:
                groups_raw = df_full[group_column].dropna().unique().tolist()
                # Reorder weekday
                if group_column == "WEEKDAY_APPR_PROCESS_START":
                    groups = [
                        day for day in WEEKDAY_ORDER if day in groups_raw
                    ]
                else:
                    groups = sorted(groups_raw)

                color_map = {
                    group: palette[i % len(palette)]
                    for i, group in enumerate(groups)
                }

                for group in groups:
                    mask = df_full[group_column] == group
                    group_size = mask.sum()
                    y_vals = df_full[feature][mask]
                    color = color_map[group]

                    if use_bar_plot:
                        value_counts = y_vals.value_counts().sort_index()
                        fig.add_trace(
                            go.Bar(
                                x=[str(x) for x in value_counts.index],
                                y=value_counts.values,
                                name=f"{group} ({group_size})",
                                marker_color=color,
                                marker_line_color=color,
                                opacity=0.6,
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Violin(
                                x=df_full[group_column][mask],
                                y=y_vals,
                                name=f"{group} ({group_size})",
                                box_visible=True,
                                meanline_visible=True,
                                points=False,
                            )
                        )
                client_group = row.get(group_column)
                x_client = (
                    [str(client_val)] if use_bar_plot else [client_group]
                )
            else:
                if use_bar_plot:
                    value_counts = df_full[feature].value_counts().sort_index()
                    fig.add_trace(
                        go.Bar(
                            x=[str(x) for x in value_counts.index],
                            y=value_counts.values,
                            name=f"All clients ({len(df_full)})",
                            opacity=0.6,
                        )
                    )
                    x_client = [str(client_val)]

                else:
                    fig.add_trace(
                        go.Violin(
                            x=["All clients"] * len(df_full),
                            y=df_full[feature],
                            name=f"All clients ({len(df_full)})",
                            box_visible=True,
                            meanline_visible=True,
                            points=False,
                        )
                    )
                    x_client = ["All clients"]

            # Add a trace for the selected client
            fig.add_trace(
                go.Scatter(
                    x=[x_client],
                    y=[client_val],
                    mode="markers+text",
                    name="Client",
                    marker=dict(color="black", size=12),
                    text=[f"Client: {row[feature]:.1f}"],
                    textposition="top center",
                    showlegend=False,
                )
            )
            title = f"Distribution of {feature}\n"
            if group_column:
                title += f" by {group_column} (Client vs Population)"
            else:
                title += "\n(Client vs Population)"

            layout_config = {
                "title": title,
                "xaxis": {"showticklabels": False},
                "yaxis_title": feature,
                "legend": {
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": -0.4,
                    "xanchor": "center",
                    "x": 0.5,
                },
            }
            if not use_bar_plot:
                layout_config.update(
                    {
                        "violingap": 0.3,
                        "violingroupgap": 0.4,
                    }
                )
            fig.update_layout(**layout_config)
            figures.append(
                html.Div(
                    dcc.Graph(figure=fig),
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "margin": "1%",
                    },
                )
            )

        return html.Div(
            figures,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "center",
            },
        )

    except Exception as e:
        print(f"[ERROR updating violin plots] {e}")
        return html.Div("Error during violin plots construction")
