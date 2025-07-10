from typing import Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from components.graphs import shap_waterfall_plot
from dash import Input, Output, callback, dcc, html, register_page
from dash.exceptions import PreventUpdate
from utils import (
    add_group_trace,
    build_layout,
    call_prediction_api,
    get_row_and_index,
    get_top_features,
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


def Card(children, title=None, **kwargs):
    header = (
        dbc.CardHeader(
            html.H3(
                title,
                className="h5 mb-0",
                role="heading",
                **{"aria-level": "2"},
            )
        )
        if title
        else None
    )

    content = (
        [header, dbc.CardBody(children)]
        if header
        else [dbc.CardBody(children)]
    )
    return dbc.Card(
        content,
        className="shadow-sm mb-4 h-100",
        **kwargs,
    )


layout = html.Div(
    [
        # Section: Filter and Summary
        dbc.Row(
            [
                dbc.Col(
                    Card(
                        title="Filter",
                        children=[
                            dbc.Label("Client ID", html_for="client-id"),
                            dcc.Dropdown(
                                id="client-id",
                                options=[
                                    {"label": str(i), "value": i}
                                    for i in client_ids
                                ],
                                placeholder="Select a client ID",
                                value=None,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    md=4,
                    sm=12,
                ),
                dbc.Col(
                    Card(
                        title="Client Summary",
                        children=[
                            html.Div(
                                id="client-summary",
                                role="region",
                                **{"aria-live": "polite"},
                            ),
                        ],
                    ),
                    md=8,
                    sm=12,
                ),
            ],
            className="gy-4 mb-4 align-items-stretch",
        ),
        # Section: Tabs
        Card(
            title="Detailed Analysis",
            children=[
                dbc.Tabs(
                    id="dashboard-tabs",
                    active_tab="tab-score",
                    className="mb-3",
                    children=[
                        # Tab: Score vs Threshold
                        dbc.Tab(
                            label="Score vs Threshold",
                            tab_id="tab-score",
                            children=[
                                dcc.Graph(
                                    id="proba-vs-threshold",
                                    config={"displayModeBar": False},
                                    style={"width": "100%", "height": "40vh"},
                                )
                            ],
                        ),
                        # Tab: Feature Explanation
                        dbc.Tab(
                            label="Feature Explanation",
                            tab_id="tab-feature-explanation",
                            children=[
                                # SHAP local
                                html.Div(
                                    [
                                        dbc.Label(
                                            "Select features number",
                                            html_for="max-feature-count",
                                        ),
                                        dcc.Dropdown(
                                            id="max-feature-count",
                                            options=[
                                                {"label": str(n), "value": n}
                                                for n in [5, 10, 15, 20]
                                            ],
                                            value=5,
                                            clearable=False,
                                            style={
                                                "width": "6rem",
                                                "marginBottom": "1rem",
                                            },
                                        ),
                                        html.Div(
                                            id="feature-importance-local",
                                            role="region",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                                # Violin or bar plots
                                html.Div(
                                    [
                                        dbc.Label(
                                            "Group clients by",
                                            html_for="group-column",
                                        ),
                                        dcc.Dropdown(
                                            id="group-column",
                                            options=[
                                                {"label": "", "value": ""}
                                            ]
                                            + [
                                                {
                                                    "label": c,
                                                    "value": c,
                                                }
                                                for c in groupable_columns
                                            ],
                                            value="",
                                            clearable=False,
                                            style={
                                                "width": "100%",
                                                "maxWidth": "20rem",
                                                "marginBottom": "1rem",
                                            },
                                        ),
                                        html.Div(
                                            id="plots",
                                            role="region",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                            ],
                        ),
                        # Tab: Custom graphs
                        dbc.Tab(
                            label="Customs columns",
                            tab_id="tab-custon-columns",
                            children=[
                                html.Div(
                                    [
                                        dbc.Label(
                                            "Features",
                                            html_for="custom-columns",
                                        ),
                                        dcc.Dropdown(
                                            id="custom-columns",
                                            options=[],
                                            multi=True,
                                            placeholder=(
                                                "Choose features to "
                                                + "display graphs"
                                            ),
                                            style={
                                                "width": "100%",
                                                "marginBottom": "1rem",
                                            },
                                        ),
                                        html.Div(
                                            id="custom-plots",
                                            role="region",
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ],
    style={
        "maxWidth": "100%",
        "margin": "0 auto",
        "padding": "2rem 1rem",
        "backgroundColor": "#f2f2f2",
    },
)


@callback(
    [
        Output("custom-columns", "options"),
        Output("custom-columns", "value"),
    ],
    [
        Input("client-id", "value"),
    ],
)
def update_custom_columns_options(client_id):
    if client_id is None:
        raise PreventUpdate
    _, obs_index = get_row_and_index(df_full, "SK_ID_CURR", client_id)

    shap_vals = shap_values_full.values[obs_index]
    feature_names = list(shap_values_full.feature_names)

    ordered = np.argsort(np.abs(shap_vals))[::-1]
    top_features = [
        fn for i in ordered if (fn := feature_names[i]) in df_full.columns
    ]

    options = [{"label": feat, "value": feat} for feat in top_features]

    return options, []


@callback(
    Output("client-summary", "children"),
    Output("proba-vs-threshold", "figure"),
    Output("feature-importance-local", "children"),
    Input("client-id", "value"),
    Input("max-feature-count", "value"),
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
        summary = html.Div(
            [
                html.Span("Predicted score: ", style={"fontWeight": "bold"}),
                html.Span(f"{positive_proba:.2f}"),
                html.Br(),
                html.Span("Threshold: ", style={"fontWeight": "bold"}),
                html.Span(f"{threshold:.2f}"),
                html.Br(),
                html.Span("Decision: ", style={"fontWeight": "bold"}),
                html.Span(
                    decision,
                    style={
                        "color": "#28a745"
                        if decision == "Credit validated"
                        else "#dc3545",
                        "fontWeight": "600",
                        "marginLeft": "5px",
                    },
                ),
            ]
        )

        return summary, figure, dcc.Graph(figure=fig_shap)

    except Exception as e:
        print(f"[ERROR update_dashboard] {e}")
        return (
            html.Div(f"Error API: {e}"),
            {},
            html.Div("Error during SHAP construction"),
        )


@callback(
    Output("plots", "children"),
    Input("client-id", "value"),
    Input("max-feature-count", "value"),
    Input("group-column", "value"),
)
def update_plots(client_id, max_features, group_column):
    if client_id is None or group_column is None:
        raise PreventUpdate

    try:
        row, obs_index = get_row_and_index(df_full, "SK_ID_CURR", client_id)
        shap_vals = shap_values_full.values[obs_index]
        feature_names = shap_values_full.feature_names
        top_features = get_top_features(
            shap_vals, feature_names, df_full, max_features
        )

        if group_column:
            groups_raw = df_full[group_column].dropna().unique().tolist()
            # Reorder day if group_column contains days
            groups = (
                [d for d in WEEKDAY_ORDER if d in groups_raw]
                if group_column == "WEEKDAY_APPR_PROCESS_START"
                else sorted(groups_raw)
            )
        else:
            groups = ["All clients"]
        cards = []
        for feature in top_features:
            client_val = row[feature]
            client_group = row[group_column] if group_column else None
            is_numeric = pd.api.types.is_numeric_dtype(df_full[feature])

            if is_numeric:
                unique_vals = df_full[feature].nunique()
                numeric_to_categorical = unique_vals < LOW_UNIQUE_THRESHOLD
            else:
                numeric_to_categorical = False

            fig = go.Figure()

            for group in groups:
                add_group_trace(
                    fig,
                    df_full,
                    feature,
                    group_column,
                    group,
                    numeric_to_categorical,
                    client_val,
                    client_group,
                )

            fig.update_layout(
                **build_layout(feature, group_column, numeric_to_categorical)
            )
            cards.append(
                Card(
                    title=feature,
                    children=[
                        dcc.Graph(
                            figure=fig,
                            config={"displayModeBar": False},
                            style={"height": "35vh"},
                        )
                    ],
                    style={
                        "flex": "1 1 45%",
                        "margin": "0.5rem",
                    },
                )
            )

        return html.Div(
            cards,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "center",
                "gap": "1rem",
            },
        )

    except Exception as e:
        print(f"[ERROR updating plots] {e}")
        return html.Div("Error during plots construction")


@callback(
    Output("custom-plots", "children"),
    Input("client-id", "value"),
    Input("custom-columns", "value"),
    Input("group-column", "value"),
    Input("max-feature-count", "value"),
)
def render_custom_plots(client_id, columns, group_column, max_features):
    if client_id is None or columns is None or group_column is None:
        raise PreventUpdate

    row, _ = get_row_and_index(df_full, "SK_ID_CURR", client_id)

    if group_column:
        groups_raw = df_full[group_column].dropna().unique().tolist()
        # Reorder day if group_column contains days
        groups = (
            [d for d in WEEKDAY_ORDER if d in groups_raw]
            if group_column == "WEEKDAY_APPR_PROCESS_START"
            else sorted(groups_raw)
        )
    else:
        groups = ["All clients"]

    figures = []
    for _, feature in enumerate(columns):
        client_val = row[feature]
        client_group = row[group_column] if group_column else None
        is_numeric = pd.api.types.is_numeric_dtype(df_full[feature])

        if is_numeric:
            unique_vals = df_full[feature].nunique()
            numeric_to_categorical = unique_vals < LOW_UNIQUE_THRESHOLD
        else:
            numeric_to_categorical = False

        fig = go.Figure()

        for group in groups:
            add_group_trace(
                fig,
                df_full,
                feature,
                group_column,
                group,
                numeric_to_categorical,
                client_val,
                client_group,
            )

        fig.update_layout(
            **build_layout(feature, group_column, numeric_to_categorical)
        )
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
