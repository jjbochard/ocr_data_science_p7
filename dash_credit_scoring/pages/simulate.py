import math
from typing import List, Tuple, Union

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import ALL, Input, Output, State, callback, dcc, html, register_page
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from utils import (
    Card,
    build_layout,
    call_prediction_api,
    get_feature_differences,
    get_row_and_index,
    load_model_and_data,
    make_score_figure,
)

register_page(__name__, path="/simulate", name="Simulation")
(
    shap_values_full,
    Y_pred_proba_full,
    df_full,
    df_transformed,
    explainer,
    _,
    allowed_values_per_feature,
) = load_model_and_data()

client_ids = df_full["SK_ID_CURR"].tolist()

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    Card(
                        title="Simulation Filters",
                        children=[
                            html.Label("Client ID", htmlFor="client-id"),
                            dcc.Dropdown(
                                id="client-id",
                                options=[{"label": str(i), "value": i} for i in client_ids],
                                placeholder="Select client ID",
                                className="mb-3",
                            ),
                            html.Label("Max Features", htmlFor="max-form-fields"),
                            dcc.Dropdown(
                                id="max-form-fields",
                                options=[{"label": f"{n} features", "value": n} for n in [5, 10, 15, 20, 25, 50]],
                                value=5,
                                clearable=False,
                                className="mb-3",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Recalculate",
                                        id="recalculate-btn",
                                        color="primary",
                                        className="me-2",
                                    ),
                                    dbc.Button("Reset", id="reset-btn", color="danger"),
                                ],
                                className="d-flex",
                            ),
                        ],
                    ),
                    md=4,
                    sm=12,
                ),
                dbc.Col(
                    Card(
                        title="Editable Features",
                        children=html.Div(id="editable-features"),
                    ),
                    md=8,
                    sm=12,
                ),
            ],
            className="gy-4 mb-4 align-items-stretch",
        ),
        html.Div(id="client-summary-simulate", className="mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    Card(
                        title="Score Before Simulation",
                        children=[
                            dcc.Graph(
                                id="proba-before-simulate",
                                config={"displayModeBar": False},
                                style={"width": "100%", "height": "40vh"},
                            )
                        ],
                    ),
                    md=6,
                    sm=12,
                ),
                dbc.Col(
                    Card(
                        title="Score After Simulation",
                        children=[
                            dcc.Graph(
                                id="proba-after-simulate",
                                config={"displayModeBar": False},
                                style={"width": "100%", "height": "40vh"},
                            )
                        ],
                    ),
                    md=6,
                    sm=12,
                ),
            ],
            className="gx-4 gy-4 justify-content-center",
        ),
    ],
    style={"padding": "2rem 1rem", "backgroundColor": "#f2f2f2"},
)


@callback(
    Output("editable-features", "children"),
    Input("client-id", "value"),
    Input("max-form-fields", "value"),
)
def display_editable_form(client_id: int, max_fields: int) -> List[html.Div]:
    """
    Generate a dynamic form with editable input fields
    for the most important features of a selected client.

    This callback is triggered when a client ID is selected
    or when the number of form fields is changed.
    It computes SHAP values for the selected client,
    ranks the features by absolute importance, and displays
    up to `max_fields` of them as editable input fields.

    Args:
        client_id (int): The ID of the selected client.
        max_fields (int): The maximum number of editable features to display.

    Returns:
        list[html.Div]: A list of HTML elements (labels + input fields)
        corresponding to the top SHAP features.
    """

    if client_id is None:
        raise PreventUpdate
    # Get the client row and its index
    feature_values, obs_index = get_row_and_index(df_full, "SK_ID_CURR", client_id)

    # SHAP values for the selected client
    shap_vals = shap_values_full.values[obs_index]
    feature_names = shap_values_full.feature_names

    # Rank features by absolute SHAP value
    ranked_features = sorted(
        ((name, shap_val, feature_values[name]) for name, shap_val in zip(feature_names, shap_vals)),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Select top N features
    top_features = ranked_features[:max_fields]

    # Create input fields
    controls = []
    for name, _, value in top_features:
        if name in allowed_values_per_feature:
            control = html.Div(
                [
                    html.Label(name),
                    dcc.Dropdown(
                        id={"type": "input-feature", "feature": name},
                        options=[{"label": str(val), "value": val} for val in allowed_values_per_feature[name]],
                        value=None if pd.isna(value) else value,
                        placeholder=f"Select {name}",
                        style={"marginBottom": "5px", "width": "100%"},
                    ),
                ]
            )
        else:
            control = html.Div(
                [
                    html.Label(name),
                    dcc.Input(
                        id={"type": "input-feature", "feature": name},
                        type="text",
                        value="" if pd.isna(value) else str(value),
                        style={"marginBottom": "5px", "width": "100%"},
                        className="form-control",
                        placeholder=f"Enter {name}",
                    ),
                ]
            )
        controls.append(control)

    return dbc.Row(
        [dbc.Col(ctrl, md=6, sm=12) for ctrl in controls],
        className="gx-3 gy-2",
    )


@callback(
    [
        Output("client-summary-simulate", "children"),
        Output("proba-before-simulate", "figure"),
        Output("proba-after-simulate", "figure"),
    ],
    Input("recalculate-btn", "n_clicks"),
    State("client-id", "value"),
    State({"type": "input-feature", "feature": ALL}, "value"),
    State({"type": "input-feature", "feature": ALL}, "id"),
    prevent_initial_call=True,
)
def recalculate_prediction(
    n_clicks: int,
    client_id: int,
    values: List[Union[str, float, int, None]],
    ids: List[dict],
) -> Tuple[Component, go.Figure, go.Figure]:
    """
    Recalculate and compare the credit score prediction before
    and after modifications of a subset of input features,
    and render the result.

    Args:
        n_clicks (int): Number of clicks on the recalculate button.
        client_id (int): Selected client ID.
        values (List): Updated values for the features from the editable form.
        ids (List): Corresponding feature identifiers from the form.

    Returns:
        Tuple[Component, go.Figure, go.Figure]:
            Summary div, bar chart before modification,
            bar chart after modification.
    """
    if client_id is None or not ids:
        raise PreventUpdate

    # Original data
    original_features, obs_index = get_row_and_index(df_full, "SK_ID_CURR", client_id)

    original_features = original_features.to_dict()
    modified_features = original_features.copy()

    # Apply modifications
    for val, id_dict in zip(values, ids):
        key = id_dict["feature"]
        if val is None or val == "" or str(val).lower() == "nan":
            modified_features[key] = None
            continue
        try:
            val = pd.to_numeric(val)
        except Exception:
            pass
        modified_features[key] = val.item() if isinstance(val, (np.integer, np.floating)) else val

    # Clean values
    for k, v in modified_features.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            modified_features[k] = None

    # API call
    result = call_prediction_api(modified_features)

    # Results
    proba_after = result.get("predict_proba")[0][1]
    threshold = result.get("threshold", 0.5)
    decision = "Credit refused" if proba_after >= threshold else "Credit validated"

    # Original score
    proba_before = float(Y_pred_proba_full[obs_index])

    # Get features differences
    diffs = get_feature_differences(original_features, modified_features)

    diff_df = pd.DataFrame(diffs, columns=["Feature", "Before", "After"])

    diff_table = dbc.Table.from_dataframe(
        diff_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm",
    )
    results_card = dbc.Card(
        [
            dbc.CardHeader(html.H4("Results")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.P(f"Score BEFORE : {proba_before:.2f}"),
                                width="auto",
                            ),
                            dbc.Col(
                                html.P(f"Threshold : {threshold:.2f}"),
                                width="auto",
                            ),
                            dbc.Col(
                                html.P(
                                    decision,
                                    className=("text-success" if decision == "Credit validated" else "text-danger"),
                                ),
                                width="auto",
                            ),
                        ],
                        className="align-items-center mb-3",
                    ),
                    html.H5("Modified features:", className="mt-3"),
                    diff_table,
                ]
            ),
        ],
        className="shadow-sm mb-4",
    )
    figure_before = make_score_figure(proba_before, "", threshold)
    figure_before.update_layout(**build_layout(use_bar=False))
    figure_after = make_score_figure(proba_after, "", threshold)
    figure_after.update_layout(**build_layout(use_bar=False))
    return (
        results_card,
        figure_before,
        figure_after,
    )


@callback(
    [
        Output("client-id", "value", allow_duplicate=True),
        Output("editable-features", "children", allow_duplicate=True),
        Output("client-summary-simulate", "children", allow_duplicate=True),
        Output("proba-before-simulate", "figure", allow_duplicate=True),
        Output("proba-after-simulate", "figure", allow_duplicate=True),
    ],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_simulation_page(
    n_clicks: int,
) -> Tuple[None, List, None, dict, dict]:
    """
    Reset all interactive fields and graphs on the simulation page.

    Triggered by the reset button click. Clears the selected client,
    editable form inputs, summary, and score comparison charts.
    """
    return None, [], None, {}, {}


@callback(
    Output({"type": "input-feature", "feature": ALL}, "className"),
    Input({"type": "input-feature", "feature": ALL}, "value"),
)
def validate_features(values):
    classes = []
    for v in values:
        if v is None or (isinstance(v, str) and not v.strip()):
            classes.append("is-invalid")
        else:
            classes.append("is-valid")
    return classes
