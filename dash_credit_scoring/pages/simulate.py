import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import ALL, Input, Output, State, callback, dcc, html, register_page
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from utils import (
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
        html.H3("Do a simulation"),
        dcc.Dropdown(
            id="client-id",
            options=[{"label": str(i), "value": i} for i in client_ids],
            placeholder="Choose a client ID",
        ),
        dcc.Dropdown(
            id="max-form-fields",
            options=[
                {"label": f"{n} features", "value": n}
                for n in [5, 10, 15, 20, 25, 50]
            ],
            value=5,
            clearable=False,
            style={"marginTop": "10px"},
        ),
        html.Div(id="editable-features"),
        html.Button("Recalculate", id="recalculate-btn", n_clicks=0),
        html.Button(
            "Reset",
            id="reset-btn",
            n_clicks=0,
            style={"marginTop": "10px"},
        ),
        html.Div(id="client-summary-simulate"),
        dcc.Graph(
            id="proba-before-simulate",
            style={"display": "inline-block", "width": "49%"},
        ),
        dcc.Graph(
            id="proba-after-simulate",
            style={"display": "inline-block", "width": "49%"},
        ),
    ]
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
    feature_values, obs_index = get_row_and_index(
        df_full, "SK_ID_CURR", client_id
    )

    # SHAP values for the selected client
    shap_vals = shap_values_full.values[obs_index]
    feature_names = shap_values_full.feature_names

    # Rank features by absolute SHAP value
    ranked_features = sorted(
        (
            (name, shap_val, feature_values[name])
            for name, shap_val in zip(feature_names, shap_vals)
        ),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Select top N features
    top_features = ranked_features[:max_fields]

    # Create input fields
    return [
        html.Div(
            [
                html.Label(name),
                dcc.Dropdown(
                    id={"type": "input-feature", "feature": name},
                    options=[
                        {"label": str(val), "value": val}
                        for val in allowed_values_per_feature[name]
                    ],
                    value=None if pd.isna(value) else value,
                    placeholder=f"Select {name}",
                    style={"marginBottom": "5px", "width": "100%"},
                ),
            ]
        )
        if name in allowed_values_per_feature
        else html.Div(
            [
                html.Label(name),
                dcc.Input(
                    id={"type": "input-feature", "feature": name},
                    type="text",
                    value="" if pd.isna(value) else str(value),
                    style={"marginBottom": "5px", "width": "100%"},
                ),
            ]
        )
        for name, _, value in top_features
    ]


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
    original_features, obs_index = get_row_and_index(
        df_full, "SK_ID_CURR", client_id
    )

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
        modified_features[key] = (
            val.item() if isinstance(val, (np.integer, np.floating)) else val
        )

    # Clean values
    for k, v in modified_features.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            modified_features[k] = None

    # API call
    result = call_prediction_api(modified_features)

    # Results
    proba_after = result.get("predict_proba")[0][1]
    threshold = result.get("threshold", 0.5)
    decision = (
        "Credit refused" if proba_after >= threshold else "Credit validated"
    )

    # Original score
    proba_before = float(Y_pred_proba_full[obs_index])

    # Get features differences
    diffs = get_feature_differences(original_features, modified_features)

    # Table of differences
    diff_table = html.Table(
        [html.Tr([html.Th("Feature"), html.Th("Avant"), html.Th("Après")])]
        + [
            html.Tr([html.Td(k), html.Td(str(v1)), html.Td(str(v2))])
            for k, v1, v2 in diffs
        ]
    )

    return (
        html.Div(
            [
                html.H4("Results :"),
                html.P(
                    f"Score before : {proba_after:.2f} | "
                    + f"Threshold : {threshold:.2f} → {decision}"
                ),
                html.Hr(),
                html.H5("Modified features :"),
                diff_table,
            ]
        ),
        make_score_figure(
            proba_before, "Score BEFORE modification", threshold
        ),
        make_score_figure(proba_after, "Score AFTER modification", threshold),
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
