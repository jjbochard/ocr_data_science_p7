import os
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import requests
import shap
from dash import html

# from dash import html
from dotenv import load_dotenv
from pandas.api.types import is_numeric_dtype
from scipy.special import expit

pio.templates.default = "plotly_white"
load_dotenv(override=True)

TRACKING_URI = os.getenv("TRACKING_URI")
FINAL_RUN = os.getenv("FINAL_RUN")

mlflow.set_tracking_uri(TRACKING_URI)


def load_model_and_data() -> Tuple[
    shap.Explanation, np.ndarray, pd.DataFrame, pd.DataFrame, shap.Explainer
]:
    df = pd.read_csv(
        "data/home_credit_selected_features.csv.gz", compression="gzip"
    )
    # TODO: Remove this line when app in production
    # Filter only the first 10 rows for performance
    df = df.iloc[:1000].copy()

    # Load model
    model_uri = f"runs:/{FINAL_RUN}/final_model"
    pipeline = mlflow.sklearn.load_model(model_uri)

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["processor"]

    df_no_id = df.drop(columns=["SK_ID_CURR"])
    X_transformed = preprocessor.transform(df_no_id)

    # Clean column names (last part after __)
    X_df = pd.DataFrame(X_transformed)
    X_df.columns = X_df.columns.map(lambda col: col.split("__")[-1])

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)

    # Predict probabilities
    proba = model.predict_proba(X_df)[:, 1]

    # Get unique values per object columns
    allowed_values_per_objects_columns = {
        col: df[col].dropna().unique().tolist()
        for col in df.columns
        if df[col].dtype == "object" or df[col].dtype.name == "category"
    }
    groupable_columns = [
        col
        for col in df.select_dtypes(include=["object", "category"]).columns
        if df[col].nunique() <= 10
    ]

    return (
        shap_values,
        proba,
        df,
        X_df,
        explainer,
        groupable_columns,
        allowed_values_per_objects_columns,
    )


def transform_shap_to_proba(shap_values, pred_proba, i):
    """
    Transform SHAP values from log-odds to approximate probability scale.
    """
    base_value_logodds = shap_values.base_values[i]
    base_proba = expit(base_value_logodds)

    # Calculate explain distance
    delta_model = pred_proba[i] - base_proba
    delta_shap = np.sum(shap_values.values[i])

    # Ratio
    scale = delta_shap / delta_model if delta_model != 0 else 1

    # Rescale contributions
    values_scaled = shap_values.values[i] / scale

    return shap.Explanation(
        values=values_scaled,
        base_values=base_proba,
        data=shap_values.data[i],
        feature_names=shap_values.feature_names,
    )


def make_score_figure(score: float, title: str, threshold: float) -> go.Figure:
    """
    Create a Plotly bar chart showing a single prediction score,
    with a dashed red line representing the decision threshold.

    Args:
        score (float): The predicted probability to display.
        title (str): Title of the chart (e.g., 'Score BEFORE modification').
        threshold (float): Decision threshold to compare the score against.

    Returns:
        go.Figure: A Plotly figure object ready to be rendered.
    """
    fig = go.Figure()

    fig.add_bar(
        x=[title],
        y=[score],
        text=[f"{score:.2f}"],
        textposition="auto",
    )

    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=0.5,
        y0=threshold,
        y1=threshold,
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=0,
        y=threshold,
        text=f"Threshold = {threshold:.2f}",
        showarrow=False,
        font=dict(color="red"),
        yshift=10,
    )

    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1]),
        showlegend=False,
    )

    return fig


def get_row_and_index(
    df: pd.DataFrame, id_column: str, client_id: int
) -> Tuple[pd.Series, int]:
    """
    Retrieve the data row and its index for a specific client ID.

    Args:
        df (pd.DataFrame): The full dataset containing client records.
        id_column (str): The name of the column containing client IDs
        client_id (int): The client ID to retrieve.

    Returns:
        Tuple[pd.Series, int]: A Series with the client's feature values
        (excluding the ID),
        and the index of the row in the original DataFrame.
    """
    row = df[df[id_column] == client_id].drop(columns=[id_column])
    index = df[df[id_column] == client_id].index[0]
    return row.squeeze(), index


def call_prediction_api(features: dict) -> dict:
    """
    Send a feature dictionary to the prediction API and return the result.

    Args:
        features (dict): A dictionary of feature names and their values.

    Returns:
        dict: The API response as a JSON dictionary.
        If the call fails, a fallback dict is returned.
    """
    try:
        payload = {"features": features}
        response = requests.post("http://localhost:8000/predict", json=payload)
        response.raise_for_status()
    except Exception as e:
        return (html.Div(f"Erreur API: {e}"), {}, {})
    return response.json()


def get_feature_differences(
    original: dict, modified: dict
) -> List[Tuple[str, any, any]]:
    """
    Compare two dictionaries of feature values and
    return a list of differences.

    Args:
        original (dict): The original feature values.
        modified (dict): The modified feature values (after user input).

    Returns:
        List[Tuple[str, any, any]]: A list of tuples containing
        (feature name, original value, modified value)
        for all features that have changed and are not both NaN.
    """
    return [
        (key, original[key], modified[key])
        for key in original
        if original[key] != modified.get(key)
        and not (pd.isna(original[key]) and pd.isna(modified.get(key)))
    ]


def get_top_features(shap_values, feature_names, df, max_features):
    top = sorted(
        zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
    )
    names = [name for name, _ in top[:max_features]]
    return [f for f in names if f in names]


def build_color_map(groups):
    palette = px.colors.qualitative.Plotly
    return {grp: palette[i % len(palette)] for i, grp in enumerate(groups)}


def build_layout(feature, group_column, use_bar):
    title = f"{feature}"
    if group_column:
        title += f"<br>by {group_column}"

    cfg = {
        "title": title,
        "xaxis": {"showticklabels": True},
    }
    if not use_bar:
        cfg.update({"violingap": 0.3, "violingroupgap": 0.4})
    return cfg


def add_horizontal_bar(fig, y_labels, counts, feature):
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=counts,
            orientation="h",
            hovertemplate=f"{feature}: <b>%{{y}}</b><br>Count: <b>%{{x}}"
            + "</b><extra></extra>",
            text=counts,
            textposition="outside",
        )
    )
    fig.update_traces(cliponaxis=False)
    fig.update_yaxes(automargin=True, showgrid=False)
    fig.update_xaxes(automargin=True)

    fig.update_layout(
        yaxis_title=feature,
        xaxis_title="Count",
        yaxis=dict(type="category"),
    )


def add_heatmap(fig, df, feature, group_column):
    pivot = pd.crosstab(df[feature], df[group_column])
    pivot_t = pivot.T

    x_labels = [
        str(int(x))
        if isinstance(x, (int, float)) and float(x).is_integer()
        else str(x)
        for x in pivot_t.columns
    ]

    y_labels = pivot_t.index.tolist()
    z_values = pivot_t.values

    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=z_values,
            texttemplate="%{text}",
            colorscale="Cividis",
            hovertemplate=f"{feature}: %{{x}}<br>{group_column}: %{{y}}<br>"
            + "Count: %{{z}}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title=group_column,
        xaxis=dict(type="category", automargin=True),
        yaxis=dict(type="category", automargin=True),
    )
    fig.update_xaxes(autotickangles=[0, 45, 60, 90])


def add_violin_plot(fig, x_labels, y_values, name):
    fig.add_trace(
        go.Violin(
            x=x_labels,
            y=y_values,
            name=name,
            box_visible=True,
            meanline_visible=True,
            points=False,
            showlegend=False,
        )
    )


def add_group_trace(
    fig,
    df,
    feature,
    group_column,
    group,
    numeric_to_categorical,
):
    mask = df[group_column] == group if group_column else slice(None)
    vals = df.loc[mask, feature]
    size = mask.sum() if group_column else len(df)

    is_numeric = is_numeric_dtype(vals)

    if not is_numeric:
        if group_column:
            add_heatmap(fig, df, feature, group_column)
        else:
            vc = vals.value_counts().sort_values(ascending=True)
            add_horizontal_bar(
                fig,
                vc.index.to_list(),
                vc.values.tolist(),
                feature,
            )

    else:
        if numeric_to_categorical:
            if not group_column:
                vc = vals.value_counts().sort_values(ascending=True)
                labels = list(map(int, vc.index))
                add_horizontal_bar(
                    fig,
                    labels,
                    vc.values.tolist(),
                    feature,
                )

            else:
                add_heatmap(fig, df, feature, group_column)
        else:
            size = len(vals)
            group_label = f"{group} ({size})"
            add_violin_plot(
                fig, [group_label] * size, vals.tolist(), group_label
            )
