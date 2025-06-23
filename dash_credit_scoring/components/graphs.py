import numpy as np
import plotly.graph_objects as go


def shap_waterfall_plot(
    shap_values_transformed, max_display=10, stable_threshold=0.001
):
    values = shap_values_transformed.values
    features = np.array(shap_values_transformed.feature_names)
    data_values = shap_values_transformed.data
    base_value = shap_values_transformed.base_values
    fx = base_value + values.sum()

    # Descending sort by absolute importance
    sorted_inds = np.argsort(-np.abs(values))
    features = features[sorted_inds]
    values = values[sorted_inds]
    data_values = data_values[sorted_inds]

    if max_display < len(values):
        number_features_rest = len(values) - max_display
        rest = values[max_display:].sum()
        features = np.append(
            features[:max_display],
            "Count other features",
        )
        values = np.append(values[:max_display], rest)
        data_values = np.append(
            data_values[:max_display], f"{number_features_rest}"
        )
    else:
        features = features[:max_display]
        values = values[:max_display]
        data_values = data_values[:max_display]

    starts = np.cumsum([0] + list(values[:-1])) + base_value
    y_labels = [f"{f} = {d}" for f, d in zip(features, data_values)]

    fig = go.Figure()

    for i in range(len(values)):
        color = (
            "green"
            if abs(values[i]) < stable_threshold
            else "red"
            if values[i] > 0
            else "royalblue"
        )
        fig.add_trace(
            go.Bar(
                y=[y_labels[i]],
                x=[values[i]],
                orientation="h",
                base=starts[i],
                marker_color=color,
                text=f"{values[i]:+.3f}",
                hovertemplate=f"{y_labels[i]}<br>Contribution: "
                + f"{values[i]:+.3f}<extra></extra>",
            )
        )

    fig.add_vline(
        x=base_value,
        line_dash="dot",
        annotation_text=f"Base value ({base_value:.3f})",
        annotation_position="bottom left",
    )
    fig.add_vline(
        x=fx,
        line_dash="dash",
        annotation_text=f"f(x) = {fx:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="SHAP Waterfall (contribution to proba)",
        xaxis_title="Cumulative contribution",
        barmode="stack",
        showlegend=False,
    )
    fig.update_yaxes(automargin=True)

    return fig
