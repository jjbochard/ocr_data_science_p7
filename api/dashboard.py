import dash
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from dash import dcc, html

# Download data from github
# url = "https://github.com/jjbochard/ocr_data_science_p7/releases/download/1.0.0/home_credit_selected_features.csv.gz"
# df = pd.read_csv(url, compression="gzip")
df = pd.read_csv("data/home_credit_selected_features_preview.csv")
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
    ]
)


@app.callback(
    [
        dash.Output("client-summary", "children"),
        dash.Output("proba-vs-threshold", "figure"),
    ],
    [dash.Input("client-id", "value")],
)
def update_dashboard(client_id):
    if client_id is None:
        return "", {}

    client_row = df[df["SK_ID_CURR"] == client_id]

    client_row.drop(columns=["SK_ID_CURR"], inplace=True)
    features = client_row.squeeze().to_dict()

    # Manage NaN values
    features = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in features.items()
    }
    payload = {"features": features}

    try:
        api_url = "http://localhost:8000/predict"
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()

        positive_proba = result.get("predict_proba", 0.0)[0][1]

        threshold = result.get("threshold", 0.5)

        # Graph for predicted score vs threshold
        fig = px.bar(
            x=[
                "Client score",
            ],
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
        summary = f"Score client predicted : {positive_proba:.2f} | Threshold : {threshold:.2f} → {decision}"

        return summary, fig

    except Exception as e:
        return f"API Error: {e}", {}


if __name__ == "__main__":
    app.run(debug=True)
