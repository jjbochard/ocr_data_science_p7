import pandas as pd
from dash import dcc, html

# Chargement des IDs clients depuis le fichier local
df = pd.read_csv(
    "data/home_credit_selected_features.csv.gz", compression="gzip"
)
client_ids = df["SK_ID_CURR"].tolist()

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
