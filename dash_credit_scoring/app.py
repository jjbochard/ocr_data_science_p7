from dash import Dash, dcc, html, page_container

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
app.title = "Credit Scoring Dashboard"


app.layout = html.Div(
    [
        html.H2("Dashboard Credit"),
        dcc.Link("Home", href="/home"),
        dcc.Link("Simulation", href="/simulate"),
        html.Hr(),
        page_container,
    ]
)
if __name__ == "__main__":
    app.run(debug=True)
