import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html, page_container

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.layout = html.Div([html.H1("Mon application Dash", className="display-4 text-center my-4")])
server = app.server
app.title = "Credit Scoring Dashboard"


@server.route("/healthz")
def health_check():
    return "OK", 200


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            className="app-container",
            children=[
                html.Nav(
                    [
                        html.Img(
                            src=app.get_asset_url("logo.png"),
                            alt="logo",
                            className="navbar-logo-img",
                        ),
                        dcc.Link(
                            "Home",
                            href="/home",
                            className="navbar-link",
                            id="link-home",
                        ),
                        dcc.Link(
                            "Simulation",
                            href="/simulate",
                            className="navbar-link",
                            id="link-simulate",
                        ),
                    ],
                    className="navbar",
                    role="navigation",
                    **{"aria-label": "Main navigation"},
                ),
                html.Div(page_container, className="main-content"),
            ],
        ),
    ]
)


@app.callback(
    [Output(f"link-{p}", "className") for p in ["home", "simulate"]],
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def set_active_links(pathname):
    classes = []
    mapping = {
        "/home": "link-home",
        "/simulate": "link-simulate",
    }
    for page, _ in mapping.items():
        classes.append("navbar-link active" if pathname.startswith(page) else "navbar-link")
    return classes


print("Dash app is running...")
if __name__ == "__main__":
    if __name__ == "__main__":
        app.run_server(debug=False, host="0.0.0.0", port=8050)
