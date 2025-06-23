from callbacks import register_callbacks
from dash import Dash
from layout import layout

app = Dash(__name__)
app.title = "Credit Scoring Dashboard"
app.layout = layout

register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
