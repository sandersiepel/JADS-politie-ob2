import dash
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import shared

# This is the main file that is being called with "python app2.py". 

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "backgroundColor": "#f8f9fa",
    "padding": "20px",
    "height":"100vh"
}

sidebar = html.Div(
    [
        html.H2("Menu"),
        html.Div([
            html.Div(
                dcc.Link(f"{page['name']}", href=page["relative_path"])
            ) for page in dash.page_registry.values()
        ]),
        dcc.Interval(id='interval-component', interval=300, n_intervals=0),  # Update every 2 seconds
        html.Div(id="log-display", style={"whiteSpace":"pre-wrap", "paddingTop":"15px", "height":"100%", "overflow":"auto"}),
    ],
    style=SIDEBAR_STYLE,
)

CONTENT_STYLE = {
    "padding": "20px",
    "minHeight": "50px",
    "paddingBottom":"20px",
}

maindiv = html.Div([dash.page_container], style=CONTENT_STYLE)

app.layout = html.Div([
    dbc.Row([
        dbc.Col(sidebar, width=2, className="position-fixed"), 
        dbc.Col(maindiv, width=10, className="offset-2"), 
    ])
])

@app.callback(
    Output("log-display", "children"),
    Input("interval-component", "n_intervals")
)
def update_log_display(_):
    log_texts = "\n".join(shared.log_messages)
    return log_texts


if __name__ == '__main__':
    app.run(debug=True)