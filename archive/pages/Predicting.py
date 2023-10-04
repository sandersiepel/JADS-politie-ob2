import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__)

layout = html.Div([
    html.H1('Predicting'),
    html.Div('This page allows you to make predictions with a Machine Learning model.'), html.Hr()
    
])