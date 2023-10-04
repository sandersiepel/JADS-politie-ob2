import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__)

layout = html.Div([
    html.H1('Clustering'),
    html.Div('This page allows you to perform clustering to identify the subject\'s significant locations.'), html.Hr()
    
])