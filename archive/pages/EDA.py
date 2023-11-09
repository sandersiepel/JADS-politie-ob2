import dash
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import DataLoader as DL
import DataTransformer as DT
import shared

# The Homepage is the EDA view. 

data_source = "routined"  # Can be either 'google_maps' or 'routined'.
hours_offset = 0 # Should be 0 for routined and 2 for google_maps. 
begin_date = "2023-05-01" # Begin_date and end_date filter the loaded dataset
end_date = "2023-10-02"
fraction = 1
outputs_folder_name = "politiedemo" # All of the outputs will be placed in output/outputs_folder_name
# predictability_graph_rolling_window_size = 5 # See docstring of Visualizations.DataPredictability for more info on this parameter.


dash.register_page(__name__, path='/')

layout = html.Div([
    dcc.Store(id='data-store'),
    html.H1('Exploratory Data Analysis'),
    html.Div('This page allows you to perform basic EDA on the dataset that is loaded.'), html.Br(),
    dbc.Button('Run EDA', id='submit-val', n_clicks=0, color="primary", className="me-1"),
    html.Hr(),
    dcc.Graph(id="counts_per_day"),
])

@callback(
    [
        Output('counts_per_day', 'figure'),
        Output('data-store', 'data')
    ],
    Input('submit-val', 'n_clicks'),
    prevent_initial_call=True
)
def run_eda(_):   
    shared.add_log_message("Start loading dataset")

    df, fig = DL.load_data(
        data_source,
        begin_date,
        end_date,
        fraction,
        hours_offset,
        outputs_folder_name=outputs_folder_name,
        verbose=True,
        perform_eda=True
    )

    shared.add_log_message(f"Loaded dataset with size: {len(df)}")

    return fig, df.to_dict('records')