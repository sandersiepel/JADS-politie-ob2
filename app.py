from dash import Dash, dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import DataLoader as DL
from Cluster import Cluster
import DataTransformer as DT
from TrainAndEvaluateV2 import TrainAndEvaluate
import pandas as pd
from Visualisations import ModelPerformanceVisualizer
from collections import deque
import dash_bootstrap_components as dbc


# Initialize parameters.
data_source = "google_maps"  # Can be either 'google_maps' or 'routined'.
# hours_offset is used to offset the timestamps to account for timezone differences. For google maps, timestamp comes in GMT+0
# which means that we need to offset it by 2 hours to make it GMT+2 (Dutch timezone). Value must be INT!
hours_offset = 2 # Should be 0 for routined and 2 for google_maps. 
# begin_date and end_date are used to filter the data for your analysis.
begin_date = "2022-09-01"
end_date = "2023-05-01"  # End date is INclusive! 
# FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
fraction = 1
# For the model performance class we need to specify the number of training days (range) and testing horizon (also in days)
training_window_size = 60
horizon_size = 21
window_step_size = 1
outputs_folder_name = f"dash1-{training_window_size}-{horizon_size}-{window_step_size}" # All of the outputs will be placed in output/outputs_folder_name

################### DASH VARIABLES ########################################################
SIDEBAR_STYLE = {
    "background-color": "#f8f9fa",
    "padding": "20px"
}

CONTENT_STYLE = {
    "padding": "20px"
}

maindiv = html.Div([
    html.H2("Results"),
    html.Hr(),
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Scatter mapbox")
        ]), width=12, style={"min-height": "50px"}),  # adjust width
        dbc.Col(html.Div([
            html.H3("Model performance")
        ]), width=12, style={"min-height": "50px"})  # adjust width
    ])
], style=CONTENT_STYLE)

sidebar = html.Div(
    [
        html.H2("Settings"),
        html.Hr(),
        html.P(
            "This panel allows you to change the pipeline's settings. For information on how to choose the right settings, see the documentation."
        ),
        # Text inputs
        dbc.Label("Training window size:"),
        dbc.Input(id='training_window_size', type='text', value=training_window_size),
        html.Br(),

        dbc.Label("Horizon size:"),
        dbc.Input(id='horizon_size', type='text', value=horizon_size),
        html.Br(),

        dbc.Label("Window step size:"),
        dbc.Input(id='window_step_size', type='text', value=window_step_size),
        html.Br(),

        dbc.Button('Run Pipeline', id='submit-val', n_clicks=0, color="primary", className="me-1", style={"width":"100%"}),
        html.Div(id='container-button-basic',
                 children='Log messages will appear here.'),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),  # Update every 2 seconds
        html.Div(id="log-display")
    ],
    style=SIDEBAR_STYLE,
)



################### DASH VARIABLES ########################################################

log_messages = deque(maxlen=20)  
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row([
        dbc.Col(sidebar, width=3),  # adjust width
        dbc.Col(maindiv, width=9)  # adjust width
    ])
])


@callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('training_window_size', 'value'),
    State('horizon_size', 'value'),
    State('window_step_size', 'value')
)
def run_pipeline(n_clicks, training_window_size, horizon_size, window_step_size):
    if n_clicks > 0:
        
        # Step 1. Load data
        add_log_message("Loading data")

        df = DL.load_data(
            data_source,
            begin_date,
            end_date,
            fraction,
            hours_offset,
            verbose=True,
        )

        add_log_message(f"Loaded the data with size: {len(df)}")

        # Step 2. Run clustering
        df = run_clustering(df)

        # Step 3. Transform data
        add_log_message(f"Transforming and resampling the dataset")
        df = DT.transform_start_end_times(df, outputs_folder_name, fill_gaps=True)

        # Step 4. Resample dataset
        df = DT.resample_df(df, outputs_folder_name)

        # Step 5. Train models and save performance
        add_log_message(f"Training and evaluating the model performance")
        scores, _ = TrainAndEvaluate(
            df = df,
            outputs_folder_name=outputs_folder_name,
            start_date = pd.to_datetime(f"{begin_date} 00:00:00"),
            end_date = pd.to_datetime(f"{end_date} 23:50:00"),
            training_window_size = training_window_size,
            horizon_size = horizon_size,
            window_step_size = window_step_size,
            model_features = ["day", "hour", "weekday", "window_block"],
        ).main()

        # Step 6. Visualize performance and save images
        add_log_message(f"Visualizing model performance")
        ModelPerformanceVisualizer(
            scores=scores,
            outputs_folder_name=outputs_folder_name
        )

        add_log_message(f"Pipeline finished!")
        return "Pipeline finished!"    


def run_clustering(df):
    add_log_message(f"Starting the clustering")
    # Step 2. Run clustering
    c = Cluster(
        df,  # Input dataset (with latitude, longitude, timestamp columns)
        outputs_folder_name=outputs_folder_name, 
        verbose=True,  # Do we want to see print statements?
        pre_filter=True,  # Apply filters to the data before the clustering (such as removing moving points)
        post_filter=True,  # Apply filters to the data/clusters after the clustering (such as deleting homogeneous clusters)
        filter_moving=True,  # Do we want to delete the data points where the subject was moving?
        centroid_k=10,  # Number of nearest neighbors to consider for density calculation (for cluster centroids)
        min_unique_days=1,  # If post_filter = True, then delete all clusters that have been visited on less than min_unique_days days.
    )

    # Then we run the clustering and visualisation
    df = (
        c.run_clustering(
            min_samples=200,  # The number of samples in a neighborhood for a point to be considered as a core point
            eps=0.01,  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. 0.01 = 10m
            algorithm="dbscan",  # Choose either 'dbscan' or 'hdbscan'. If 'hdbscan', only min_samples is required.
            # min_cluster_size=50,  # Param of HDBSCAN: the minimum size a final cluster can be. The higher this is, the bigger your clusters will be
        )
        .plot_clusters(
            filter_noise=False,  # Remove the -1 labels (i.e., noise) before plotting the clusters
            only_include_clusters=[],  # Add clusters if you want to filter which clusters to show in the visualization.
        )
        .add_locations_to_original_dataframe(
            export_xlsx=False,  # Export the dataframe to excel file? Useful for analyzing.
            name="test",
        )
        .df  # These functions return 'self' so we can chain them and easily access the df attribute (for input to further modeling/visualization).
    )

    add_log_message(f"Done with clustering")

    return df

@callback(
    Output("log-display", "children"),
    Output("interval-component", "n_intervals"),
    Input("interval-component", "n_intervals")
)
def update_log_display(n_intervals):
    global log_messages
    log_texts = [html.P(msg) for msg in log_messages]
    return log_texts, n_intervals

# Function to add a new log message
def add_log_message(message):
    global log_messages
    log_messages.append(message)


if __name__ == '__main__':
    app.run(debug=True)