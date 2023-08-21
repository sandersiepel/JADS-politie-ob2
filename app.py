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
from datetime import datetime


# Initialize parameters.
data_source = "routined"  # Can be either 'google_maps' or 'routined'.
# hours_offset is used to offset the timestamps to account for timezone differences. For google maps, timestamp comes in GMT+0
# which means that we need to offset it by 2 hours to make it GMT+2 (Dutch timezone). Value must be INT!
hours_offset = 0 # Should be 0 for routined and 2 for google_maps. 
# begin_date and end_date are used to filter the data for your analysis.
begin_date = "2023-05-01"
end_date = "2023-08-01"  # End date is INclusive! 
# FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
fraction = 1
# For the model performance class we need to specify the number of training days (range) and testing horizon (also in days)
training_window_size = 60
horizon_size = 21
window_step_size = 1
outputs_folder_name = f"dash1-{training_window_size}-{horizon_size}-{window_step_size}" # All of the outputs will be placed in output/outputs_folder_name

log_messages = deque(maxlen=20)  
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "background-color": "#f8f9fa",
    "padding": "20px"
}

CONTENT_STYLE = {
    "padding": "20px"
}


maindiv = html.Div([
    dbc.Row([
        html.H2("Scatter mapbox"),
        dcc.Graph(id="scatter_mapbox_graph")
    ], style={"min-height": "50px", "padding-bottom": "50px"}),
    dbc.Row([
        html.H2("Predicting"),
    ])
], style=CONTENT_STYLE)


sidebar = html.Div(
    [
        html.H2("Settings"),
        html.P(
            "This panel allows you to change the pipeline's settings."
        ),

        # Text inputs
        dbc.Label("Min samples:"),
        dbc.Input(id='min_samples', type='text', value=200),
        html.Br(),

        dbc.Label("Eps:"),
        dbc.Input(id='eps', type='text', value=0.01),
        html.Br(),

        dbc.Label("Min unique days:"),
        dbc.Input(id='min_unique_days', type='text', value=1),
        html.Br(),

        dbc.Button('Run Pipeline', id='submit-val', n_clicks=0, color="primary", className="me-1", style={"width":"100%"}),

        # Elements for log messages.
        html.Div(id='container-button-basic', style={"margin-top":"10px"}),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),  # Update every 2 seconds
        dcc.Markdown(id="log-display", style={"whiteSpace":"pre-wrap"})
    ],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([
    dbc.Row([
        dbc.Col(sidebar, width=3), 
        dbc.Col(maindiv, width=9) 
    ])
])

@callback(
    Output('scatter_mapbox_graph', 'figure'),
    [Input('submit-val', 'n_clicks'),
    State('min_samples', 'value'),
    State('eps', 'value'),
    State('min_unique_days', 'value')]
)
def run_pipeline(n_clicks, min_samples, eps, min_unique_days):
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

        # Step 2. Run clustering. Returns df and a fig with a scattermapbox. 
        df, fig = run_clustering(df, int(min_samples), float(eps), int(min_unique_days))
        # fig = fig.update_layout()

        # Step 3. Transform data
        add_log_message(f"Transforming and resampling the dataset")
        df = DT.transform_start_end_times(df, outputs_folder_name, fill_gaps=True)

        # Step 4. Resample dataset. This saves the data at output/outputs_folder_name/resampled_df_10_min.xlsx.
        df = DT.resample_df(df, outputs_folder_name)
        add_log_message(f"Done, saving datasets at output/{outputs_folder_name}")

        # # Step 5. Train models and save performance
        # add_log_message(f"Training and evaluating the model performance")
        # scores, _ = TrainAndEvaluate(
        #     df = df,
        #     outputs_folder_name=outputs_folder_name,
        #     start_date = pd.to_datetime(f"{begin_date} 00:00:00"),
        #     end_date = pd.to_datetime(f"{end_date} 23:50:00"),
        #     training_window_size = training_window_size,
        #     horizon_size = horizon_size,
        #     window_step_size = window_step_size,
        #     model_features = ["day", "hour", "weekday", "window_block"],
        # ).main()

        # # Step 6. Visualize performance and save images
        # add_log_message(f"Visualizing model performance")
        # ModelPerformanceVisualizer(
        #     scores=scores,
        #     outputs_folder_name=outputs_folder_name
        # )

        # add_log_message(f"Pipeline finished!")
        # return "Pipeline finished!"    
        return fig
    else:
        return {}


def run_clustering(df, min_samples, eps, min_unique_days):
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
        min_unique_days=min_unique_days,  # If post_filter = True, then delete all clusters that have been visited on less than min_unique_days days.
    )

    # Then we run the clustering and visualisation
    df = (
        c.run_clustering(
            min_samples=min_samples,  # The number of samples in a neighborhood for a point to be considered as a core point
            eps=eps,  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. 0.01 = 10m
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

    return df, c.fig

@callback(
    [Output("log-display", "children"),
    Output("interval-component", "n_intervals")],
    Input("interval-component", "n_intervals")
)
def update_log_display(n_intervals):
    global log_messages
    log_texts = "\n".join(log_messages)
    return log_texts, n_intervals

# Function to add a new log message
def add_log_message(message):
    global log_messages
    log_messages.append(get_time() + message)

def get_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S") + ": "
    

if __name__ == '__main__':
    app.run(debug=True)