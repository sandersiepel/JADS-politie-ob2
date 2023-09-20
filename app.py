from dash import Dash, dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import DataLoader as DL
from Cluster import Cluster
import DataTransformer as DT
from TrainAndEvaluateV2 import TrainAndEvaluate
import pandas as pd
from Visualisations import ModelPerformanceVisualizer, EDA, DataPredicatability, HeatmapVisualizer
from collections import deque
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.express as px
import dash
from datetime import date
from sklearn.preprocessing import LabelEncoder as le
from sklearn.ensemble import RandomForestClassifier 


# Initialize parameters.
data_source = "routined"  # Can be either 'google_maps' or 'routined'.
# hours_offset is used to offset the timestamps to account for timezone differences. For google maps, timestamp comes in GMT+0
# which means that we need to offset it by 2 hours to make it GMT+2 (Dutch timezone). Value must be INT!
hours_offset = 0 # Should be 0 for routined and 2 for google_maps. 
# begin_date and end_date are used to filter the data for your analysis.
begin_date = "2023-05-05"
end_date = "2023-09-19"  # End date is INclusive! 
# FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
fraction = 1
# For the model performance class we need to specify the number of training days (range) and testing horizon (also in days)
training_window_size = 100
horizon_size = 30
window_step_size = 1
outputs_folder_name = f"politiedemo-{training_window_size}-{horizon_size}-{window_step_size}" # All of the outputs will be placed in output/outputs_folder_name
predictability_graph_rolling_window_size = 5 # See docstring of Visualizations.DataPredicatability for more info on this parameter.

log_messages = deque(maxlen=5)  
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "backgroundColor": "#f8f9fa",
    "padding": "20px",
    "height":"100vh"
}

CONTENT_STYLE = {
    "padding": "20px"
}


maindiv = html.Div([
    dbc.Row([
        # This row contains the cards for EDA (scatter mapbox, records per day)
        # html.H2("Scatter mapbox"),
        # dcc.Graph(id="scatter_mapbox_graph"),
        html.H2("EDA"),
        # html.P("The tabs below contain results of the clustering steps. The first tap contains a scattermapbox with the raw data and the results of the clustering, indicated by red circles. The second tap contains a graph with the number of data points per day. "),
        # dcc.Graph(id="counts_per_day")
        dbc.Tabs(
            id="eda-tabs",
            active_tab="tab-1",
            children=
            [
                dbc.Tab(dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Markdown(["This graph shows the <span style='color:#0d6efd;' children=\"number of datapoints per day\" /> in the dataset."], dangerously_allow_html=True),
                            dcc.Graph(id="counts_per_day"),
                        ]
                    ), className="border-0"
                ), label="Records Per Day", tab_id="tab-1"),
                
                dbc.Tab(dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Markdown(["This graph shows the <span style='color:#0d6efd;' children=\"identified clusters and their centroids\" /> that were found by the DBSCAN algorithm."], dangerously_allow_html=True),
                            dcc.Graph(id="scatter_mapbox_graph"),
                        ]
                    ), className="border-0"
                ), label="Scatter Mapbox", tab_id="tab-2"),

                dbc.Tab(dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Markdown(["This graph shows the <span style='color:#0d6efd;' children=\"estimated predictability\" /> of the person's locations over time. Analysing this graph is helpful to determine if 1) the person's location behavior is stable and 2) if the person is likely to be predictable."], dangerously_allow_html=True),
                            dcc.Graph(id="predictability_graph"),
                        ]
                    ), className="border-0"
                ), label="Predictability", tab_id="tab-3"),    

                dbc.Tab(dbc.Card(
                    dbc.CardBody(
                        [
                            # dcc.Graph(id="location_history_heatmap"),
                            dcc.Markdown(["This graph shows the <span style='color:#0d6efd;' children=\"visited locations\" /> of the person between two dates. Use the date range below to change the dates."], dangerously_allow_html=True),
                            html.Img(id="location_history_heatmap", style={"width":"100%", "min-height":"300px"}),
                            dcc.DatePickerRange(
                                id='heatmap-picker-range',
                                min_date_allowed=date(2000, 1, 1),
                                max_date_allowed=date(2023, 12, 31),
                                initial_visible_month=date(2023, 1, 1),
                                start_date=date(2023, 1, 1),
                                end_date=date(2023, 1, 7),
                                with_portal=True,
                                number_of_months_shown=3,
                            ),
                        ]
                    ), className="border-0"
                ), label="Location History", tab_id="tab-4"),                
            ], style={"marginLeft":"10px"},
            
        )

    ], style={"minHeight": "50px", "paddingBottom": "20px"}),
    dbc.Row([
        html.H2("Predicting"),
        html.P("After running the clustering, you can make predictions. Select the training period for the model, select the horizon, and click on 'make predictions'. "),
        
        dbc.Col([
            # Add an input field for year selection
            dbc.Label("Select starting year"),
            dcc.Dropdown([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], 2023, id='year-prediction'),
            html.Br(),
            dcc.DatePickerRange(
                id='heatmap-picker-range-prediction',
                min_date_allowed=date(2000, 1, 1),
                max_date_allowed=date(2023, 12, 31),
                initial_visible_month=date(2023, 1, 1),
                with_portal=True,
                clearable=True,
                number_of_months_shown=3,
                start_date_placeholder_text='Start day',
                end_date_placeholder_text='End day',
            ),
            html.Br(),html.Br(),
            dbc.Label("Horizon length in days:"),
            dbc.Input(id='horizon-length', type='text', value=7),
            html.Br(), 
            dbc.Button('Train and predict', id='train-predict-button', n_clicks=0, color="primary", className="me-1", style={"width":"100%"}),

        ], width=2, style={}), 
        dbc.Col([
            html.Img(id="prediction_heatmap", style={"width":"100%", "min-height":"300px"}),
        ], width=10), 
        
    ])
], style=CONTENT_STYLE)


sidebar = html.Div(
    [
        html.H2("Settings"),
        # html.P(
        #     "This panel allows you to change the pipeline's settings."
        # ),

        # Text inputs
        dbc.Label("Min samples:"),
        dbc.Input(id='min_samples', type='text', value=200),
        html.Br(),

        dbc.Label("Eps:"),
        dbc.Input(id='eps', type='text', value=0.02),
        html.Br(),

        dbc.Label("Min unique days:"),
        dbc.Input(id='min_unique_days', type='text', value=1),
        html.Br(),

        dbc.Button('Run Clustering', id='submit-val', n_clicks=0, color="primary", className="me-1", style={"width":"100%"}),

        # Elements for log messages.
        # html.Div(id='container-button-basic', style={"margin-top":"10px", "overflow":"auto", "height":"400px"}),
        dcc.Interval(id='interval-component', interval=300, n_intervals=0),  # Update every 2 seconds
        html.Div(id="log-display", style={"whiteSpace":"pre-wrap", "paddingTop":"15px", "height":"100%", "overflow":"auto"}),

        html.Div(id="output"), # For chaining output of function 
        dcc.Store(id='data-store'),
        dcc.Store(id='data-store2'),
    ],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([
    dbc.Row([
        dbc.Col(sidebar, width=2, className="position-fixed", style={}), 
        dbc.Col(maindiv, width=10, className="offset-2") 
    ])
])


@app.callback(
    Output('heatmap-picker-range-prediction', 'initial_visible_month'), 
    Input('year-prediction', 'value'),
    prevent_initial_call=True,
)
def update_year_prediction(selected_year):
    return date(selected_year, 1, 1)


@app.callback(
    [
        Output('counts_per_day', 'figure'),
        Output('data-store', 'data')
    ],
    Input('submit-val', 'n_clicks'),
    prevent_initial_call=True
)
def run_eda(_):    
    add_log_message(f"Loading the dataset...")

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

    add_log_message(f"Loaded the data with size: {len(df)}")

    return fig, df.to_dict('records')
    

@app.callback(
    [
        Output('scatter_mapbox_graph', 'figure'),
        Output('data-store2', 'data'), 
        Output('eda-tabs', 'active_tab', allow_duplicate=True)
    ],
    [
        Input('counts_per_day', 'figure'),
        Input('data-store', 'data'),
        State('submit-val', 'n_clicks'),
        State('min_samples', 'value'),
        State('eps', 'value'),
        State('min_unique_days', 'value')
    ],
    prevent_initial_call=True  
)
def run_pipeline(_, df, n_clicks, min_samples, eps, min_unique_days):
    if n_clicks <= 0:
        return dash.no_update
    
    df = pd.DataFrame(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    add_log_message(f"Running pipeline...")

    # Step 2. Run clustering. Returns df and a fig with a scattermapbox. 
    df, fig = run_clustering(df, int(min_samples), float(eps), int(min_unique_days))

    # Step 3. Transform data
    add_log_message(f"Transforming and resampling the dataset...")
    df = DT.transform_start_end_times(df, outputs_folder_name, fill_gaps=True)

    # Step 4. Resample dataset. This saves the data at output/outputs_folder_name/resampled_df_10_min.xlsx.
    df = DT.resample_df(df, outputs_folder_name)
    add_log_message(f"Done, saving datasets at output/{outputs_folder_name}")

    return fig, df.to_dict('records'), "tab-2"


@app.callback(
    [
        Output('predictability_graph', 'figure'), 
        Output('eda-tabs', 'active_tab', allow_duplicate=True)
    ],
    [
        Input('scatter_mapbox_graph', 'figure'), 
        Input('data-store2', 'data'), 
    ],
    prevent_initial_call=True
)
def show_predictability(_, data):
    add_log_message("Making predictability graph...")

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = DT.add_temporal_features(df)

    fig = DataPredicatability(df, predictability_graph_rolling_window_size).run()
    add_log_message("Done with predictability graph")
    return fig, "tab-3"


@app.callback(
    [
        Output('location_history_heatmap', 'src', allow_duplicate=True), 
        Output('eda-tabs', 'active_tab', allow_duplicate=True),
        Output('heatmap-picker-range', 'start_date'),
        Output('heatmap-picker-range', 'end_date'),
        Output('heatmap-picker-range', 'initial_visible_month', allow_duplicate=True),
        Output('heatmap-picker-range', 'min_date_allowed'),
        Output('heatmap-picker-range', 'max_date_allowed'),
        Output('heatmap-picker-range-prediction', 'min_date_allowed'),
        Output('heatmap-picker-range-prediction', 'max_date_allowed')
    ],
    [
        Input('predictability_graph', 'figure'), 
        Input('data-store2', 'data'), 
    ],
    prevent_initial_call=True
)
def show_location_history_heatmap(_, data):
    add_log_message("Making location history heatmap...")

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Last day of dataset
    end_day = df.timestamp.max()
    start_day = (end_day - pd.Timedelta(days=6))

    encoded_image = HeatmapVisualizer(
        start_day.strftime("%Y-%m-%d"), end_day.strftime("%Y-%m-%d"), df, outputs_folder_name=outputs_folder_name, verbose=False, name="heatmap", title=""
    ).get_encoded_fig()

    src = f"data:image/png;base64,{encoded_image}"
    add_log_message("Done with location history heatmap")

    # Set the min_date_allowed and max_date_allowed (based on the ranges of our dataset)

    return src, "tab-4", start_day.date(), end_day.date(), start_day.date(), df.timestamp.min().date(), df.timestamp.max().date(), df.timestamp.min().date(), df.timestamp.max().date()


@callback(
    [
        Output('location_history_heatmap', 'src'),
        Output('heatmap-picker-range', 'initial_visible_month')
    ],
    [
        Input('heatmap-picker-range', 'start_date'),
        Input('heatmap-picker-range', 'end_date'),
        Input('data-store2', 'data')
    ],
    prevent_initial_call=True)
def update_location_history_heatmap(start_date, end_date, data):
    add_log_message("Updating location history heatmap...")
    # Do not execute this function if data-store2 has changed, but only when heatmap-picker-range has changed (either start date or end date).
    if not dash.callback_context.triggered[0]['prop_id'] in ['heatmap-picker-range.start_date', 'heatmap-picker-range.end_date']:
        raise PreventUpdate
    
    # TODO: check if data is available (i.e., clustering has been performed)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Updating heatmap, Start date: {start_date}, end date: {end_date}, len data: {len(df)}")

    encoded_image = HeatmapVisualizer(
        start_date, end_date, df, outputs_folder_name=outputs_folder_name, verbose=False, name="heatmap", title=""
    ).get_encoded_fig()

    # why no update? 

    add_log_message("Done with location history heatmap")
    src = f"data:image/png;base64,{encoded_image}"
    return src, datetime.strptime(start_date, "%Y-%m-%d").date()
    

def run_clustering(df, min_samples, eps, min_unique_days):
    add_log_message(f"Starting the clustering...")
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
        .add_locations_to_original_dataframe(
            export_xlsx=False,  # Export the dataframe to excel file? Useful for analyzing.
            name="test",
        )
        .plot_clusters(
            filter_noise=False,  # Remove the -1 labels (i.e., noise) before plotting the clusters
        )
        .df  # These functions return 'self' so we can chain them and easily access the df attribute (for input to further modeling/visualization).
    )

    add_log_message(f"Done with clustering")

    return df, c.fig

@app.callback(
    # Inputs: 
    Output('prediction_heatmap', 'src'),
    [
        Input('train-predict-button', 'n_clicks'),
        Input('heatmap-picker-range-prediction', 'start_date'),
        Input('heatmap-picker-range-prediction', 'end_date'),
        Input('data-store2', 'data'),
        State('horizon-length', 'value'),
    ],
    prevent_initial_call=True
)
def train_model(n_clicks, start_date, end_date, data, horizon_length):
    # Even though we have multiple inputs, this function should only fire when the button 'train-predict-button' is pressed
    if not dash.callback_context.triggered[0]['prop_id'] in ['train-predict-button.n_clicks']:
        raise PreventUpdate
    
    # Load local file if data-store2 doesn't contain any data, or else take data from data-store2 and build df. 
    if data is None:
        add_log_message("Loading training data from local file")
        df = pd.read_excel(f"output/{outputs_folder_name}/resampled_df_10_min.xlsx", index_col=[0])
    else:
        df = pd.DataFrame(data)
        add_log_message("Loading training data from browser storage")
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add temporal features and encode the location labels to integers
    df = DT.add_temporal_features(df)
    label_encoder = le()
    df.location = label_encoder.fit_transform(df.location)

    # Make train_start, train_end, predict_start, predict_end date(time) objects
    train_start = pd.to_datetime(f"{start_date} 00:00:00")
    train_end = pd.to_datetime(f"{end_date} 23:50:00")
    print(f"Training starts at {train_start} and ends at {train_end}")

    train_mask = df["timestamp"].between(train_start, train_end)
    X_train = df.loc[train_mask, ["weekday", "hour", "window_block"]]
    y_train = df.loc[train_mask, "location"]

    # Train model
    add_log_message("Training ML model")
    model = RandomForestClassifier()
    model.fit(X_train, y_train) # , sample_weight=np.linspace(0, 1, len(X_train))

    # Make X_test, starting one day after the last day in the dataset
    start_day = df.timestamp.max() + pd.Timedelta(minutes=10) # Max() always ends at 23:50 so we add 10 mins to get the next day.
    end_day = start_day + pd.Timedelta(days=int(horizon_length)-1, hours=23, minutes=50)

    print(f"Predicting starts at {start_day} and ends at {end_day}")

    # Create a DataFrame with the 'time' column and the 'location' column that holds the predicted locations (strings).
    df_predictions = pd.DataFrame({"timestamp": pd.date_range(start=start_day, end=end_day, freq="10T")})
    df_predictions = DT.add_temporal_features(df_predictions)

    # Make predictions and inverse transform them
    df_predictions['location'] = model.predict(df_predictions[["weekday", "hour", "window_block"]])
    df_predictions['location'] = label_encoder.inverse_transform(df_predictions['location'])

    print(df_predictions.head())
    print(df_predictions.tail())

    # Build heatmap
    add_log_message("Making heatmap with predictions")
    print(f"df predictions min: {df_predictions.timestamp.min().date().strftime('%Y-%m-%d')} and max: {df_predictions.timestamp.max()}")
    encoded_image = HeatmapVisualizer(
        df_predictions.timestamp.min().date().strftime('%Y-%m-%d'), df_predictions.timestamp.max().date().strftime('%Y-%m-%d'), df_predictions[["timestamp", "location"]], outputs_folder_name=outputs_folder_name, verbose=False, name="heatmap", title=""
    ).get_encoded_fig()

    src = f"data:image/png;base64,{encoded_image}"

    return src

@app.callback(
    Output("log-display", "children"),
    Input("interval-component", "n_intervals")
)
def update_log_display(_):
    log_texts = "\n".join(log_messages)
    return log_texts

# Function to add a new log message
def add_log_message(message):
    log_messages.append(get_time() + message)

def get_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S") + ": "
    

if __name__ == '__main__':
    app.run(debug=True)