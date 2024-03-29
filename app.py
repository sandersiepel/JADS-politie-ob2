from dash import Dash, dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import DataLoader as DL
import DataTransformer as DT
import pandas as pd
from Visualisations import DataPredictability, HeatmapVisualizerV2
from collections import deque
import dash_bootstrap_components as dbc
from datetime import datetime
import dash
from datetime import date
from sklearn.preprocessing import LabelEncoder as le
from datetime import datetime
from app_functions import *
import plotly.graph_objs as go


# Initialize parameters.
# begin_date and end_date are used to filter the data for your analysis.
begin_date = "2023-05-01"
end_date = "2024-01-31"  # End date is INclusive!
# FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
fraction = 1
# For the model performance class we need to specify the number of training days (range) and testing horizon (also in days)
outputs_folder_name = (
    f"demo"  # All of the outputs will be placed in output/outputs_folder_name
)
predictability_graph_rolling_window_size = 5  # See docstring of Visualizations.DataPredictability for more info on this parameter.
features_list = {
    "day": "Day of the Month",
    "weekday": "Day of the Week",
    "hour": "Hour of the Day",
    "window_block": "Block of the Hour",
}

log_messages = deque(maxlen=5)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

SIDEBAR_STYLE = {"backgroundColor": "#f8f9fa", "padding": "20px", "height": "100vh"}

CONTENT_STYLE = {"padding": "20px"}


maindiv = html.Div(
    [
        # First a few data store components to save data (mostly dataframes) in between functions.
        dcc.Store(id="data-store"),
        dcc.Store(id="data-store2"),
        dcc.Store(id="data-store3"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Probabilities"), close_button=True),
                dbc.ModalBody(
                    dcc.Graph(id="probabilities-table"), id="placeholderModal"
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="closeButton",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="modal-explanation-probabilities",
            centered=True,
            is_open=False,
            style={"width": "100%"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Explanation: Probability Graph"), close_button=True
                ),
                dbc.ModalBody(
                    dcc.Markdown(
                        [
                            "This graph shows the <span style='color:#0d6efd;' children=\"estimated predictability\" /> of the subject's locations over time. From the predictability graph, two things can be derived. <br><br>First, the higher the values (or magnitude), the better the subject's predictability for that day. Thus, a high value means that for most of the moments on that day, the person's location matched the historical pattern observed in the training dataset. <br><br>Second, the more stable the values, the more stable someone's predictability is. Stability here refers to the consistency or steadiness of the predictability values over time. When the graph line is relatively flat or horizontal, it indicates that the predictability is stable. "
                        ],
                        dangerously_allow_html=True,
                    )
                ),
            ],
            id="modal-predictability-graph",
            centered=True,
            is_open=False,
            style={"width": "100%"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Explanation: Load Dataset"), close_button=True
                ),
                dbc.ModalBody(
                    dcc.Markdown(
                        [
                            "This tab allows you to <span style='color:#0d6efd;' children=\"load your dataset\" />. Make sure that your dataset is placed in the 'data' folder. After selecting your dataset, click on 'load data'. The graph on the right will show you how many data points per day the dataset contains. <br><br>The 'hours offset' option can be used to shift all timestamps by an offset. This is useful if the data was collected in a different timezone."
                        ],
                        dangerously_allow_html=True,
                    )
                ),
            ],
            id="modal-load-data",
            centered=True,
            is_open=False,
            style={"width": "100%"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Explanation: Clustering"), close_button=True
                ),
                dbc.ModalBody(
                    dcc.Markdown(
                        [
                            "This tab allows you to perform the <span style='color:#0d6efd;' children=\"clustering\" />. First, you select the data that you want to use for clustering and subsequent analyses. Then you select the scale for clustering: street, city or country. This will automatically change the min samples and eps parameters. Alternatively, you can manually adjust these parameters. <br><br><b>Eps:</b> The maximum distance two points can be from one another while still belonging to the same cluster. <br><b>Min samples: </b>The fewest number of points required to form a cluster. <br><b>Min unique days: </b> The minimum number of days a clusters was visited to be included in the results <br><br>For eps, a value of 1 represents one kilometer, 0.01 represents 10 meters. Increasing the min samples parameter results in 'more important' clusters. The same goes for increasing the min unique days value. "
                        ],
                        dangerously_allow_html=True,
                    )
                ),
            ],
            id="modal-clustering",
            centered=True,
            is_open=False,
            style={"width": "100%"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Explanation: Location History"), close_button=True
                ),
                dbc.ModalBody(
                    dcc.Markdown(
                        [
                            "This visualization shows the subject's location history, based on the identified significant locations. You can treat this visualization as a historical calendar for the subject (at least, for their significant location visits). "
                        ],
                        dangerously_allow_html=True,
                    )
                ),
            ],
            id="modal-location-history",
            centered=True,
            is_open=False,
            style={"width": "100%"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Explanation: Predicting"), close_button=True
                ),
                dbc.ModalBody(
                    dcc.Markdown(
                        [
                            "This section allows you predict the future visits to the subject's significant locations. First, select the training data. This data is fed to a Machine Learning model, which tries to learn the relationships between the temporal features and the visited locations. After running the ML model, the predictions are shown in the visualization on the right. <br><br>The visualization allows you to click on each data point (i.e., 10-minute window) to see the probabilities for each significant location. <br><br><b>Note:</b> the model predicts your future locations, treating each prediction independently without considering the physical distance between consecutive locations. This means that in the predictions, you might see consecutive predictions that are physically not possible, while separately (isolated from other predictions) they are possible. This is intended behavior. "
                        ],
                        dangerously_allow_html=True,
                    )
                ),
            ],
            id="modal-predicting",
            centered=True,
            is_open=False,
            style={"width": "100%"},
        ),
        # First row is about
        dbc.Row(
            [
                html.H2("Significant Locations"),
                dbc.Tabs(
                    id="eda-tabs",
                    active_tab="tab-1",
                    children=[
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Data source:"),
                                                        dcc.Dropdown(
                                                            options=[
                                                                {
                                                                    "label": "Google Maps",
                                                                    "value": "google_maps",
                                                                },
                                                                {
                                                                    "label": "Routined",
                                                                    "value": "routined",
                                                                },
                                                            ],  # TODO: make these options dynamic, based on the available datasets in the /data folder.
                                                            value="routined",
                                                            id="dd-data-source",
                                                        ),
                                                        html.Br(),
                                                        dbc.Label("Hours offset:"),
                                                        dbc.Input(
                                                            id="i-hours-offset",
                                                            type="text",
                                                            value=0,
                                                        ),
                                                        html.Br(),
                                                        dbc.Button(
                                                            children=["Load Data"],
                                                            outline=True,
                                                            id="btn-load-data",
                                                            n_clicks=0,
                                                            color="primary",
                                                            className="me-1",
                                                            style={"width": "100%"},
                                                        ),
                                                        dbc.Button(
                                                            children=[
                                                                html.I(
                                                                    className="bi bi-info-circle-fill me-2"
                                                                ),
                                                                "Explanation",
                                                            ],
                                                            id="btn-load-data-explanation",
                                                            color="info",
                                                            className="me-1",
                                                            n_clicks=0,
                                                            style={
                                                                "backgroundColor": "white",
                                                                "border": "none",
                                                                "padding": "0px",
                                                                "marginTop": "15px",
                                                            },
                                                        ),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(id="counts_per_day"),
                                                    ],
                                                    width=10,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                className="border-0",
                            ),
                            label="Load Dataset",
                            tab_id="tab-1",
                        ),
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Select data:"),
                                                        dcc.DatePickerRange(
                                                            id="date-range-clustering",
                                                            min_date_allowed=date(
                                                                2000, 1, 1
                                                            ),
                                                            max_date_allowed=date(
                                                                2023, 12, 31
                                                            ),
                                                            initial_visible_month=date(
                                                                2023, 7, 1
                                                            ),
                                                            with_portal=True,
                                                            clearable=True,
                                                            number_of_months_shown=3,
                                                            start_date_placeholder_text="Start day",
                                                            end_date_placeholder_text="End day",
                                                        ),
                                                        html.Br(),
                                                        html.Br(),
                                                        dcc.Dropdown(
                                                            options=[
                                                                {
                                                                    "label": "Scale: street",
                                                                    "value": "street",
                                                                },
                                                                {
                                                                    "label": "Scale: City",
                                                                    "value": "city",
                                                                },
                                                                {
                                                                    "label": "Scale: Country",
                                                                    "value": "country",
                                                                },
                                                            ],  # TODO: make these options dynamic, based on the available datasets in the /data folder.
                                                            value="street",
                                                            id="dd-scale",
                                                        ),
                                                        html.Br(),
                                                        # Text inputs
                                                        dbc.Label("Min samples:"),
                                                        dbc.Input(
                                                            id="min_samples",
                                                            type="text",
                                                            value=200,
                                                        ),
                                                        html.Br(),
                                                        dbc.Label("Eps:"),
                                                        dbc.Input(
                                                            id="eps",
                                                            type="text",
                                                            value=0.02,
                                                        ),
                                                        html.Br(),
                                                        dbc.Label("Min unique days:"),
                                                        dbc.Input(
                                                            id="min_unique_days",
                                                            type="text",
                                                            value=1,
                                                        ),
                                                        html.Br(),
                                                        dbc.Button(
                                                            "Run Clustering",
                                                            outline=True,
                                                            id="btn-clustering",
                                                            n_clicks=0,
                                                            color="primary",
                                                            className="me-1",
                                                            style={"width": "100%"},
                                                        ),
                                                        dbc.Button(
                                                            children=[
                                                                html.I(
                                                                    className="bi bi-info-circle-fill me-2"
                                                                ),
                                                                "Explanation",
                                                            ],
                                                            id="btn-clustering-explanation",
                                                            color="info",
                                                            className="me-1",
                                                            n_clicks=0,
                                                            style={
                                                                "backgroundColor": "white",
                                                                "border": "none",
                                                                "padding": "0px",
                                                                "marginTop": "15px",
                                                            },
                                                        ),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="scatter_mapbox_graph"
                                                        ),
                                                    ],
                                                    width=10,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                className="border-0",
                            ),
                            label="Clustering",
                            tab_id="tab-2",
                        ),
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody(
                                    children=[
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    [
                                                        dcc.Dropdown(
                                                            id="dd-features",
                                                            options=[
                                                                {
                                                                    "label": name,
                                                                    "value": feature,
                                                                }
                                                                for feature, name in features_list.items()
                                                            ],
                                                            value=[
                                                                "window_block",
                                                                "hour",
                                                            ],
                                                            multi=True,
                                                        ),
                                                    ],
                                                    width=10,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Load Graph",
                                                            outline=True,
                                                            id="btn-predictability-graph",
                                                            color="primary",
                                                            className="me-1",
                                                            n_clicks=0,
                                                            style={
                                                                "width": "100%",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                    ],
                                                    width=2,
                                                ),
                                            ]
                                        ),
                                        dbc.Button(
                                            children=[
                                                html.I(
                                                    className="bi bi-info-circle-fill me-2"
                                                ),
                                                "Explanation",
                                            ],
                                            id="btn-predictability-graph-explanation",
                                            color="info",
                                            className="me-1",
                                            n_clicks=0,
                                            style={
                                                "backgroundColor": "white",
                                                "border": "none",
                                                "padding": "0px",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        dcc.Graph(id="predictability_graph"),
                                    ]
                                ),
                                className="border-0",
                            ),
                            label="Predictability Graph",
                            tab_id="tab-3",
                        ),
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        # dcc.Graph(id="location_history_heatmap"),
                                        dcc.DatePickerRange(
                                            id="heatmap-picker-range",
                                            min_date_allowed=date(2000, 1, 1),
                                            max_date_allowed=date(2023, 12, 31),
                                            initial_visible_month=date(2023, 1, 1),
                                            start_date=date(2023, 1, 1),
                                            end_date=date(2023, 1, 7),
                                            with_portal=True,
                                            number_of_months_shown=3,
                                        ),
                                        html.Br(),
                                        dbc.Button(
                                            children=[
                                                html.I(
                                                    className="bi bi-info-circle-fill me-2"
                                                ),
                                                "Explanation",
                                            ],
                                            id="btn-location-history-explanation",
                                            color="info",
                                            className="me-1",
                                            n_clicks=0,
                                            style={
                                                "backgroundColor": "white",
                                                "border": "none",
                                                "padding": "0px",
                                                "marginTop": "15px",
                                            },
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        dcc.Graph(id="location_history_heatmap"),
                                    ]
                                ),
                                className="border-0",
                            ),
                            label="Location History",
                            tab_id="tab-4",
                        ),
                    ],
                    style={"marginLeft": "10px"},
                ),
            ],
            style={"minHeight": "50px", "paddingBottom": "20px"},
        ),
        dbc.Row(
            [
                html.H2("Predicting"),
                dbc.Col(
                    [
                        # Add an input field for year selection
                        dbc.Label("Select starting year"),
                        dcc.Dropdown(
                            [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                            2023,
                            id="year-prediction",
                        ),
                        html.Br(),
                        dcc.DatePickerRange(
                            id="heatmap-picker-range-prediction",
                            min_date_allowed=date(2000, 1, 1),
                            max_date_allowed=date(2023, 12, 31),
                            initial_visible_month=date(2023, 7, 1),
                            with_portal=True,
                            clearable=True,
                            number_of_months_shown=3,
                            start_date_placeholder_text="Start day",
                            end_date_placeholder_text="End day",
                        ),
                        html.Br(),
                        html.Br(),
                        dbc.Label("Horizon length in days:"),
                        dbc.Input(id="horizon-length", type="text", value=7),
                        html.Br(),
                        dbc.Button(
                            "Train and Predict",
                            outline=True,
                            id="train-predict-button",
                            n_clicks=0,
                            color="primary",
                            className="me-1",
                            style={"width": "100%"},
                        ),
                        dbc.Button(
                            children=[
                                html.I(className="bi bi-info-circle-fill me-2"),
                                "Explanation",
                            ],
                            id="btn-predicting-explanation",
                            color="info",
                            className="me-1",
                            n_clicks=0,
                            style={
                                "backgroundColor": "white",
                                "border": "none",
                                "padding": "0px",
                                "marginTop": "15px",
                            },
                        ),
                    ],
                    width=2,
                    style={},
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Graph(id="prediction_heatmap"),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                    width=10,
                ),
            ]
        ),
    ],
    style=CONTENT_STYLE,
)


sidebar = html.Div(
    [
        html.H2("Log Messages"),
        # Elements for log messages.
        # html.Div(id='container-button-basic', style={"margin-top":"10px", "overflow":"auto", "height":"400px"}),
        dcc.Interval(
            id="interval-component", interval=300, n_intervals=0
        ),  # Update every 2 seconds
        html.Div(
            id="log-display",
            style={
                "whiteSpace": "pre-wrap",
                "paddingTop": "15px",
                "height": "100%",
                "overflow": "auto",
            },
        ),
        html.Div(id="output"),  # For chaining output of function
    ],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=2, className="position-fixed", style={}),
                dbc.Col(maindiv, width=10, className="offset-2"),
            ]
        )
    ]
)


@app.callback(
    Output("heatmap-picker-range-prediction", "initial_visible_month"),
    Input("year-prediction", "value"),
    prevent_initial_call=True,
)
def update_year_prediction(selected_year):
    return date(selected_year, 1, 1)


# Disable button load data and show loading
@app.callback(
    [
        Output("btn-load-data", "children", allow_duplicate=True),
        Output("btn-load-data", "disabled", allow_duplicate=True),
    ],
    Input("btn-load-data", "n_clicks"),
    prevent_initial_call=True,
)
def button_loading_state(n_clicks):
    return [dbc.Spinner(size="sm"), " Loading..."], True


@app.callback(
    [
        Output("btn-clustering", "children", allow_duplicate=True),
        Output("btn-clustering", "disabled", allow_duplicate=True),
    ],
    Input("btn-clustering", "n_clicks"),
    prevent_initial_call=True,
)
def button_loading_state(n_clicks):
    return [dbc.Spinner(size="sm"), " Clustering..."], True


@app.callback(
    [
        Output("btn-predictability-graph", "children", allow_duplicate=True),
        Output("btn-predictability-graph", "disabled", allow_duplicate=True),
    ],
    Input("btn-predictability-graph", "n_clicks"),
    prevent_initial_call=True,
)
def button_loading_state(n_clicks):
    return [dbc.Spinner(size="sm"), " Loading..."], True


@app.callback(
    [
        Output("train-predict-button", "children", allow_duplicate=True),
        Output("train-predict-button", "disabled", allow_duplicate=True),
    ],
    Input("train-predict-button", "n_clicks"),
    prevent_initial_call=True,
)
def button_loading_state(n_clicks):
    return [dbc.Spinner(size="sm"), " Loading..."], True


@app.callback(
    [
        Output("counts_per_day", "figure"),
        Output("data-store", "data"),
        Output("date-range-clustering", "initial_visible_month"),
        Output("date-range-clustering", "min_date_allowed"),
        Output("date-range-clustering", "max_date_allowed"),
        Output("btn-load-data", "children"),
        Output("btn-load-data", "disabled"),
    ],
    [
        Input("btn-load-data", "n_clicks"),
        State("dd-data-source", "value"),
        State("i-hours-offset", "value"),
    ],
    prevent_initial_call=True,
)
def run_eda(_, source, offset):
    add_log_message(f"Loading the dataset...")

    df, fig = DL.load_data(
        source,
        begin_date,
        end_date,
        fraction,
        int(offset),
        outputs_folder_name=outputs_folder_name,
        verbose=True,
        perform_eda=True,
    )

    add_log_message(f"Loaded the data with size: {len(df)}")

    max_date_allowed = (df.timestamp.max()).date()
    min_date_allowed = (df.timestamp.min()).date()

    return (
        fig,
        df.to_dict("records"),
        min_date_allowed,
        min_date_allowed,
        max_date_allowed,
        "Load Data",
        False,
    )


@app.callback(
    [
        Output("scatter_mapbox_graph", "figure"),
        Output("data-store2", "data"),
        Output("btn-clustering", "children"),
        Output("btn-clustering", "disabled"),
    ],
    [
        Input("btn-clustering", "n_clicks"),
        Input("data-store", "data"),
        State("min_samples", "value"),
        State("eps", "value"),
        State("min_unique_days", "value"),
        State("dd-scale", "value"),
        State("date-range-clustering", "start_date"),
        State("date-range-clustering", "end_date"),
    ],
    prevent_initial_call=True,
)
def run_pipeline(_, df, min_samples, eps, min_unique_days, scale, start_date, end_date):
    # Do not execute this function if data-store has changed, but only when btn-clustering has been clicked.
    if not dash.callback_context.triggered[0]["prop_id"] == "btn-clustering.n_clicks":
        raise PreventUpdate

    add_log_message(f"Starting the clustering with scale: {scale}")

    df = pd.DataFrame(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # Step 1. Filter the dataset based on start and end time from date-range-clustering
    start = pd.to_datetime(f"{start_date} 00:00:00")
    end = pd.to_datetime(f"{end_date} 23:50:00")
    df = df.loc[df["timestamp"].between(start, end)]

    # Step 2. Run clustering. Returns df and a fig with a scattermapbox.
    df, fig = run_clustering(
        df,
        int(min_samples),
        float(eps),
        int(min_unique_days),
        outputs_folder_name,
        add_log_message,
        scale,
    )

    # Step 3. Transform data
    add_log_message(f"Transforming and resampling the dataset")
    df = DT.transform_start_end_times(df, outputs_folder_name, fill_gaps=True)

    # Step 4. Resample dataset. This saves the data at output/outputs_folder_name/resampled_df_10_min.xlsx.
    # TODO: currently, resample_df adds data points after the last data point to fill the day; avoid this behavior.
    df = DT.resample_df(df, outputs_folder_name)
    add_log_message(f"Done, saving datasets at output/{outputs_folder_name}")

    return fig, df.to_dict("records"), "Run Clustering", False


@app.callback(
    [
        Output("predictability_graph", "figure"),
        Output("btn-predictability-graph", "children"),
        Output("btn-predictability-graph", "disabled"),
    ],
    [
        Input("btn-predictability-graph", "n_clicks"),
        Input("data-store2", "data"),
        State("dd-features", "value"),
    ],
    prevent_initial_call=True,
)
def show_predictability(_, data, features):
    if not dash.callback_context.triggered[0]["prop_id"] in [
        "btn-predictability-graph.n_clicks"
    ]:
        raise PreventUpdate

    add_log_message("Making predictability graph...")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    df = DT.add_temporal_features(df, features)

    # Create instance of the DataPredictability class, which will return a fig that contains the predictability graph.
    fig = DataPredictability(
        df, features, outputs_folder_name, predictability_graph_rolling_window_size
    ).run()
    add_log_message("Done with predictability graph")

    return fig, "Load Graph", False


@app.callback(
    [
        Output("location_history_heatmap", "figure", allow_duplicate=True),
        Output("heatmap-picker-range", "start_date"),
        Output("heatmap-picker-range", "end_date"),
        Output("heatmap-picker-range", "initial_visible_month", allow_duplicate=True),
        Output("heatmap-picker-range", "min_date_allowed"),
        Output("heatmap-picker-range", "max_date_allowed"),
        Output("heatmap-picker-range-prediction", "min_date_allowed"),
        Output("heatmap-picker-range-prediction", "max_date_allowed"),
    ],
    [
        Input("scatter_mapbox_graph", "figure"),
        Input("data-store2", "data"),
    ],
    prevent_initial_call=True,
)
def show_location_history_heatmap(_, data):
    if not dash.callback_context.triggered[0]["prop_id"] in [
        "scatter_mapbox_graph.figure"
    ]:
        raise PreventUpdate

    add_log_message("Making location history heatmap...")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # Last day of dataset
    end_day = df.timestamp.max()
    start_day = end_day - pd.Timedelta(days=6)

    fig = HeatmapVisualizerV2(
        start_day.strftime("%Y-%m-%d"),
        end_day.strftime("%Y-%m-%d"),
        df,
        outputs_folder_name=outputs_folder_name,
    ).get_fig()

    add_log_message("Done with location history heatmap")

    # Set the min_date_allowed and max_date_allowed (based on the ranges of our dataset)

    return (
        fig,
        start_day.date(),
        end_day.date(),
        start_day.date(),
        df.timestamp.min().date(),
        df.timestamp.max().date(),
        df.timestamp.min().date(),
        df.timestamp.max().date(),
    )


@callback(
    [
        Output("location_history_heatmap", "figure"),
        Output("heatmap-picker-range", "initial_visible_month"),
    ],
    [
        Input("heatmap-picker-range", "start_date"),
        Input("heatmap-picker-range", "end_date"),
        Input("data-store2", "data"),
    ],
    prevent_initial_call=True,
)
def update_location_history_heatmap(start_date, end_date, data):
    add_log_message("Updating location history heatmap...")
    # Do not execute this function if data-store2 has changed, but only when heatmap-picker-range has changed (either start date or end date).
    if not dash.callback_context.triggered[0]["prop_id"] in [
        "heatmap-picker-range.start_date",
        "heatmap-picker-range.end_date",
    ]:
        raise PreventUpdate

    # TODO: check if data is available (i.e., clustering has been performed)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    fig = HeatmapVisualizerV2(
        start_date, end_date, df, outputs_folder_name=outputs_folder_name
    ).get_fig()

    add_log_message("Done with location history heatmap")
    return fig, datetime.strptime(start_date, "%Y-%m-%d").date()


@app.callback(
    # Inputs:
    [
        Output("prediction_heatmap", "figure"),
        Output("data-store3", "data"),
        Output("train-predict-button", "children"),
        Output("train-predict-button", "disabled"),
    ],
    [
        Input("train-predict-button", "n_clicks"),
        Input("heatmap-picker-range-prediction", "start_date"),
        Input("heatmap-picker-range-prediction", "end_date"),
        Input("data-store2", "data"),
        State("horizon-length", "value"),
    ],
    prevent_initial_call=True,
)
def train_model(_, start_date, end_date, data, horizon_length):
    # Even though we have multiple inputs, this function should only fire when the button 'train-predict-button' is pressed
    if not dash.callback_context.triggered[0]["prop_id"] in [
        "train-predict-button.n_clicks"
    ]:
        raise PreventUpdate

    # Load local file if data-store2 doesn't contain any data, or else take data from data-store2 and build df.
    if data is None:
        add_log_message("Loading training data from local file")
        df = pd.read_excel(
            f"output/{outputs_folder_name}/resampled_df_10_min.xlsx", index_col=[0]
        )
    else:
        df = pd.DataFrame(data)
        add_log_message("Loading training data from browser storage")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # Delete rows where location is "Unknown" since we don't want to predict the value "Unknown"
    df = df[df.location != "Unknown"]

    # Add temporal features and encode the location labels to integers
    df = DT.add_temporal_features(df, ["window_block", "hour", "weekday"])
    label_encoder = le()
    df.location = label_encoder.fit_transform(df.location)

    # Make training datasets
    X_train, y_train = make_train_data(start_date, end_date, df)

    # Train model and make predictions
    df_predictions, df_probabilities = train_and_predict(
        add_log_message, X_train, y_train, horizon_length, label_encoder
    )

    # Build heatmap
    add_log_message("Making heatmap with predictions")
    fig = HeatmapVisualizerV2(
        df_predictions.timestamp.min().date().strftime("%Y-%m-%d"),
        df_predictions.timestamp.max().date().strftime("%Y-%m-%d"),
        df_predictions[["timestamp", "location"]],
        outputs_folder_name=outputs_folder_name,
    ).get_fig()

    return fig, df_probabilities.to_dict("records"), "Train and Predict", False


@app.callback(
    [
        Output("probabilities-table", "figure"),
        Output("modal-explanation-probabilities", "is_open"),
    ],
    [Input("prediction_heatmap", "clickData"), Input("data-store3", "data")],
    prevent_initial_call=True,
)
def show_probabilities(clickData, df_probabilities):
    # Only fire when user clicks in the heatmap
    if not dash.callback_context.triggered[0]["prop_id"] in [
        "prediction_heatmap.clickData"
    ]:
        raise PreventUpdate

    df_probabilities = pd.DataFrame(df_probabilities)
    df_probabilities["timestamp"] = pd.to_datetime(df_probabilities["timestamp"])

    x_data = clickData["points"][0]["x"]
    y_data = clickData["points"][0]["y"]

    # With x_data and y_data we can select the right row in df_probabilities
    data_row = df_probabilities[
        df_probabilities.timestamp
        == pd.to_datetime(x_data + " " + y_data.split(" ")[0])
    ].drop("timestamp", axis=1)
    df_row = data_row.melt(var_name="location", value_name="value").sort_values(
        by="value", ascending=False
    )
    df_row["value"] = (df_row["value"] * 100).round(2)

    table = go.Table(
        header=dict(values=["Location", "Probability (0-100%)"]),
        cells=dict(values=[df_row["location"], df_row["value"]]),
    )

    fig = go.Figure(data=[table])

    return fig, True


@callback(
    [
        Output("min_samples", "value"),
        Output("eps", "value"),
    ],
    Input("dd-scale", "value"),
)
def update_output(scale):
    if scale == "street":
        return 200, 0.02
    elif scale == "city":
        return 1000, 5
    elif scale == "country":
        return 1000, 100


@app.callback(
    Output("log-display", "children"), Input("interval-component", "n_intervals")
)
def update_log_display(_):
    log_texts = "\n".join(log_messages)
    return log_texts


# Function to add a new log message
def add_log_message(message):
    log_messages.append(get_time() + message)


@callback(
    Output("modal-explanation-probabilities", "is_open", allow_duplicate=True),
    Input("closeButton", "n_clicks"),
    prevent_initial_call=True,
)
def close_modal(_):
    return False


@callback(
    Output("modal-load-data", "is_open", allow_duplicate=True),
    Input("closeButton", "n_clicks"),
    prevent_initial_call=True,
)
def close_modal(_):
    return False


@callback(
    Output("modal-clustering", "is_open", allow_duplicate=True),
    Input("closeButton", "n_clicks"),
    prevent_initial_call=True,
)
def close_modal(_):
    return False


@callback(
    Output("modal-predictability-graph", "is_open", allow_duplicate=True),
    Input("closeButton", "n_clicks"),
    prevent_initial_call=True,
)
def close_modal(_):
    return False


@callback(
    Output("modal-location-history", "is_open", allow_duplicate=True),
    Input("closeButton", "n_clicks"),
    prevent_initial_call=True,
)
def close_modal(_):
    return False


@callback(
    Output("modal-predicting", "is_open", allow_duplicate=True),
    Input("closeButton", "n_clicks"),
    prevent_initial_call=True,
)
def close_modal(_):
    return False


@app.callback(
    Output("modal-predictability-graph", "is_open", allow_duplicate=True),
    Input("btn-predictability-graph-explanation", "n_clicks"),
    prevent_initial_call=True,
)
def show_predictability_graph_explanation(n_clicks):
    return True


@app.callback(
    Output("modal-load-data", "is_open"),
    Input("btn-load-data-explanation", "n_clicks"),
    prevent_initial_call=True,
)
def show_predictability_graph_explanation(n_clicks):
    return True


@app.callback(
    Output("modal-clustering", "is_open"),
    Input("btn-clustering-explanation", "n_clicks"),
    prevent_initial_call=True,
)
def show_predictability_graph_explanation(n_clicks):
    return True


@app.callback(
    Output("modal-location-history", "is_open"),
    Input("btn-location-history-explanation", "n_clicks"),
    prevent_initial_call=True,
)
def show_predictability_graph_explanation(n_clicks):
    return True


@app.callback(
    Output("modal-predicting", "is_open"),
    Input("btn-predicting-explanation", "n_clicks"),
    prevent_initial_call=True,
)
def show_predictability_graph_explanation(n_clicks):
    return True


if __name__ == "__main__":
    app.run(debug=True)
