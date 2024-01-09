from Cluster import Cluster
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import DataTransformer as DT


def get_time():
    """
    Retrieves current time in specific format. Used for the log messages in the Dash app. 

    Parameters
    ----------

    Returns
    -------
    str
        Current time as string in fixed format.

    """
    now = datetime.now()
    return now.strftime("%H:%M:%S") + ": "

def run_clustering(df:pd.DataFrame, min_samples:int, eps:float, min_unique_days:int, outputs_folder_name:str, add_log_message, scale:str):
    """
    Collection function for the "run clustering" button in the Dash app. 

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe for the clustering with columns: timestamp, latitude, longigude, id
    min_samples : int
        min_samples value for the clustering algorithm (see DBSCAN docs)
    eps : float
        eps value for the clustering algorithm (see DBSCAN docs)
    min_unique_days : int
        minimum number of unique days required for each cluster to be a valid cluster
    outputs_folder_name : str
        name of the folder in which the output files are placed
    add_log_message : func
        instance of add_log_message function so we can use it within this function
    scale : str
        scale parameter takes in either "street", "city", or "country". This value changes the eps and min_samples parameters. 

    Returns
    -------
    df : pd.DataFrame
        output dataframe with columns: timestamp, latitude, longitude, id, moving, cluster, and location
    c.fig : fig
        the Plotly scattermapbox that shows the different clusters and their labels
    """

    c = Cluster(
        df,  # Input dataset (with latitude, longitude, timestamp columns)
        outputs_folder_name=outputs_folder_name, 
        verbose=True,  # Do we want to see print statements?
        pre_filter=True,  # Apply filters to the data before the clustering (such as removing moving points)
        post_filter=True,  # Apply filters to the data/clusters after the clustering (such as deleting homogeneous clusters)
        filter_moving=True,  # Do we want to delete the data points where the subject was moving?
        centroid_k=10,  # Number of nearest neighbors to consider for density calculation (for cluster centroids)
        min_unique_days=min_unique_days,  # If post_filter = True, then delete all clusters that have been visited on less than min_unique_days days.
        scale=scale  # The scale of clustering (choose between street, city, or country)
    )

    # Then we run the clustering and visualisation
    df = (
        c.run_clustering(
            min_samples=min_samples,  # The number of samples in a neighborhood for a point to be considered as a core point
            eps=eps,  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. 0.01 = 10m
        )
        .add_locations_to_original_dataframe(
            export_xlsx=False,  # Export the dataframe to excel file? Useful for analyzing.
            name="resulting_clusters",
        )
        .plot_clusters(
            filter_noise=False,  # Remove the -1 labels (i.e., noise) before plotting the clusters
        )
        .df  # These functions return 'self' so we can chain them and easily access the df attribute (for input to further modeling/visualization).
    )

    add_log_message(f"Done with clustering")

    return df, c.fig

def train_and_predict(add_log_message, X_train:pd.DataFrame, y_train:pd.DataFrame, horizon_length:int, label_encoder):
    """
    Trains the ML model and makes the predictions. 

    Parameters
    ----------
    add_log_message : func
        instance of add_log_message function so we can use it within this function
    X_train : pd.DataFrame
        dataframe with the temporal featuers that is used to train the model
    y_train : pd.DataFrame
        dataframe with the target variable that is used to train the model
    horizon_length : int
        number of days in the future to predict
    label_encoder 
        instance of the label encoder that was created earlier. This label_encoder can be used to retrieve the clusters' labels 

    Returns
    -------
    df_predictions : pd.DataFrame
        dataframe with the input data (temporal features for each 10-minute window) and an additional value for the predictions
    df_probabilities : pd.DataFrame
        dataframe with the probabilities for each location, for each 10-minute window in the future 

    """

    add_log_message("Training ML model")
    model = RandomForestClassifier()
    model.fit(X_train, y_train) # , sample_weight=np.linspace(0, 1, len(X_train))

    # Make X_test, starting one day after the last day in the dataset
    current_time = datetime.now().replace(microsecond=0)
    X_test_start = current_time - pd.Timedelta(minutes=current_time.minute % 10, seconds=current_time.second) # Round the current time to the nearest 10-minute interval
    X_test_end = X_test_start + pd.Timedelta(days=int(horizon_length)-1, hours=23, minutes=50)

    # Create a DataFrame with the 'time' column and the 'location' column that holds the predicted locations (strings).
    df_predictions = pd.DataFrame({"timestamp": pd.date_range(start=X_test_start, end=X_test_end, freq="10T")})
    df_predictions = DT.add_temporal_features(df_predictions, ["weekday", "hour", "window_block"])

    # Make predictions and inverse transform them
    df_predictions['location'] = model.predict(df_predictions[["weekday", "hour", "window_block"]])
    df_predictions['location'] = label_encoder.inverse_transform(df_predictions['location'])

    # Save probabilities for each 10-min block 
    probas = model.predict_proba(df_predictions[["weekday", "hour", "window_block"]])
    df_probabilities = pd.DataFrame(probas, columns=label_encoder.inverse_transform(model.classes_))
    df_probabilities = pd.concat([df_predictions[["timestamp"]], df_probabilities], axis=1)

    return df_predictions, df_probabilities

def make_train_data(start_date, end_date, df):
    """
    Function to make train data set

    Parameters
    ----------
    start_data : str
        taken from the date selector (heatmap-picker-range-prediction:start_date)
    end_date : str
        taken from the date selector (heatmap-picker-range-prediction:end_date)

    Returns
    -------
    X_train : pd.DataFrame
        dataframe that is used for training a ML model, containing the columns of the temporal features
    y_train : pd.Series
        series containing the target values (locations) that is used for training a ML model
    """

    # Make train_start, train_end, predict_start, predict_end date(time) objects
    train_start = pd.to_datetime(f"{start_date} 00:00:00")
    train_end = pd.to_datetime(f"{end_date} 23:50:00")

    train_mask = df["timestamp"].between(train_start, train_end)
    X_train = df.loc[train_mask, ["weekday", "hour", "window_block"]]
    y_train = df.loc[train_mask, "location"]

    return X_train, y_train