from Cluster import Cluster
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import DataTransformer as DT


def get_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S") + ": "

def run_clustering(df:pd.DataFrame, min_samples:int, eps:float, min_unique_days:int, outputs_folder_name:str, add_log_message):
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
        )
        .add_locations_to_original_dataframe(
            export_xlsx=True,  # Export the dataframe to excel file? Useful for analyzing.
            name="test",
        )
        .plot_clusters(
            filter_noise=False,  # Remove the -1 labels (i.e., noise) before plotting the clusters
        )
        .df  # These functions return 'self' so we can chain them and easily access the df attribute (for input to further modeling/visualization).
    )

    add_log_message(f"Done with clustering")

    return df, c.fig

def train_and_predict(add_log_message, X_train, y_train, horizon_length, label_encoder):
    add_log_message("Training ML model")
    model = RandomForestClassifier()
    model.fit(X_train, y_train) # , sample_weight=np.linspace(0, 1, len(X_train))

    # Make X_test, starting one day after the last day in the dataset
    current_time = datetime.now().replace(microsecond=0)
    X_test_start = current_time - pd.Timedelta(minutes=current_time.minute % 10, seconds=current_time.second) # Round the current time to the nearest 10-minute interval
    X_test_end = X_test_start + pd.Timedelta(days=int(horizon_length)-1, hours=23, minutes=50)

    # Create a DataFrame with the 'time' column and the 'location' column that holds the predicted locations (strings).
    df_predictions = pd.DataFrame({"timestamp": pd.date_range(start=X_test_start, end=X_test_end, freq="10T")})
    df_predictions = DT.add_temporal_features(df_predictions)

    # Make predictions and inverse transform them
    df_predictions['location'] = model.predict(df_predictions[["weekday", "hour", "window_block"]])
    df_predictions['location'] = label_encoder.inverse_transform(df_predictions['location'])

    # Save probabilities for each 10-min block 
    probas = model.predict_proba(df_predictions[["weekday", "hour", "window_block"]])
    df_probabilities = pd.DataFrame(probas, columns=label_encoder.inverse_transform(model.classes_))
    df_probabilities = pd.concat([df_predictions[["timestamp"]], df_probabilities], axis=1)

    return df_predictions, df_probabilities

def make_train_data(start_date, end_date, df):
    # Make train_start, train_end, predict_start, predict_end date(time) objects
    train_start = pd.to_datetime(f"{start_date} 00:00:00")
    train_end = pd.to_datetime(f"{end_date} 23:50:00")

    train_mask = df["timestamp"].between(train_start, train_end)
    X_train = df.loc[train_mask, ["weekday", "hour", "window_block"]]
    y_train = df.loc[train_mask, "location"]

    return X_train, y_train