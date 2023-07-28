from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from Visualisations import HeatmapVisualizer
import sys
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import json
import numpy as np
from datetime import datetime


class Predict:
    def __init__(self, df: pd.DataFrame, model_date_start: str, model_date_end: str, n_training_days:int, n_testing_days:int, model_features:list, heatmaps:bool = True) -> None:
        self.df = df
        self.model_date_start = model_date_start
        self.model_date_end = model_date_end
        self.heatmaps = heatmaps
        self.model_features = model_features

        self.n_training_days = n_training_days
        self.n_testing_days = n_testing_days
        self.performance = {}
        self.n_validation_loops = ((self.model_date_end - self.model_date_start).days + 1) - (self.n_training_days + self.n_testing_days)

        self.main()

    def main(self):
        # Step 1. Load dataset.
        self.load_data()

        # Step 2. Make temporal features.
        self.make_temporal_features()

        # Here we enter the train/validation loop.
        for i in range(self.n_validation_loops):
            self.i = i

            # Step 3. Make train/test split. 
            self.make_train_test_split(i)

            # Step 4. Run XGBoost model and make the predictions.
            self.run_model()

            # Step 5. Evaluate model performance and store results in self.performance dict.
            self.evaluate_model()

        # Step 6. Visualize the predictions, the actual values, and the training values in heatmaps.
        if self.heatmaps:
            self.visualize()

        print (json.dumps(self.performance, indent=2, default=str))

    def load_data(self) -> pd.DataFrame:
        # If df is None, it is not set, hence we have to load it from xlsx.
        if self.df == None:
            try:
                self.df = pd.read_excel(
                    "output/resampled_df_10_min.xlsx", index_col=[0]
                )
            except FileNotFoundError as e:
                print(
                    f"{e}: Make sure to put your resampled_df_10_min.xlsx file in the 'output' folder."
                )

                sys.exit(1)

        self.validate_data()
        self.filter_data()
        self.df_original = self.df.copy() # Make a copy of the original data so that we can compare the predictions with the original data (via heatmaps).

        # We need to transform our string representations of locations to integers, for the ML models to work. 
        self.le = preprocessing.LabelEncoder()
        self.df.location = self.le.fit_transform(self.df.location)

        return self.df

    def validate_data(self) -> None:
        # Check if the loaded df satisfies all criteria.
        if not "time" in self.df or not "location" in self.df:
            raise ValueError(
                "Make sure that df contains both the columns 'time' (datetime) and 'location' (strings of locations)!"
            )

    def filter_data(self) -> None:
        self.df[self.df['time'].between(self.model_date_start, self.model_date_end)]

        print(f"Message (ML filter): after filtering we have {len(self.df)} records, starting at {str(self.df.iloc[0].time)} and ending at {str(self.df.iloc[-1].time)}.")

    def make_temporal_features(self) -> None:
        if "weekday" in self.model_features:
            self.df["weekday"] = self.df["time"].dt.dayofweek

        if "hour" in self.model_features:
            self.df["hour"] = self.df["time"].dt.hour

        if "day" in self.model_features:
            self.df["day"] = self.df["time"].dt.day

        print(self.df.head())

    def make_train_test_split(self, i: int) -> None:
        """ This function calculates the train/test begin and end dates, based on the parameter i (from the validation loop) and the number of training and testing days. 

        Parameters:
            i (int): parameter from the validation loop. This value is between [0, self.n_validation_loops]. 

        Returns:
            self

        """

        # Define the end of training and the beginning of testing. 
        self.train_start_date = self.model_date_start + pd.Timedelta(days=i) # Days is the loop parameter, which goes from [0, self.n_validation_loops].
        self.train_end_date = self.train_start_date + pd.Timedelta(days=self.n_training_days-1, hours=23, minutes=50) # We want to end on the last day at 23:50.
        self.test_start_date = self.train_end_date + pd.Timedelta(minutes=10) # Start = 10 minutes after training set ends, i.e., begin is at 00:00. 
        self.test_end_date = self.test_start_date + pd.Timedelta(days=self.n_testing_days-1, hours=23, minutes=50)

        # Create masks to filter the data based on dates
        train_mask = self.df['time'].between(self.train_start_date, self.train_end_date)
        test_mask = self.df['time'].between(self.test_start_date, self.test_end_date)

        # Split the data into train and test sets
        self.X_train = self.df.loc[train_mask, self.model_features]
        self.y_train = self.df.loc[train_mask, "location"]
        self.X_test = self.df.loc[test_mask, self.model_features]
        self.y_test = self.df.loc[test_mask, "location"]

    def run_model(self) -> None:
        self.model = RandomForestClassifier()

        print(f"Training model with {len(self.X_train)} data points from {self.train_start_date} until {self.train_end_date}.")

        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

        print(f"Predicting {len(self.X_test)} data points from {self.test_start_date} until {self.test_end_date}.")

    def evaluate_model(self) -> None:
        # Add meta data and evaluation metrics to self.performance.
        # First, for each day in self.n_testing_days, we calculate the performance and save it in a dict.
        performance_metrics_per_day = {}
        for d in range(self.n_testing_days):
            this_day_predictions = self.predictions[d*144:(d+1)*144]
            this_day_actual_values = self.y_test[d*144:(d+1)*144]
            acc = accuracy_score(this_day_actual_values, this_day_predictions)

            # And add them to a dictionary where the key is the day, increasing from 0 to self.n_testing_days.
            performance_metrics_per_day[d] = {
                "acc":acc,
            }

        self.performance[self.i] = {
            "meta":{
                "train_start_date":self.train_start_date,
                "train_end_date":self.train_end_date,
                "test_start_date":self.test_start_date,
                "test_end_date":self.test_end_date,
            }, # We add performance metrics per day.
            "performance_metrics_per_day":performance_metrics_per_day,
            "predictions":self.predictions,
            "true_values":np.array(self.y_test.values.tolist())
        }

    def visualize(self, test_start_date:datetime, test_end_date:datetime, train_start_date:datetime, train_end_date:datetime, 
                  train_data: pd.DataFrame, test_data:pd.DataFrame, predictions:list) -> None:
        # Create a datetime index with 10-minute intervals.
        time_intervals = pd.date_range(
            start=test_start_date, end=test_end_date, freq="10T"
        )

        # Create a DataFrame with the 'time' column and the 'location' column that holds the predicted locations (strings).
        df_predictions = pd.DataFrame(
            {
                "time": time_intervals,
                "location": self.le.inverse_transform(predictions),
            }
        )

        # Visualize the predictions in a heatmap and save it as heatmap_predicted.png.
        HeatmapVisualizer(
            str(test_start_date.date()),
            str(test_end_date.date()),
            df_predictions,
            name="heatmap_predicted",
        )

        # And also visualize the actual values in a heatmap named heatmap_actual.png
        HeatmapVisualizer(
            str(test_start_date.date()),
            str(test_end_date.date()),
            test_data, # Now we use the original dataframe (with time and location, 10 min intervals) to visualize the actual data.
            name="heatmap_actual",
        )

        # And lastly, visualize the training data as well as heatmap_training.png.
        HeatmapVisualizer(
            str(train_start_date.date()),
            str(train_end_date.date()),
            train_data, # Now we use the original dataframe (with time and location, 10 min intervals) to visualize the actual data.
            name="heatmap_training",
        )


p = Predict(
    df=None, # Choose df = None if you want to load the dataframe from resampled_df_10_min.xlsx.
    model_date_start=pd.to_datetime("2023-06-15 00:00:00"),
    model_date_end=pd.to_datetime("2023-07-20 23:50:00"),
    n_training_days=21,
    n_testing_days=7,
    model_features=["weekday", "hour", "day"], # All options are: "weekday", "day", "hour"
    heatmaps=False,
)

# selected_p = p.performance[40]
# p.df.location = p.le.inverse_transform(p.df.location)

# p.visualize(
#     test_start_date=selected_p["meta"]["test_start_date"], 
#     test_end_date=selected_p["meta"]["test_end_date"], 
#     train_start_date=selected_p["meta"]["train_start_date"], 
#     train_end_date=selected_p["meta"]["train_end_date"], 
#     train_data=p.df[p.df['time'].between(selected_p["meta"]["train_start_date"], selected_p["meta"]["train_end_date"])], 
#     test_data=p.df[p.df['time'].between(selected_p["meta"]["test_start_date"], selected_p["meta"]["test_end_date"])],
#     predictions=selected_p["predictions"]
# )