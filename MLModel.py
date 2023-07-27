from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from Visualisations import HeatmapVisualizer
import sys
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class Predict:
    def __init__(self, df: pd.DataFrame, model_date_start: str, model_date_end: str, num_last_days_for_testing: int, heatmaps:bool = True) -> None:
        self.df = df
        self.model_date_start = model_date_start
        self.model_date_end = model_date_end
        self.num_last_days_for_testing = num_last_days_for_testing
        self.heatmaps = heatmaps

        self.main()

    def main(self):
        # Step 1. Load dataset.
        self.load_data()

        # Step 2. Make temporal features.
        self.make_temporal_features()

        # Step 3. Make train/test split.
        self.make_train_test_split()

        # Step 4. Run XGBoost model and make the predictions.
        self.run_model()

        # Step 5. Evaluate model performance.
        self.evaluate_model()

        # Step 6. Visualize the predictions, the actual values, and the training values in heatmaps.
        if self.heatmaps:
            self.visualize()

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
        self.df["weekday"] = self.df["time"].dt.dayofweek
        self.df["hour"] = self.df["time"].dt.hour
        self.df["day"] = self.df["time"].dt.day

        print(self.df.head(30))

    def make_train_test_split(self) -> None:
        # Define the end of training and the beginning of testing. For clarity, we define all four variables.

        n_training_days = 30
        n_testing_days = 7

        # Starting at model_date_start, we take n_training_days of data.
        # TODO: calculate number of loops we can do
        self.train_start_date = self.model_date_start + pd.Timedelta(days=0) # Days should be loop parameter.
        self.train_end_date = self.train_start_date + pd.Timedelta(days=n_training_days-1, hours=23, minutes=50) # We want to end on the last day at 23:50.
        self.test_start_date = self.train_end_date + pd.Timedelta(minutes=10) # Start = 10 minutes after training set ends, i.e., begin is at 00:00. 
        self.test_end_date = self.test_start_date + pd.Timedelta(days=n_testing_days)

        # Create masks to filter the data based on dates
        train_mask = self.df['time'].between(self.train_start_date, self.train_end_date)
        test_mask = self.df['time'].between(self.test_start_date, self.test_end_date)

        # Split the data into train and test sets
        self.X_train = self.df.loc[train_mask, ["weekday", "hour", "day"]]
        self.y_train = self.df.loc[train_mask, "location"]
        self.X_test = self.df.loc[test_mask, ["weekday", "hour", "day"]]
        self.y_test = self.df.loc[test_mask, "location"]

    def run_model(self) -> None:
        self.model = RandomForestClassifier()

        print(f"Training model with {len(self.X_train)} data points from {self.model_date_start} until {self.train_end_date}.")

        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

        print(f"Predicting {len(self.X_test)} data points from {self.test_start_date} until {self.model_date_end}.")

    def evaluate_model(self) -> None:
        self.model_accuracy = accuracy_score(self.y_test, self.predictions)
        print("Accuracy: %.2f%%" % (self.model_accuracy * 100.0))

        self.class_report = classification_report(self.y_test, self.predictions)
        print(f"Classification report: \n{self.class_report}")

    def visualize(self) -> None:
        # Create a datetime index with 10-minute intervals.
        time_intervals = pd.date_range(
            start=self.test_start_date, end=self.model_date_end, freq="10T"
        )

        # Create a DataFrame with the 'time' column and the 'location' column that holds the predicted locations (strings).
        df_predictions = pd.DataFrame(
            {
                "time": time_intervals,
                "location": self.le.inverse_transform(self.predictions),
            }
        )

        # Visualize the predictions in a heatmap and save it as heatmap_predicted.png.
        HeatmapVisualizer(
            str(self.test_start_date.date()),
            str(self.model_date_end.date()),
            df_predictions,
            name="heatmap_predicted",
        )

        # And also visualize the actual values in a heatmap named heatmap_actual.png
        HeatmapVisualizer(
            str(self.test_start_date.date()),
            str(self.model_date_end.date()),
            self.df_original, # Now we use the original dataframe (with time and location, 10 min intervals) to visualize the actual data.
            name="heatmap_actual",
        )

        # And lastly, visualize the training data as well as heatmap_training.png.
        HeatmapVisualizer(
            str(self.model_date_start.date()),
            str(self.train_end_date.date()),
            self.df_original, # Now we use the original dataframe (with time and location, 10 min intervals) to visualize the actual data.
            name="heatmap_training",
        )


p = Predict(
    df=None, # Choose df = None if you want to load the dataframe from resampled_df_10_min.xlsx.
    model_date_start=pd.to_datetime("2022-05-25 00:00:00"),
    model_date_end=pd.to_datetime("2022-07-25 23:50:00"),
    num_last_days_for_testing = 7,
    heatmaps=True
)
