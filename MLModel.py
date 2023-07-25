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
    def __init__(self, df: pd.DataFrame, model_date_start: str, model_date_end: str, num_last_days_for_testing: int) -> None:
        self.df = df
        self.model_date_start = model_date_start
        self.model_date_end = model_date_end
        self.num_last_days_for_testing = num_last_days_for_testing

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

        # Step 6. Visualize the predictions in a heatmap.
        self.visualize_predictions()

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
        self.df_original = self.df.copy()

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

    def make_train_test_split(self) -> None:
        # Define the end of training and the beginning of testing
        self.train_end_date = self.model_date_end - pd.Timedelta(days=self.num_last_days_for_testing)
        self.test_start_date = self.model_date_end - pd.Timedelta(days=self.num_last_days_for_testing) + pd.Timedelta(minutes=10)

        # Create masks to filter the data based on dates
        train_mask = self.df['time'].between(self.model_date_start, self.train_end_date)
        test_mask = self.df['time'].between(self.test_start_date, self.model_date_end)

        # Split the data into train and test sets
        self.X_train = self.df.loc[train_mask, ["weekday", "hour", "day"]]
        self.y_train = self.df.loc[train_mask, "location"]
        self.X_test = self.df.loc[test_mask, ["weekday", "hour", "day"]]
        self.y_test = self.df.loc[test_mask, "location"]

    def run_model(self) -> None:
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

    def evaluate_model(self) -> None:
        self.model_accuracy = accuracy_score(self.y_test, self.predictions)
        print("Accuracy: %.2f%%" % (self.model_accuracy * 100.0))

        self.class_report = classification_report(self.y_test, self.predictions)
        print(f"Classification report: \n{self.class_report}")

    def visualize_predictions(self) -> None:
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
            verbose=True,
            name="heatmap_predicted",
        )

        # And also visualize the actual values in a heatmap named heatmap_actual.png
        HeatmapVisualizer(
            str(self.test_start_date.date()),
            str(self.model_date_end.date()),
            self.df_original,
            verbose=True,
            name="heatmap_actual",
        )


p = Predict(
    df=None, # Choose df = None if you want to load the dataframe from resampled_df_10_min.xlsx.
    model_date_start=pd.to_datetime("2023-06-21 00:00:00"),  #  + pd.Timedelta(hours=5)
    model_date_end=pd.to_datetime("2023-07-25 23:50:00"),
    num_last_days_for_testing = 2
)
