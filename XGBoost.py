from xgboost import XGBClassifier
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from Visualisations import HeatmapVisualizer
import sys


class Predict:
    def __init__(self, df, model_date_start, model_date_end):
        self.df = df
        self.model_date_start = model_date_start
        self.model_date_end = model_date_end

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

    def load_data(self):
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

        self.le = preprocessing.LabelEncoder()
        self.df.location = self.le.fit_transform(self.df.location)

    def validate_data(self):
        # Check if the loaded df satisfies all criteria.
        if not "time" in self.df or not "location" in self.df:
            raise ValueError(
                "Make sure that df contains both the columns 'time' (datetime) and 'location' (strings of locations)!"
            )

    def filter_data(self):
        self.df = self.df[
            (self.df["time"] >= self.model_date_start)
            & (self.df["time"] <= self.model_date_end)
        ]

    def make_temporal_features(self):
        self.df["weekday"] = self.df["time"].dt.dayofweek
        self.df["hour"] = self.df["time"].dt.hour
        self.df["day"] = self.df["time"].dt.day

    def make_train_test_split(self):
        # Define train and test date ranges
        self.train_start_date = self.model_date_start
        self.train_end_date = "2023-04-30 23:50:00"
        self.test_start_date = "2023-05-01 00:00:00"
        self.test_end_date = self.model_date_end

        # Create masks to filter the data based on dates
        train_mask = (self.df["time"] >= self.train_start_date) & (
            self.df["time"] <= self.train_end_date
        )
        test_mask = (self.df["time"] >= self.test_start_date) & (
            self.df["time"] <= self.test_end_date
        )

        # Split the data into train and test sets
        self.X_train = self.df.loc[train_mask, ["weekday", "hour", "day"]]
        self.y_train = self.df.loc[train_mask, "location"]
        self.X_test = self.df.loc[test_mask, ["weekday", "hour", "day"]]
        self.y_test = self.df.loc[test_mask, "location"]

    def run_model(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

    def evaluate_model(self):
        self.model_accuracy = accuracy_score(self.y_test, self.predictions)
        print("Accuracy: %.2f%%" % (self.model_accuracy * 100.0))

        self.class_report = classification_report(self.y_test, self.predictions)
        print(f"Classification report: \n{self.class_report}")

    def visualize_predictions(self):
        # Create a datetime index with 10-minute intervals.
        time_intervals = pd.date_range(
            start=self.test_start_date, end=self.test_end_date, freq="10T"
        )

        # Create a DataFrame with the 'time' column and the 'location' column that holds the predicted locations (strings).
        df = pd.DataFrame(
            {
                "time": time_intervals,
                "location": self.le.inverse_transform(self.predictions),
            }
        )

        # Visualize the predictions in a heatmap and save it as heatmap_predicted.png.
        HeatmapVisualizer(
            self.test_start_date.split(" ")[0],
            self.test_end_date.split(" ")[0],
            df,
            verbose=True,
            name="heatmap_predicted",
        )


p = Predict(
    df=None,
    model_date_start="2023-04-01 00:00:00",
    model_date_end="2023-05-07 23:50:00",
)
