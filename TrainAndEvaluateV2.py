from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from Visualisations import HeatmapVisualizer
import sys
import warnings
import sklearn.exceptions
from datetime import datetime
from collections import defaultdict
import pickle
from tqdm import tqdm
import math

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', message='Mean of empty slice')


class TrainAndEvaluate:
    def __init__(self, df: pd.DataFrame, start_date:datetime, end_date:datetime, model_features=list) -> None:
        """ This class works with the resampled_df_10_min.xlsx file (or, with its df). It will train a ML model multiple times, each time validating the results with a sliding window technique. Output is a heatmap 
        with performances for each combination of  set length and 
        training
        """
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        self.model_features = model_features

        # The self.performance dict contains, for each training size and each validation loop, the accuracy scores for all the days that were predicted (where number of days = (max_n_testing_days - min_n_testing_days))
        self.performance = defaultdict(dict)

    def main(self):
        # Step 1. Load dataset.
        self.make_dataset()

        # Step 2. Make temporal features.
        self.make_temporal_features()

        # Step 3. Make train/test split. We need the loop index to offset the days (for making the training/testing sets).
        n_windows = math.floor((self.end_date - self.start_date).days / 74) # 74 = 60 days training, 14 days predicting (i.e., max window size)
        offset_days = 1 + ((self.end_date - self.start_date).days % 74)
        print(f"n_windows: {n_windows}")

        for block_index in tqdm(range(n_windows), desc=" Block loop", position=1): # Loop 7 times    
            self.block_index = block_index        

            for train_index in tqdm(range(60), desc=" Training window size loop", position=0, leave=False):
                self.train_index = train_index

                self.make_train_test_split(block_index, train_index, offset_days)

                # Step 4. Run model and make the predictions.
                self.run_model()

                # Step 5. Evaluate model performance and store results in self.performance dict.
                self.evaluate_model()

        with open('output/model_performances.pkl', 'wb') as f:
            pickle.dump(self.performance, f)

        print("\nSaved model performance to output/model_performances.pkl")

        return self.performance

    def make_dataset(self) -> pd.DataFrame:
        # If df is None, it is not set, hence we have to load it from xlsx. Normally, when this class is used in the main pipeline, the 10-minute-interval dataset is passed on as self.df. 
        if not isinstance(self.df, pd.DataFrame):
            try:
                self.df = pd.read_excel(
                    "output/resampled_df_10_min.xlsx", index_col=[0]
                )
            except FileNotFoundError as e:
                print(
                    f"{e}: Make sure to put your resampled_df_10_min.xlsx file in the 'output' folder."
                )

                sys.exit(1)

        self.filter_data()
        self.df_original = (
            self.df.copy()
        )  # Make a copy of the original data so that we can compare the predictions with the original data (via heatmaps).

        # We need to transform our string representations of locations to integers, for the ML models to work.
        self.le = preprocessing.LabelEncoder()
        self.df.location = self.le.fit_transform(self.df.location)

        return self.df

    def filter_data(self) -> None:
        self.df = self.df[self.df["time"].between(self.start_date, self.end_date)]

        print(
            f"Message (ML filter): after filtering we have {len(self.df)} records (with 10-minute intervals), starting at {str(self.df.iloc[0].time)} and ending at {str(self.df.iloc[-1].time)}."
        )

    def make_temporal_features(self) -> None:
        if "weekday" in self.model_features:
            self.df["weekday"] = self.df["time"].dt.dayofweek

        if "hour" in self.model_features:
            self.df["hour"] = self.df["time"].dt.hour

        if "day" in self.model_features:
            self.df["day"] = self.df["time"].dt.day

    def make_train_test_split(self, block_index: int, train_index: int, offset_days:int) -> None:
        start_offset_days = offset_days + (block_index * 74)

        self.train_start_date = self.start_date + pd.Timedelta(days=start_offset_days)
        self.train_end_date = self.train_start_date + pd.Timedelta(days=train_index, hours=23, minutes=50)
        self.test_start_date = self.train_end_date + pd.Timedelta(minutes=10) 
        self.test_end_date = self.test_start_date + pd.Timedelta(days=13, hours=23, minutes=50)

        train_mask = self.df["time"].between(self.train_start_date, self.train_end_date)
        test_mask = self.df["time"].between(self.test_start_date, self.test_end_date)

        # Split the data into train and test sets
        self.X_train = self.df.loc[train_mask, self.model_features]
        self.y_train = self.df.loc[train_mask, "location"]
        self.X_test = self.df.loc[test_mask, self.model_features]
        self.y_test = self.df.loc[test_mask, "location"]

        # print(f"Block {self.block_index}, train size: {self.train_index}. Training: {self.train_start_date}-{self.train_end_date}, testing: {self.test_start_date}-{self.test_end_date}.")

    def run_model(self) -> None:
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)

        # Make predictions for 14 days into the future. 
        self.predictions = self.model.predict(self.X_test)

    def evaluate_model(self) -> None:
        for d in range(14):
            this_day_predictions = self.predictions[d*144:(d+1)*144]
            this_day_actual_values = self.y_test[d*144:(d+1)*144]
            acc = accuracy_score(this_day_actual_values, this_day_predictions)

            if f"days_into_future_{d}" not in self.performance[f"training_set_size_{self.train_index}"]:
                self.performance[f"training_set_size_{self.train_index}"][f"days_into_future_{d}"] = []

            self.performance[f"training_set_size_{self.train_index}"][f"days_into_future_{d}"].append(acc)
            # print(f"Added acc: {round(acc, 3)} to self.performance[{self.train_index}][{d}]")
    

scores = TrainAndEvaluate(
    df = None,
    start_date = pd.to_datetime("2021-03-01 00:00:00"),
    end_date = pd.to_datetime("2022-07-15 23:50:00"),
    model_features = ["day", "hour", "weekday"]
).main()

import matplotlib.pyplot as plt

# Extract data from the scores dictionary
training_sizes = []
performance_scores = []

for training_size, forecast_scores in scores.items():
    training_sizes.append(int(training_size.split("_")[-1]))
    performance_scores.append(sum(sum(score_list) / len(score_list) for score_list in forecast_scores.values()) / len(forecast_scores))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, performance_scores, marker='o')
plt.title("Performance vs. Number of Training Days")
plt.xlabel("Number of Training Days")
plt.ylabel("Performance Score")
plt.grid(True)
plt.xticks(training_sizes)
plt.tight_layout()

plt.show()