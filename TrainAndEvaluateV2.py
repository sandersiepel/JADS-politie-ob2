from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sys
import warnings
import sklearn.exceptions
from datetime import datetime
from collections import defaultdict
import pickle
from tqdm import tqdm
import math
import numpy as np
from Visualisations import ModelPerformanceVisualizer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', message='Mean of empty slice')


class TrainAndEvaluate:
    def __init__(self, df: pd.DataFrame, start_date:datetime, end_date:datetime, training_window_size: int, horizon_size: int, window_step_size: int, model_features=list) -> None:
        """ This class works with the resampled_df_10_min.xlsx file (or, with its df). It will train a ML model multiple times with the following scheme:
        train with a training set size between 1 day and training_window_size days. Then test on horizon_size number of days. Then move train + test window
        with window_step_size days into the future. 

        Also, all of the steps are also performed for a baseline model. This model simply predicts based on counts; it will always predict the location
        with the highest count for a particular time-window. 
        """
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        self.model_features = model_features
        self.training_window_size = training_window_size
        self.horizon_size = horizon_size
        self.window_step_size = window_step_size
        self.log = []

        # The self.performance dict contains, for each training size and each validation loop, the accuracy scores for all the days that were predicted (where number of days = (max_n_testing_days - min_n_testing_days))
        self.performance, self.baseline_performance = defaultdict(dict), defaultdict(dict)

    def main(self):
        # Step 1. Load dataset.
        self.make_dataset()

        # Step 2. Make temporal features.
        self.make_temporal_features()

        # Step 3. Make train/test split. We need the loop index to offset the days (for making the training/testing sets).
        self.n_windows = 1 + math.floor(((self.end_date - self.start_date).days - self.training_window_size - self.horizon_size) / self.window_step_size)
        self.offset_days = ((self.end_date - self.start_date).days - self.training_window_size - self.horizon_size) % self.window_step_size
        print(f"Message (ML model): n_days: {(self.end_date - self.start_date).days}, n_windows (ie blocks): {self.n_windows}, offset_days: {self.offset_days}")

        for block_index in tqdm(range(self.n_windows), desc=" Block loop", position=1): # Loop 7 times    
            self.block_index = block_index

            for train_index in tqdm(range(self.training_window_size), desc=" Training window size loop", position=0, leave=False):
                self.train_index = train_index

                self.make_train_test_split()

                # Step 4. Run model and make the predictions.
                self.run_model()

                # Step 5. Evaluate model performance and store results in self.performance dict.
                self.evaluate_model()

            # Save current heatmap
            ModelPerformanceVisualizer(
                scores=self.performance,
                name=f"test1/performance_{block_index}"
            )

        with open('output/model_performances.pkl', 'wb') as f:
            pickle.dump(self.performance, f)

        # with open('output/baseline_performances.pkl', 'wb') as f:
        #     pickle.dump(self.baseline_performance, f)

        print("\nSaved model performance to output/model_performances.pkl")
        # print("\nSaved baseline performance to output/baseline_performances.pkl")

        # Save log file
        with open('output/logfile.txt', 'w') as f:
            for line in self.log:
                f.write(f"{line}\n")

        print("Saved logfile to output/logfile.txt")

        return self.performance, self.baseline_performance

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
        if "day" in self.model_features:
            self.df["day"] = self.df["time"].dt.day

        if "weekday" in self.model_features:
            self.df["weekday"] = self.df["time"].dt.dayofweek

        if "hour" in self.model_features:
            self.df["hour"] = self.df["time"].dt.hour

        if "window_block" in self.model_features:
            self.df["window_block"] = ((self.df['time'].dt.minute * 60 + self.df['time'].dt.second) // 600).astype(int)

    def make_train_test_split(self) -> None:        
        self.train_start_date = self.start_date + pd.Timedelta(days=self.offset_days+(self.block_index * self.window_step_size)+self.train_index)
        self.train_end_date = self.train_start_date + pd.Timedelta(days=(self.training_window_size-1)-self.train_index, hours=23, minutes=50)
        self.test_start_date = self.train_end_date + pd.Timedelta(minutes=10)
        self.test_end_date = self.test_start_date + pd.Timedelta(days=self.horizon_size-1, hours=23, minutes=50)

        self.train_mask = self.df["time"].between(self.train_start_date, self.train_end_date)
        self.test_mask = self.df["time"].between(self.test_start_date, self.test_end_date)

        # Split the data into train and test sets
        self.X_train = self.df.loc[self.train_mask, self.model_features]
        self.y_train = self.df.loc[self.train_mask, "location"]
        self.X_test = self.df.loc[self.test_mask, self.model_features]
        self.y_test = self.df.loc[self.test_mask, "location"]

        # print(f"Block {self.block_index}, train size: {self.train_index}. Training: {self.train_start_date}-{self.train_end_date}, testing: {self.test_start_date}-{self.test_end_date}.")

    def run_model(self) -> None:
        self.model = RandomForestClassifier()

        # We use the sample_weight parameter to favour more recent datapoints. TODO: maybe using a weekly pattern in these sampling weights is better?
        self.model.fit(self.X_train, self.y_train) # , sample_weight=np.logspace(0.1, 1, len(self.X_train))/10

        # Make predictions for 14 days into the future. 
        self.predictions = self.model.predict(self.X_test)

        # Lastly, we run our baseline model (for comparison reasons).
        # self.run_baseline_model()

    def run_baseline_model(self):
        # First we create a dataframe "most_common_locations" that holds the most common location for each combination of the model's features (e.g., hourofday, dayofweek, windowblock)
        # This should be based on the training data (X+y)! 
        training_data = self.df.loc[self.train_mask]
        testing_data = self.df.loc[self.test_mask]

        most_common_locations = training_data.groupby(self.model_features)['location'].apply(lambda x: x.value_counts().idxmax()).reset_index()
        result_df = testing_data.merge(most_common_locations, how="left", left_on=self.model_features, right_on=self.model_features)
        
        features_to_use = self.model_features[1:]
        while result_df['location_y'].isna().sum() > 0:
            most_common_locations = training_data[["location"] + features_to_use].groupby(features_to_use)['location'].apply(lambda x: x.value_counts().idxmax()).reset_index()
            result_df = result_df = testing_data.merge(most_common_locations, how="left", left_on=features_to_use, right_on=features_to_use)
            features_to_use = features_to_use[1:]  # Remove the first element to exclude it from the next merge

        # print(f"Number of NAN values in predictions: {result_df.location_y.isna().sum()}\n")
        self.baseline_predictions = result_df.location_y

    def evaluate_model(self) -> None:
        self.log.append(f"Block {self.block_index}, train window size: {self.train_index}. Training: {self.train_start_date}-{self.train_end_date}, testing: {self.test_start_date}-{self.test_end_date}.")
        accs = []
        for d in range(self.horizon_size):
            # First, evaluate the ML model's predictions and store acc in self.performance
            this_day_predictions = self.predictions[d*144:(d+1)*144]
            this_day_actual_values = self.y_test[d*144:(d+1)*144]
            acc = accuracy_score(this_day_actual_values, this_day_predictions)

            if f"days_into_future_{d}" not in self.performance[f"training_set_size_{self.training_window_size - self.train_index}"]:
                self.performance[f"training_set_size_{self.training_window_size - self.train_index}"][f"days_into_future_{d}"] = []

            self.performance[f"training_set_size_{self.training_window_size - self.train_index}"][f"days_into_future_{d}"].append(round(acc, 4))
            accs.append(acc)

        self.log.append(f"Found accuracies: {accs}. \n")
        # for d in range(self.horizon_size):
        #     # Then, evaluate the baseline's predictions and store acc in self.baseline_performance
        #     this_day_predictions = self.baseline_predictions[d*144:(d+1)*144]
        #     this_day_actual_values = self.y_test[d*144:(d+1)*144]
        #     acc = accuracy_score(this_day_actual_values, this_day_predictions)

        #     if f"days_into_future_{d}" not in self.baseline_performance[f"training_set_size_{self.training_window_size - self.train_index}"]:
        #         self.baseline_performance[f"training_set_size_{self.training_window_size - self.train_index}"][f"days_into_future_{d}"] = []

        #     self.baseline_performance[f"training_set_size_{self.training_window_size - self.train_index}"][f"days_into_future_{d}"].append(round(acc, 4))


# # Initialize parameters.
# data_source = "google_maps"  # Can be either 'google_maps' or 'routined'.
# # hours_offset is used to offset the timestamps to account for timezone differences. For google maps, timestamp comes in GMT+0
# # which means that we need to offset it by 2 hours to make it GMT+2 (Dutch timezone). Value must be INT!
# hours_offset = 2 # Should be 0 for routined and 2 for google_maps. 
# # begin_date and end_date are used to filter the data for your analysis.
# begin_date = "2022-10-01"
# end_date = "2023-05-01"  # End date is INclusive! 
# # FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
# fraction = 1
# # For the model performance class we need to specify the number of training days (range) and testing horizon (also in days)
# training_window_size = 30
# horizon_size = 10
# window_step_size = 1

# from Visualisations import ModelPerformanceVisualizer

# # scores, baseline_scores = TrainAndEvaluate(
# #         df = None,
# #         start_date = pd.to_datetime(f"{begin_date} 00:00:00"),
# #         end_date = pd.to_datetime(f"{end_date} 23:50:00"),
# #         training_window_size = training_window_size,
# #         horizon_size = horizon_size,
# #         window_step_size = window_step_size,
# #         model_features = ["day", "weekday", "hour", "window_block"],
# #     ).main()

# # Step 8. Visualize model performance. Input: 'scores', which is a dict. 
# ModelPerformanceVisualizer(
#     scores=None,
#     name="model_performances"
# )

# # Step 8. Visualize model performance. Input: 'scores', which is a dict. 
# ModelPerformanceVisualizer(
#     scores=None,
#     name="baseline_performances"
# )