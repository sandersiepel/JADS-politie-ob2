import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime
import re
import sys
from sklearn import preprocessing
import pickle
import seaborn as sns
import plotly.express as px
import base64
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go


class HeatmapVisualizer:
    def __init__(self, begin_day: str, end_day: str, df: pd.DataFrame, outputs_folder_name:str, name: str, title: str, verbose: bool = True) -> None:
        """TODO: add docstring!
 
        df is a pd.DataFrame with "timestamp" column (10-minute intervals) and "location" column (string labels of locations).

        """
        # First we make sure to validate the user input.
        self.verbose = verbose
        self.name = name
        self.title = title
        self.outputs_folder_name = outputs_folder_name
        self.validate_input(begin_day, end_day, df)

        # Set validated params
        self.begin_day = begin_day
        self.end_day = end_day

        # Filter the full dataset based on begin- and endday. We add "23:59" to the endday to also include the data of that day.
        self.df = df[df["timestamp"].between(self.begin_day, self.end_day + " 23:59", inclusive='both')]
        self.df = self.df.set_index("timestamp")

        # Optionally, print some info for debugging purposes.
        if self.verbose:
            print(
                f"Message (heatmap visualizer): Making heatmap with {len(self.df)} records, starting at {self.df.index.values[0]} and ending at {self.df.index.values[-1]}."
            )

        # Init funcs
        self.calc_parameters()
        self.make_plot()

    def validate_input(self, begin_day: str, end_day: str, df: pd.DataFrame) -> None:
        """Function that checks if the user's input satisfies all criteria. Officially there are more criteria, but these are the most important ones."""
        try:
            # Check if both begin_day and end_day are of the right type (str).
            if not isinstance(begin_day, str) or not isinstance(end_day, str):
                raise TypeError(
                    "Make sure that both begin_day and end_day are strings!"
                )

            # Check if both begin_day and end_day have a valid value ("yyyy-mm-dd").
            pat = r"\d{4}-\d{2}-\d{2}"
            if not re.fullmatch(pat, begin_day) or not re.fullmatch(pat, end_day):
                raise ValueError(
                    "Make sure that both begin_day and end_day are strings of format yyyy-mm-dd!"
                )

            # Check that self.df is of type pd.DataFrame
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Make sure that df is a pd.DataFrame!")

            # Check if df contains both columns 'time' and 'location'.
            if not "timestamp" in df or not "location" in df:
                raise ValueError(
                    "Make sure that df contains both the columns 'timestamp' (datetime) and 'location' (strings of locations)!"
                )

            # Check if df.time is of type datetime64[ns] and that df.location is of type
            if (
                not df["timestamp"].dtypes == "datetime64[ns]"
                or not df["location"].dtypes == "object"
            ):
                raise TypeError(
                    "Make sure that df.timestamp is of dtype datetime64[ns] and df.location is of dtype object!"
                )

        except ValueError as ve:
            print(f"Error! {ve}")
            sys.exit(1)

        except TypeError as te:
            print(f"Error! {te}")
            sys.exit(1)

        # TODO: Check if the intervals are correct.

    def make_plot(self) -> None:
        """Main function that is automatically ran upon initializing an instance of the HeatmapVisualizer class. This function adds everything together."""
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.set_y_ticks()
        self.set_x_ticks()
        self.im = self.ax.imshow(
            self.data,
            cmap=self.cmap,
            norm=self.boundaryNorm,
            aspect="auto",
            extent=[0, 144, 0, self.n_days],
        )
        self.add_grid()
        self.add_colorbar()
        # self.ax.set_title(
        #     f"{self.title}Location history of Significant Locations (\u0394t = 10min) from {self.begin_day} 00:00 to {self.end_day} 23:50"
        # )
        self.ax.set_xlabel("Timestamp")
        self.fig.tight_layout()

        # We encode the image with a buffer so that we can load it in a Dash application. 
        path = f"output/{self.outputs_folder_name}/{self.name}.png"
        plt.savefig(path, format="png", dpi=300)
        with open(path, 'rb') as image_file:
            self.encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
        if self.verbose:
            print(
                f"Message (heatmap visualizer): Succesfully downloaded heatmap to output/{self.outputs_folder_name}/{self.name}.png."
            )

    def get_encoded_fig(self) -> None:
        return self.encoded_image

    def calc_parameters(self) -> None:
        """This function calculates all the parameters that are required for this visualization."""
        self.run_label_encoder()
        self.n_per_day = 144 
        self.n_days = (
            1 + (
                datetime.strptime(self.end_day, "%Y-%m-%d").date()
                - datetime.strptime(self.begin_day, "%Y-%m-%d").date()
            ).days
        )

        # Reshape the data to create a matrix of n_days by n_per_day. TODO: check for ValueError (e.g., ValueError: cannot reshape array of size 1978 into shape (14,144)).
        self.original_data = self.df.location.values.reshape(
            self.n_days, self.n_per_day
        )  # Keep version of original dataset

        res, ind = np.unique(self.original_data.flatten(), return_index=True)
        self.locations = self.le.inverse_transform(
            res[np.argsort(ind)]
        )  # A list of the discrete locations that are found in sub_df, in their respective order.

        self.n_locations = len(
            self.locations
        )  # The number of unique locations that are in sub_df.

        self.data = self.calc_input_data()
        self.y_axis_labels = self.make_y_axis_labels()
        self.cmap = self.make_cmap()
        self.boundaryNorm = self.make_boundaryNorm()

    def run_label_encoder(self) -> None:
        """Our labels are, initially, strings of locations. For our visualization to work, we need integers. Therefore, we transform them with a LabelEncoder to integers."""
        # Check if df.locations is set and is integers. If not integers, use transformation
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.df.location)

        if not self.df.location.dtype == np.int32:
            # We need to transform self.df
            self.df.location = self.le.transform(self.df.location)

    def make_y_axis_labels(self) -> list:
        """Here we calculate which dates are included in our dataset and we add the weekday to it. These serve as our y-axis labels that we set in self.set_y_ticks()."""
        days = pd.DataFrame(self.df.index.strftime("%m/%d/%Y").unique())
        days.timestamp = pd.to_datetime(days.timestamp)
        days["weekday"] = days.timestamp.dt.day_name()
        return list(
            reversed(
                [
                    "{}, {}".format(a_, b_)
                    for a_, b_ in zip(
                        days.timestamp.dt.strftime("%Y/%m/%d").values.tolist(),
                        days.weekday.values.tolist(),
                    )
                ]
            )
        )

    def calc_input_data(self) -> None:
        """We have to transform the input data so that the class labels (integers) correspond to an ordered version of the labels.
        This is necessary to ensure that the labels correspond to the cmap (which uses the ordered values to assign colors).
        """
        trans_dict = {}
        res, ind = np.unique(self.original_data.flatten(), return_index=True)
        original_order = res[np.argsort(ind)]
        uniques = np.unique(self.original_data)

        for i, loc in enumerate(original_order):
            trans_dict[loc] = uniques[i]

        _data = self.df.location.values.tolist()
        return np.array([trans_dict[ele] for ele in _data]).reshape(
            self.n_days, self.n_per_day
        )

    def make_cmap(self) -> ListedColormap:
        """Make the colormap based on a pre-defined color scheme from matplotlib (in this case, the pastel1 scheme)."""
        color_list = [
            "#a6cee3",
            "#1f78b4",
            "#b2df8a",
            "#33a02c",
            "#fb9a99",
            "#e31a1c",
            "#fdbf6f",
            "#ff7f00",
            "#cab2d6",
            "#6a3d9a",
            "#ffff99",
            "#b15928",
            "#8dd3c7",
            "#ffffb3",
            "#bebada",
            "#fb8072",
            "#80b1d3",
            "#fdb462",
            "#b3de69",
            "#fccde5",
            "#d9d9d9",
            "#bc80bd",
            "#ccebc5",
            "#ffed6f",
        ]

        return ListedColormap(color_list[: self.n_locations])

    def make_boundaryNorm(self) -> BoundaryNorm:
        """BoundaryNorm is used to specify the boundaries for each class and its colors. We use this to make sure each class (i.e., location) is represented by one color.
        For each class (which is an integer), the bound should be [class-0.5, class+0.5] so that it always includes the class value.
        """
        # Calculate the bounds for the colors
        self.bounds = []  # Keep the bounds list for the colorbar ticks
        elements = np.unique(
            self.original_data
        )  # Elements = list of ordered unique location values
        for i, e in enumerate(elements):
            if i == 0:
                self.bounds.append(e - 0.5)
            self.bounds.append(e + 0.5)

        return BoundaryNorm(self.bounds, self.cmap.N)

    def set_y_ticks(self) -> None:
        """Set the y-axis ticks, i.e., the days of the week and the corresponding dates."""
        y_ticks = (
            np.arange(self.n_days) + 0.5
        )  # Offset the y-axis labels. Offset is +0.5 so that the labels are vertically centered in each row.
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(self.y_axis_labels)  # Labels are the days of the week

    def set_x_ticks(self) -> None:
        """Set the x-axis ticks for the time values."""
        x_ticks = np.arange(
            0, self.n_per_day, (self.n_per_day / 24)
        )  # Calculate the ticks depending on the number of intervals per day.

        # Make the intervals (strings) in format "year:month:day, weekday". TODO: make flexible for n_per_day.
        x_tick_labels = [
            f"{i:02d}:{j:02d}" for i in range(0, 24) for j in range(0, 60, 60)
        ]

        # Center the x-axis labels
        x_tick_positions = (
            x_ticks + 1
        )  # Offset is to shift the labels slightly to the right, for better alignment.
        self.ax.set_xticks(x_tick_positions)
        self.ax.set_xticklabels(
            x_tick_labels, rotation=90
        )  # Rotate x-axis labels by 90 degrees for better readability.

    def add_grid(self) -> None:
        """We disable the default grid (since its position is not flexible) and, instead, add hlines and vlines to replace the grid."""
        # TODO: make this flexible based on the number of intervals per day (default=144, aka 10 minutes).
        self.ax.grid(False)  # Disable default grid.
        self.ax.hlines(
            np.arange(self.n_days + 1), 0, 144, colors="white", linewidths=0.2
        )
        self.ax.vlines(
            np.arange(0.5, 144, 6), 0, self.n_days, colors="white", linewidths=0.5
        )

    def add_colorbar(self) -> None:
        """This function adds the colorbar to the right of the visualization."""
        self.cbar = plt.colorbar(self.im, ax=self.ax, aspect=20, ticks=self.bounds)
        self.cbar.ax.get_yaxis().set_ticks([])

        # The location labels can be quite long; this part cuts them (or rather, adds newlines) so that the width of the labels is max 30 chars.
        locations_cut = []
        for string in self.locations:
            chunks = [string[i : i + 30] for i in range(0, len(string), 30)]
            modified_string = "\n".join(chunks)
            locations_cut.append(modified_string.lstrip())

        # Now position the labels. We can, conveniently, use the bounds to position them correctly.
        for i, b in enumerate(self.bounds[:-1]):
            val = (self.bounds[i] + self.bounds[i + 1]) / 2
            self.cbar.ax.text(2, val, f"{locations_cut[i]}", ha="left", va="center")


class HeatmapVisualizerV2:
    def __init__(self, begin_day: str, end_day: str, df: pd.DataFrame, outputs_folder_name:str) -> None:
        """TODO: add docstring!
 
        df is a pd.DataFrame with "timestamp" column (10-minute intervals) and "location" column (string labels of locations).

        """
        # First we make sure to validate the user input.
        self.outputs_folder_name = outputs_folder_name

        # Set validated params
        self.begin_day = begin_day
        self.end_day = end_day

        start_date = pd.to_datetime(f"{begin_day} 00:00:00")
        end_date = pd.to_datetime(f"{end_day} 23:50:00")

        self.df = df[df["timestamp"].between(start_date, end_date)].copy()

        self.calculate_heatmap_data()
        self.make_figure()

    def calculate_heatmap_data(self):
        # Extract days and times from the timestamp
        self.df['day'] = self.df['timestamp'].dt.date
        self.df['time'] = self.df['timestamp'].dt.time

        # Encode location strings as integers
        encoder = LabelEncoder()
        self.df['location_encoded'] = encoder.fit_transform(self.df['location'])
        self.location_labels = encoder.classes_  # Get the original location labels

        self.n_locations = self.df.location_encoded.nunique()
        # Create a pivot table to prepare data for the heatmap
        self.pivot_table = self.df.pivot_table(index='day', columns='time', values='location_encoded', aggfunc='first')

        print(self.pivot_table)

        # Convert the pivot table to a NumPy array
        self.heatmap_data = self.pivot_table.values

        print(self.heatmap_data)

    def make_figure(self):
        # Get the x-axis (time) and y-axis (days) labels
        x_labels = self.pivot_table.columns
        y_labels = self.pivot_table.index

        # Add day of the week to the y-axis labels
        y_labels_with_dow = [f"{day} - {day.strftime('%A')[:3]} " for day in y_labels]

        colors = [
            "#a6cee3",
            "#1f78b4",
            "#b2df8a",
            "#33a02c",
            "#fb9a99",
            "#e31a1c",
            "#fdbf6f",
            "#ff7f00",
            "#cab2d6",
            "#6a3d9a",
            "#ffff99",
            "#b15928",
            "#8dd3c7",
            "#ffffb3",
            "#bebada",
            "#fb8072",
            "#80b1d3",
            "#fdb462",
            "#b3de69",
            "#fccde5",
            "#d9d9d9",
            "#bc80bd",
            "#ccebc5",
            "#ffed6f",
        ]

        # First we create a list of np.linspace values where each value repeats twice, except for the beginning (0) and the ending (1)
        vals = np.r_[np.array(0), np.repeat(list(np.linspace(0, 1, self.n_locations+1))[1:-1], 2), np.array(1)]

        # Then we make a list that contains lists of the values and the corresponding colors.
        cc_scale = [[j, colors[i//2]] for i, j in enumerate(vals)]

        # Create the heatmap using Plotly
        self.fig = go.Figure(data=go.Heatmap(
            z=self.heatmap_data,
            x=x_labels,
            y=y_labels_with_dow,
            colorscale=cc_scale,
            colorbar=dict(
                tickvals=np.linspace(1/self.n_locations/2, 1 - 1/self.n_locations/2, self.n_locations) * (self.n_locations - 1), # Center the ticks 
                ticktext=self.location_labels,
                title='Location'
            ),
        ))
        
        self.fig.update_layout(
            xaxis=dict(title='Time of Day'),
            yaxis=dict(title='Date'),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # self.fig.update_yaxes(autorange="reversed")

        return self.fig

    def get_fig(self):
        return self.fig


class ModelPerformanceVisualizer():
    def __init__(self, scores:dict, outputs_folder_name:str) -> None:
        self.outputs_folder_name = outputs_folder_name

        if isinstance(scores, dict):
            self.scores = scores
        else:   
            with open(f"output/{self.outputs_folder_name}/model_performances.pkl", 'rb') as f:
                self.scores = pickle.load(f)

        # Make a heatmap with mean/median scores of each combination of number of training days and horizon days. 
        self.heatmap()

        # Make a line graph with the performance of the models for each number of days predicting into the future. 
        self.accuracy_horizons()

        # Make a line graph with the performance of the models for each number of training days.
        self.accuracy_per_training_days()

    def accuracy_horizons(self) -> None:
        new_scores = {}
        for k, _ in self.scores.items():
            # key (k) is "number of training days"
            for x, y in self.scores[k].items():
                    h = x.split("_")[-1]
                    if not h in new_scores:
                        new_scores[h] = []

                    new_scores[h].extend(y)

        means = [np.mean(v) for _, v in new_scores.items()]

        # Plot results in a line graph
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(means)
        ax.set_title("Model performance per number of days into the future")
        ax.set_xlabel("Number of days into the future")
        ax.set_ylabel("Model performance (accuracy)")
        fig.savefig(f"output/{self.outputs_folder_name}/model_performance_per_horizon.png", dpi=600)
        plt.clf()

    def accuracy_per_training_days(self) -> None:
        means = []
        for _, v in self.scores.items():
            data = []
            for _, y in v.items():
                data.extend(y)

            means.append(np.mean(data))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(means)
        ax.set_title("Model performance per number of days used for training")
        ax.set_xlabel("Number of days used for training")
        ax.set_ylabel("Model performance (accuracy)")
        fig.savefig(f"output/{self.outputs_folder_name}/model_performance_per_training_days.png", dpi=600)
        plt.clf()

    def heatmap(self) -> None:
        def prepare_data(self) -> None:
            # Prepare data for the heatmap
            heatmap_data = []

            for training_size, forecast_scores in self.scores.items():
                training_days = int(training_size.split("_")[-1])
                avg_values = [np.mean(score_list) for score_list in forecast_scores.values()]
                heatmap_data.append([training_days, *avg_values])

            # Create a DataFrame from the data
            df = pd.DataFrame(heatmap_data, columns=["Training Days", *forecast_scores.keys()])
            self.df = df.set_index('Training Days')
            self.df.index.names = ['Number of days used for training']
            self.df.columns = self.df.columns.str.split('_').str[-1]

        def make_heatmap(self) -> None:
            self.df = self.df.iloc[::-1,::-1].T # Reverse both rows and columns, and then transpose (to swap axes)
            fig, ax = plt.subplots(figsize=(10,5))
            sns.heatmap(self.df, cmap="Blues") #  xticklabels = 10, yticklabels=7
            plt.ylabel('Number of days into the future')
            plt.xticks(rotation=0) 
            plt.savefig(f"output/{self.outputs_folder_name}/model_performance_heatmap.png")
            # plt.show()
            plt.clf() # Clear figure command to avoid stacking axes in consecutive plots.

        prepare_data(self)
        make_heatmap(self)


class EDA():
    def __init__(self, data:pd.DataFrame, outputs_folder_name:str) -> None:
        self.outputs_folder_name = outputs_folder_name
        self.data = data # Here, data is the raw (meaning: timestamps and coordinates) dataset. 

        return None
    
    def records_per_day(self):
        counts = self.data.groupby(self.data.timestamp.dt.date).size().to_frame()
        counts.columns = ['day_count']
        counts = counts.reset_index()

        self.fig = px.bar(counts, x='timestamp', y='day_count')
        self.fig.update_layout(xaxis_title="Time", yaxis_title="Number of datapoints", margin=dict(l=0, r=0, t=0, b=0))
        self.fig.write_html(f"output/{self.outputs_folder_name}/EDA_records_per_day.html") 


class DataPredicatability():
    def __init__(self, df: pd.DataFrame, rolling_window_size=10):
        self.df = df # Df should contain the columns: timestamp, location, and the temporal features. Basically, this is the df after loading the resampled 10-min block dataset + the temp features.
        self.rolling_window_size = rolling_window_size # TODO: add documentation for this parameter.

    def run(self):
        data = self.make_dataset()
        return self.make_graph(data, savgol_filter=True)

    def make_dataset(self) -> list:
        from collections import defaultdict

        res, final_res = defaultdict(dict), defaultdict(dict)
        this_day_values = []

        for _, row in self.df.iterrows():
            m = (row['weekday'], row['hour'], row['window_block'])

            if not 'data' in res[m]: res[m]['data'] = {}

            try:
                # Calculate the probability of having this location in this moment
                this_day_values.append(res[m]['data'][row['location']] / sum(res[m]['data'].values()))
            except KeyError:
                # The location is not in the dict, so we default to 0. This happens when the location hasn't been visited before in this moment. 
                this_day_values.append(0)
            
            # Now we update the res dict with data of current window block
            if row['location'] in res[m]['data']:
                res[m]['data'][row['location']] += 1
            else:
                res[m]['data'][row['location']] = 1

            # If we now exceeded THRESHOLD, remove the last entry (like a rolling window approach)
            if sum(res[m]['data'].values()) >= self.rolling_window_size:
                # Minus one for last entry
                res[m]['data'][res[m]['meta']['last_location']] -= 1

            res[m]['meta'] = {'last_location':row['location']}

            # Now we check if this window block is the last one of the day. If so, save score for this day and reset variables for next day.
            if row['hour'] == 23 and row['window_block'] == 5: # We know that the day has ended when we hit timestamp 23:50. 
                final_res[row['timestamp'].date()] = sum(this_day_values)/len(this_day_values)
                this_day_values = []

        return final_res
    
    def make_graph(self, data, savgol_filter=True):
        import plotly.graph_objects as go
        from scipy import signal

        x,y = zip(*sorted(data.items()))
        df_plot = pd.DataFrame({"Time":x, "Score":y})

        fig = go.Figure()

        if savgol_filter:
            fig.add_trace(go.Scatter(
                x=df_plot.Time.values.tolist(),
                y=signal.savgol_filter(df_plot['Score'].values.tolist(),
                                    20, # window size used for filtering
                                    3), # order of fitted polynomial
                name='Savitzky-Golay'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df_plot.Time.values.tolist(),
                y=y
            ))

        fig.update_layout(xaxis_title="Time", yaxis_title="Predictability", margin=dict(l=0, r=0, t=0, b=0))

        return fig