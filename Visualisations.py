import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime
import re
import sys
from sklearn import preprocessing
import pickle
from collections import defaultdict
import seaborn as sns

class HeatmapVisualizer:
    def __init__(self, begin_day: str, end_day: str, df: pd.DataFrame, name: str, title: str, verbose: bool = True) -> None:
        """TODO: add docstring!
 
        df is a pd.DataFrame with "time" column (10-minute intervals) and "location" column (string labels of locations).

        """
        # First we make sure to validate the user input.
        self.verbose = verbose
        self.name = name
        self.title = title
        self.validate_input(begin_day, end_day, df)

        # Set validated params
        self.begin_day = begin_day
        self.end_day = end_day

        # Filter the full dataset based on begin- and endday.
        self.df = df.set_index("time").loc[self.begin_day : self.end_day]

        # Optionally, print some info for debugging purposes.
        if self.verbose:
            print(
                f"Message (heatmap visualizer): Making heatmap with {len(self.df)} records, starting at {self.df.index.values[0]} and ending at {self.df.index.values[-1]}."
            )

        # Manual settings
        self.fig_height = 6  # Height of the output figure, needed for calculations.

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
            if not "time" in df or not "location" in df:
                raise ValueError(
                    "Make sure that df contains both the columns 'time' (datetime) and 'location' (strings of locations)!"
                )

            # Check if df.time is of type datetime64[ns] and that df.location is of type
            if (
                not df["time"].dtypes == "datetime64[ns]"
                or not df["location"].dtypes == "object"
            ):
                raise TypeError(
                    "Make sure that df.time is of dtype datetime64[ns] and df.location is of dtype object!"
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
        self.fig, self.ax = plt.subplots(figsize=(16, self.fig_height))
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
        self.ax.set_title(
            f"{self.title}. Location history of Significant Locations (\u0394t = 10min) from {self.begin_day} 00:00 to {self.end_day} 23:50"
        )
        self.ax.set_xlabel("Time")
        self.fig.savefig(f"output/{self.name}.png", dpi=1000)
        print(
            f"Message (heatmap visualizer): Succesfully downloaded heatmap to output/{self.name}.png."
        )

    def calc_parameters(self) -> None:
        """This function calculates all the parameters that are required for this visualization."""
        self.run_label_encoder()
        self.n_per_day = 144  # TODO: calculate this.
        self.n_days = (
            1
            + (
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
        days.time = pd.to_datetime(days.time)
        days["weekday"] = days.time.dt.day_name()
        return list(
            reversed(
                [
                    "{}, {}".format(a_, b_)
                    for a_, b_ in zip(
                        days.time.dt.strftime("%Y/%m/%d").values.tolist(),
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

        # color_list = [
        #     "#8dd3c7",
        #     "#ffffb3",
        #     "#bebada",
        #     "#fb8072",
        #     "#80b1d3",
        #     "#fdb462",
        #     "#b3de69",
        #     "#fccde5",
        #     "#d9d9d9",
        #     "#bc80bd",
        #     "#ccebc5",
        #     "#ffed6f",
        # ]
        # color_list = [plt.cm.tab20(i) for i in range(20)]
        # color_list = [
        #     "#F2F3F4",
        #     "#222222",
        #     "#F3C300",
        #     "#875692",
        #     "#F38400",
        #     "#A1CAF1",
        #     "#BE0032",
        #     "#C2B280",
        #     "#848482",
        #     "#008856",
        #     "#E68FAC",
        #     "#0067A5",
        #     "#F99379",
        #     "#604E97",
        #     "#F6A600",
        #     "#B3446C",
        #     "#DCD300",
        #     "#882D17",
        #     "#8DB600",
        #     "#654522",
        #     "#E25822",
        #     "#2B3D26",
        # ]  # These are the so called "kelly colors of maximum contrast"
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

        # The location labels can be quite long; this part cuts them (or rather, adds newlines) so that the width of the labels is max 20 chars.
        locations_cut = []
        for string in self.locations:
            chunks = [string[i : i + 20] for i in range(0, len(string), 20)]
            modified_string = "\n".join(chunks)
            locations_cut.append(modified_string.lstrip())

        # Now position the labels. We can, conveniently, use the bounds to position them correctly.
        for i, b in enumerate(self.bounds[:-1]):
            val = (self.bounds[i] + self.bounds[i + 1]) / 2
            self.cbar.ax.text(2, val, f"{locations_cut[i]}", ha="left", va="center")


class ModelPerformanceVisualizer():
    def __init__(self, scores:dict) -> None:
        if isinstance(scores, dict):
            self.scores = scores
        else:   
            with open('output/model_performances.pkl', 'rb') as f:
                self.scores = pickle.load(f)

        self.prepare_data()
        self.make_heatmap()
    
    def prepare_data(self):
        # Prepare data for the heatmap
        heatmap_data = []

        for training_size, forecast_scores in self.scores.items():
            training_days = int(training_size.split("_")[-1])
            avg_values = [np.mean(score_list) for score_list in forecast_scores.values()]
            heatmap_data.append([training_days, *avg_values])

        # Create a DataFrame from the data
        df = pd.DataFrame(heatmap_data, columns=["Training Days", *forecast_scores.keys()])
        self.df = df.set_index('Training Days')

    def make_heatmap(self):
        self.df = self.df.iloc[::-1,::-1].T # Reverse both rows and columns, and then transpose (to swap axes)
        sns.heatmap(self.df, cmap="Blues")
        path = "output/model_performance_heatmap.png"
        plt.savefig(path)
        plt.show()
        print(f"Saved model performance heatmap to output/model_performance_heatmap.png")

