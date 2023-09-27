from __future__ import annotations
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import requests
import plotly.express as px
from typing import Tuple
import os


class Cluster:
    def __init__(
        self,
        df: pd.DataFrame,
        outputs_folder_name:str,
        verbose: bool = False,
        pre_filter: bool = True,
        post_filter: bool = True,
        filter_moving: bool = True,
        centroid_k: int = 5,
        min_unique_days: int = 2,
    ) -> None:
        """init

        Parameters:
            df (DataFrame): pd.DataFrame with columns: latitude, longitude, timestamp
            verbose (boolean): yes or no print statements
            pre_filter (boolean): apply a filter BEFORE clustering (such as removing jitter)
            post_filter (boolean): apply a filter AFTER clustering (such as deleting homogeneous clusters)
            filter_moving (boolean): filter out the coordinate points where the subject was moving
            centroid_k (int): Number of nearest neighbors to consider for density calculation (default: 5)
            min_unique_days (int): If post_filter = True, then delete all clusters that have been visited on less than min_unique_days days.
        """

        # Make a deep copy to not adjust the original df
        self.df = df.copy(deep=True)
        self.outputs_folder_name = outputs_folder_name
        self.verbose = verbose
        self.pre_filter = pre_filter
        self.post_filter = post_filter
        self.filter_moving = filter_moving
        self.centroid_k = centroid_k
        self.min_unique_days = min_unique_days

        # Run init functions
        if self.pre_filter:
            self._apply_pre_filters()

    def _apply_pre_filters(self) -> None:
        """This function applies some filters BEFORE clustering to the original/raw dataframe to clean the data.

        This includes:
        - Filter out points where the subject was moving with _filter_moving()
        - TODO: jitter
        - TODO: aggregation of data points (e.g., within 10 seconds)

        Parameters:

        Returns: None

        """

        if self.filter_moving:
            # TODO: find out why some points are misclassified as either moving/not moving.
            self._filter_moving()
            self.df = self.df[self.df.moving == False]

    def _apply_post_filters(self) -> None:
        """Here we apply filters AFTER clustering to the dataset that contains the cluster labels.

        This includes:
        - Filter clusters that are homogeneous (these are deleted) with _post_filter_homogeneous_clusters()
        - Filter clusters where the mean/std of distances to centroid is lower than MEAN_STD_LOWER and higher than MEAN_STD_UPPER with _post_filter_mean_std_ratio_distances()
        - Filter clusters where ratio between num_datapoints and unique_days < MIN_RECORDS_PER_DAY with ...

        Parameters:

        Returns: None

        """

        # Here we define all the parameters for the post filtering. TODO: how can a detective change these based on rationale?
        mean_std_lower = 0.5
        mean_std_upper = 5

        # First we delete homogeneous clusters.
        self._post_filter_homogeneous_clusters()

        # Then we delete clusters with mean/std of distances to centroid <= MEAN_STD_LOWER or >= MEAN_STD_UPPER
        self._post_filter_mean_std_ratio_distances(mean_std_lower, mean_std_upper)

        # Delete those clusters that are visited less times than self.min_unique_days
        self._post_filter_min_unique_days(self.min_unique_days)

    def _delete_clusters_from_list(self, cluster_list: list, source: str) -> None:
        """This function deletes all clusters in cluster_list from both self.df (by setting cluster label to -1) and self.df_centroids (by dropping the row).

        Parameters:
            cluster_list (list): list of cluster labels (strings)
            source (string): represents from which action we need to delete the clusters; used for printing a message.

        Returns:
            None
        """

        # For the clusters in cluster_list, set the cluster label in self.df as -1 (i.e., noise)
        self.df.loc[self.df["cluster"].isin(cluster_list), "cluster"] = "-1"

        # And drop the rows from df_centroids
        len_before = len(self.df_centroids)
        self.df_centroids.drop(
            self.df_centroids[self.df_centroids.cluster.isin(cluster_list)].index,
            inplace=True,
        )

        if self.verbose:
            print(
                f"Message ({source}): Deleted {len_before - len(self.df_centroids)} clusters (with labels: {cluster_list})"
            )

    def _post_filter_min_unique_days(self, min_unique_days: int) -> None:
        """This function applies a post filter on self.df. For those clusters where the number of unique days visited is < min_unique_days,
        the cluster label is set to -1 (i.e., noise).

        Parameters:
            min_unique_days (int): Clusters with < min_unique_days are deleted.

        Returns:
            None
        """

        # Check which clusters have df_centroids["unique_days"] < min_unique_days, and make a list of those clusters.
        clusters_to_delete = self.df_centroids[
            self.df_centroids.unique_days < min_unique_days
        ].cluster.values.tolist()

        if len(clusters_to_delete) > 0:
            # And delete them from self.df and self.df_centroids
            self._delete_clusters_from_list(
                clusters_to_delete, source="post filter min unique days"
            )

    def _post_filter_mean_std_ratio_distances(self, mean_std_lower: float, mean_std_upper: float) -> None:
        """This function applies a post filter on the mean_std variable of self.df_centroids. It deletes those clusters where the value is either < 0.5 or > 5.
        The rationale behind this is that the spreadness of data points in a "valid cluster" should follow a certain distribution
        i.e., very dense in the center and gradually becoming less dense if points are further away from the center. Many of the clusters that would qualify
        to be deleted by this function, are already deleted by _post_filter_homoegeneous_clusters.

        Parameters:
            mean_std_lower: lower threshold on which to filter mean_std
            mean_std_upper: upper threshold on which to filter mean_std

        Returns:
            None
        """

        clusters_to_delete = self.df_centroids[
            (self.df_centroids.mean_std <= mean_std_lower)
            | (self.df_centroids.mean_std >= mean_std_upper)
        ].cluster.values.tolist()

        if len(clusters_to_delete) > 0:
            # And delete them from self.df and self.df_centroids
            self._delete_clusters_from_list(
                clusters_to_delete, source="post filter mean std ratio"
            )

    def _post_filter_homogeneous_clusters(self) -> None:
        """This function removes those clusters that only contain points with the exact same latitude/longitude.
        These clusters do not represent actual clusters, but only exist because of A-GPS and were therefore
        generated by accidentally picking up wrong Access-Points/WiFi networks. Or they exist of only cell tower points
        that are very inaccurate. We call these clusters the "homogeneous clusters" i.e., invalid clusters.

        Parameters:
            None

        Returns:
            None

        """

        # TODO: check if this makes sense. Maybe these are actual locations with just a fixed GPS/AP point?
        # TODO: filter those clusters where almost all points are exactly the same?
        # TODO: possibly implement this in "cluster certainty" metric

        clusters_to_delete = []

        def _all_cols_the_same(df):
            """This function checks if a passed dataframe ("df") contains only the same rows. If so, returns True."""
            a = df.to_numpy()
            return (a[0] == a).all()

        for i in range(self.df.cluster.nunique() - 1):
            # Select long/lat data for every cluster
            d = self.df[self.df.cluster == str(i)][["latitude", "longitude"]]

            # Check if this cluster only contains the same long/lat pairs
            if _all_cols_the_same(d):
                clusters_to_delete.append(str(i))

        if len(clusters_to_delete) > 0:
            # And delete them from self.df and self.df_centroids
            self._delete_clusters_from_list(
                clusters_to_delete, source="post filter homogeneous clusters"
            )

    def _filter_moving(self) -> None:
        """This function adds a column to self.df named "moving" that states, for every coordinate point,
        whether or not the subject was moving when the data point was generated. This is calculated based on the
        difference in time in seconds (delta_t) and geographical location in meters (delta_d) between this data
        point and the next one. Delta_t/delta_d calculates the distance per seconds. If this value is < 1 then we
        assume the subject to be stationary. If > 1 we assume the subject was moving.

        Parameters:

        Returns: None

        """

        delta_time_s = self.df["timestamp"].diff().dt.total_seconds()
        delta_distance_m = [0] + [
            haversine(
                (self.df["latitude"].iloc[i], self.df["longitude"].iloc[i]),
                (self.df["latitude"].iloc[i - 1], self.df["longitude"].iloc[i - 1]),
                unit=Unit.METERS,
            )
            for i in range(1, len(self.df))
        ]
        movement_m_per_s = delta_distance_m / delta_time_s
        self.df["moving"] = movement_m_per_s > 1

        # Create new df with moving data for testing purposes. TODO: delete this later to save memory
        self.df_moving = self.df.copy(deep=True)
        self.df_moving["delta_time_s"] = delta_time_s
        self.df_moving["delta_distance_m"] = delta_distance_m
        self.df_moving["movement_m_per_s"] = movement_m_per_s

        if self.verbose:
            print(
                f"Message (filter moving): Marked {self.df.moving.sum()} data points as moving."
            )

    def _perform_DBSCAN(self, eps: float, min_samples: int) -> None:
        """Runs the sklearn DBSCAN algorithm with min_samples and eps parameters.

        Parameters:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point

        Returns: None

        """

        # Since our data is in radians (required by haversine distance function), we have to transform our
        # epsilon value to radians as well.
        epsilon = (
            eps / 6371.0088
        )  # Eps is in kms so 0.005 equals 5 meters. We divide this by the earth's radius.

        if self.verbose:
            print("Message (clustering): Start clustering...")

        start_time = time.time()

        self.db = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            algorithm="ball_tree",
            metric="haversine",
        )

        # np.radians transforms the coordinates into radians with the formula: coordinates * np.pi / 180
        self.db.fit(np.radians(self.df[["latitude", "longitude"]].values))
        runtime = time.time() - start_time

        if self.verbose:
            print(f"Message (clustering): Clustering took {runtime} seconds.")

        # Add the cluster labels to self.df
        self.df["cluster"] = self.db.labels_.astype(str)

    def _perform_HDBSCAN(self, min_samples: int, min_cluster_size: int) -> None:
        """Runs HDBSCAN algorithm with min_samples and eps parameters.

        Parameters:
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point
            min_cluster_size (int): The minimum size a final cluster can be. The higher this is, the bigger your clusters will be

        Returns: None

        """

        if self.verbose:
            print("Message (clustering): Start clustering...")

        start_time = time.time()

        self.db = hdbscan.HDBSCAN(
            cluster_selection_epsilon=0.01 / 6371.0088,
            min_cluster_size=min_cluster_size,
            gen_min_span_tree=True,
            metric="haversine",
            min_samples=min_samples,
        )
        self.db.fit(np.radians(self.df[["latitude", "longitude"]].values))

        runtime = time.time() - start_time

        if self.verbose:
            print(f"Message (clustering): Clustering took {runtime} seconds.")

        # Add the cluster labels to self.df
        self.df["cluster"] = self.db.labels_.astype(str)

    def run_clustering(self, eps: float, min_samples: int, algorithm: str, min_cluster_size: int = 20) -> Cluster:
        """This function clusters self.df with the DBSCAN algorithm.

        Parameters:
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point
        algorithm (string): Which algorithm to use (either DBSCAN or HDBSCAN)
        min_cluster_size (int): The minimum size a final cluster can be. The higher this is, the bigger your clusters will be

        Returns: self

        """

        if self.verbose:
            print(
                f"Message (clustering): Clustering {len(self.df)} data points with DBSCAN, with eps = {eps}, min_samples = {min_samples}. "
            )

        # Run DBSCAN or HDBSCAN?
        if algorithm.lower() == "dbscan":
            self._perform_DBSCAN(eps, min_samples)
        elif algorithm.lower() == "hdbscan":
            self._perform_HDBSCAN(min_samples, min_cluster_size)

        # Now that we have our clusters, we build a separate dataset that, for each cluster, contains its
        # center of mass (i.e., centroid) and that centroid's OSM data. TODO: add more descriptive features for the clusters.
        self._enrich_clusters()

        # Apply the POST clustering filters to the data/clusters
        if self.post_filter:
            self._apply_post_filters()

        if self.verbose:
            print(
                f"Message (clustering): Final number of clusters: {len(self.df_centroids)}."
            )

        return self

    def _enrich_clusters(self) -> None:
        """
        Enriches the clusters that we found by building a dataframe (self.df_centroids). This dataframe contains information
        on each cluster, namely: its centroid (center of mass), number of data points, number of unique days, and OSM data.

        Parameters: None

        Returns: None

        """

        # First create self.df_centroids that contains, for each cluster: latitude, longitude
        self.df_centroids = self._build_centroid_dataframe()

        # Then, for each of the cluster centroids, get their OSM location data
        self._add_OSM_to_clusters()

        # Add feature: nr of data points per cluster
        self.df_centroids["num_datapoints"] = self.df_centroids.apply(
            lambda row: len(self.df[self.df.cluster == str(row["cluster"])]), axis=1
        )

        # Add feature: how many days the subject was present in this cluster
        self.df_centroids["unique_days"] = self.df_centroids.apply(
            lambda row: len(
                self.df[self.df.cluster == row["cluster"]]
                .timestamp.dt.normalize()
                .unique()
            ),
            axis=1,
        )

        # Add feature: mean divided by std of the distances of each point to the centroid
        def _calc_mean_over_std(row) -> None:
            temp_df = self.df[self.df.cluster == row["cluster"]].copy()
            temp_df["distance"] = temp_df.apply(
                lambda row2: haversine(
                    (row["latitude"], row["longitude"]),
                    (row2["latitude"], row2["longitude"]),
                    unit=Unit.METERS,
                ),
                axis=1,
            )

            mean = temp_df.distance.describe()["mean"]
            std = temp_df.distance.describe()["std"]

            # Avoid zerodivision error
            if mean == 0 or std == 0:
                return 0
            else:
                return mean / std

        self.df_centroids["mean_std"] = self.df_centroids.apply(
            lambda row: _calc_mean_over_std(row), axis=1
        )


    def _add_OSM_to_clusters(self) -> None:
        """
        Enriches the already existing self.df_centroids with OSM data

        Parameters: None

        Returns: None

        """

        if self.verbose:
            print(
                f"Message (OSM): Adding OSM location data to {len(self.df_centroids)} clusters."
            )

        self.df_centroids["location"] = self.df_centroids.apply(
            lambda row: self._OSM_request(row["latitude"], row["longitude"]), axis=1,
        )

    @staticmethod
    def _OSM_request(latitude: float, longitude: float) -> str:
        """
        Returns OSM data for a coordinate point

        Parameters:
            latitude (float): latitude coordinate value
            longitude (float): longitude coordinate value

        Returns:
            list: list with json dicts with OSM data

        """

        # TODO: make this faster somehow?

        url = f"https://nominatim.openstreetmap.org/search.php?q={latitude},{longitude}&polygon_geojson=1&format=json"
        response = requests.get(url, params={})
        data = response.json()

        if not "display_name" in data[0]:
            print(f"Message (Clustering OSM): could not retrieve OSM data, returning 'unknown' as cluster label.")
            return "unknown"
        else:
            return data[0]["display_name"]

    def _build_centroid_dataframe(self) -> pd.DataFrame:
        """
        Builds the centroid dataframe that contains, for each cluster, a lat/long value, cluster label, size and color.

        Parameters: None

        Returns:
            DataFrame: containing columns: latitude, longitude, cluster, size, color. Length of df is equal to self.df.cluster.nunique().
        """

        d = []

        for cluster_label in set(self.df.cluster.values.tolist()):
            # Ignore the cluster label -1 (i.e., noise)
            if cluster_label != "-1":
                # Select the data (long/lat) for cluster x
                t_df = self.df[self.df.cluster == str(cluster_label)][
                    ["latitude", "longitude"]
                ].values.tolist()

                # Calculate long/lat with center of mass
                # TODO: scale k to length of cluster? Most likely not very important.
                latitude, longitude = self._calculate_centroids(t_df, k=self.centroid_k)

                d.append(
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "cluster": str(cluster_label),
                        "size": 10,
                        "color": "black",
                    }
                )

        return pd.DataFrame(d)

    @staticmethod
    def _calculate_centroids(points, k: int = 3) -> Tuple[int, int]:
        """
        Calculate the weighted centroid of a list of points using their density values based on the number of nearest neighbors.

        Parameters:
            points (list): List of tuples/lists in the format (latitude, longitude), where
                           latitude and longitude are floats representing the coordinates.
            k (int): Number of nearest neighbors to consider for density calculation (default: 3).

        Returns:
            tuple: Latitude and longitude coordinates of the weighted centroid.
        """

        if len(points) <= k:
            k = len(points) - 1

        # Extract latitude and longitude from input points
        latitudes = np.array([p[0] for p in points])
        longitudes = np.array([p[1] for p in points])

        # Build KDTree for efficient nearest neighbor search
        tree = KDTree(np.column_stack((latitudes, longitudes)))

        # Calculate density based on number of nearest neighbors or radius
        # Here we have to make sure that k is <= number of data points (i.e., length of latitudes/longitudes variables)
        try:
            _, indices = tree.query(np.column_stack((latitudes, longitudes)), k=k + 1)
        except ValueError:
            print(
                f"Error (calculate centroids): k must be less than or equal to the number of training points"
            )

        # Note: +1 in k to include the point itself in the neighbors
        densities = k / np.sum(
            indices != points.index, axis=1
        )  # Exclude self from neighbors

        # Normalize densities to sum up to 1
        weights = densities / np.sum(densities)

        # Calculate weighted centroid
        weighted_lat = np.average(latitudes, weights=weights)
        weighted_lon = np.average(longitudes, weights=weights)

        return (weighted_lat, weighted_lon)

    def _prep_df_for_plotting(self) -> None:
        """This function prepares the dataframe for plotting the clusters, because plotting requires:
        a column in self.df named "id" (for the hover labels)

        Parameters:

        Returns: None

        """

        if not "id" in self.df.columns:
            self.df["id"] = self.df.index

    def plot_clusters(self, filter_noise: bool = False) -> Cluster:
        """This function creates a plotly scatter_mapbox that shows the different clusters. This requires
        the columns df.cluster and df.id to be present.

        Parameters:
            filter_noise (boolean): whether or not to filter out the noise labels ("-1") before plotting

        Returns: None

        """

        # TODO: perhaps add the area of the buildings as well via the "geojson": https://nominatim.openstreetmap.org/search?q=51.70044779735723,%205.3269376918319&polygon_geojson=1&format=json
        # possibly use osmnx for this.

        # We need a "id" column for plotting, so adding it if it doesn't exist yet.
        if not "id" in self.df.columns:
            self.df["id"] = self.df.index

        # if filter_noise:
        #     self.df = self.df[self.df.cluster != "-1"]

        # We don't always have the source value, but if we do, add it to the tooltip hover data.
        hover_data = ["timestamp"]
        if "source" in self.df.columns:
            hover_data.append("source")

        # The location labels are generally quite long, so we only take the first two elements (which is almost always the street address and house number).
        self.df["location"] = (
            self.df["location"].str.split(",").str[1]
            + ", "
            + self.df["location"].str.split(",").str[0]
        )

        self.fig = px.scatter_mapbox(
            self.df,
            lat="latitude",
            lon="longitude",
            zoom=16,
            color="location",
            hover_name="id",
            hover_data=hover_data,
            center=dict(lon=self.df_centroids.loc[0].longitude, lat=self.df_centroids.loc[0].latitude), # TODO: take centroid of biggest cluster.
        )

        # Delete the last row in df_centroids (which is the placeholder row for the noise datapoints, which was added in add_locations_to_original_dataframe())
        self.df_centroids = self.df_centroids[:-1]

        # Now we add the cluster centroids from self.df_centroids to the visualization.
        self.fig.add_trace(
            px.scatter_mapbox(
                self.df_centroids,
                lat="latitude",
                lon="longitude",
                hover_name="location",
                hover_data=["location", "num_datapoints", "cluster", "unique_days"],
                size="size",
                color_discrete_sequence=["red"],
            ).data[0]
        )

        self.fig.update_layout(mapbox_style="carto-positron")
        self.fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        # fig.show()
        self.fig.write_html(f"output/{self.outputs_folder_name}/clustermap.html")

        return self

    def add_locations_to_original_dataframe(self, export_xlsx: bool = True, name: str = "test_output") -> Cluster:
        """This function adds the results from clustering (i.e., the discrete location labels) to the original input dataframe (self.df)
        by merging self.df.centroids and self.df. Finally, it saves self.df to an xlsx file in the output folder.

        Parameters:
            None

        Returns:
            self

        """

        # Here we add one row manually since we want to keep the "noise" datapoints after the merge. These data points are now merged with the location "no cluster, noise".
        self.df_centroids.loc[len(self.df_centroids)] = [0, 0, "-1", 10, "black", "no cluster, Noise", 0, 0, 0]
        self.df_centroids.sort_values(by='num_datapoints', inplace=True, ascending=False)

        self.df = pd.merge(
            self.df, self.df_centroids[["cluster", "location"]], on="cluster"
        )
        self.df = self.df.sort_values(by="timestamp")

        if export_xlsx:
            try:
                self.df.to_excel(f"output/{self.outputs_folder_name}/{name}.xlsx")
                print(
                    f"Message (saving file): Added locations to original df, saved it as output/{self.outputs_folder_name}/{name}.xlsx"
                )
            except PermissionError:
                print("Make sure to close the Excel file if it's already opened.")

        return self
