import DataLoader as DL
import DataTransformer as DT
from Cluster import Cluster
from MarkovChain import MarkovChain
from Visualisations import HeatmapVisualizer


# Initialize parameters.
data_source = "google_maps"  # Can be either 'google_maps' or 'routined'.
# hours_offset is used to offset the timestamps to account for timezone differences. For google maps, timestamp comes in GMT+0
# which means that we need to offset it by 2 hours to make it GMT+2 (Dutch timezone). Value must be INT!
hours_offset = 2 # Should be 0 for routined and 2 for google_maps. 
# begin_date and end_date are used to filter the data for your analysis.
begin_date = "2022-01-01"
end_date = "2022-12-30"  # End date is INclusive!
# FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
fraction = 1
# For the heatmap visualization we specify a separate begin_date and end_date (must be between begin_date and end_date).
# For readiness purposes, it it suggested to select between 2 and 14 days.
heatmap_begin_date = "2022-12-20"
heatmap_end_date = "2022-12-30"  # End date is INclusive! Choose a date that lies (preferably 2 days) before end_date to avoid errors. 


def main():
    # Main function for running our pipeline.

    # Step 1. Load data either from google maps or from routine-d data. Either way, df should contain the columns 'latitude',
    # 'longitude', 'and 'timestamp'.
    df = DL.load_data(
        data_source, begin_date, end_date, fraction, hours_offset, verbose=True,
    )

    # Step 2. Run clustering
    # First, make an instance of the Cluster class and define its settings.
    c = Cluster(
        df,  # Input dataset (with latitude, longitude, timestamp columns)
        verbose=True,  # Do we want to see print statements?
        pre_filter=True,  # Apply filters to the data before the clustering (such as removing moving points)
        post_filter=True,  # Apply filters to the data/clusters after the clustering (such as deleting homogeneous clusters)
        filter_moving=False,  # Do we want to delete the data points where the subject was moving?
        centroid_k=10,  # Number of nearest neighbors to consider for density calculation (for cluster centroids)
        min_unique_days=2,  # If post_filter = True, then delete all clusters that have been visited on less than min_unique_days days.
    )

    # Then we run the clustering and visualisation
    df = (
        c.run_clustering(
            min_samples=100,  # The number of samples in a neighborhood for a point to be considered as a core point
            eps=0.02,  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. 0.01 = 10m
            algorithm="dbscan",  # Choose either 'dbscan' or 'hdbscan'. If 'hdbscan', only min_samples is required.
            min_cluster_size=30,  # Param of HDBSCAN: the minimum size a final cluster can be. The higher this is, the bigger your clusters will be
        )
        .plot_clusters(
            filter_noise=False,  # Remove the -1 labels (i.e., noise) before plotting the clusters
            only_include_clusters=[],  # Add clusters if you want to filter which clusters to show in the visualization.
        )
        .add_locations_to_original_dataframe(
            export_xlsx=False,  # Export the dataframe to excel file? Useful for analyzing.
            name="test",
        )
        .df  # These functions return 'self' so we can chain them and easily access the df attribute.
    )

    # Step 3. Transform our labeled dataset (result of clustering) into a start- and endtime dataset for each "entry of location".
    # If fill_gaps = True, the gaps between location visits (often due to traveling) are filled with the location "unknown".
    df = DT.transform_start_end_times(df, fill_gaps=True)

    # Step 4. Make markov chain and visualize it with a graph. This represents the one-step transition probability from each state (i.e., location).
    # The graph is saved at outputs/markovchain.html. It will also open a browser, which cannot be avoided unfortunately.
    # MarkovChain(df)

    # Step 5. Transform data (resample) to 10-minute intervals (required for subsequent modeling and visualizations).
    df = DT.resample_df(df)

    # Step 6. Create and save heatmap visualization to output/heatmap.png.
    HeatmapVisualizer(
        heatmap_begin_date, heatmap_end_date, df, verbose=True, name="heatmap"
    )

    # Step 6. Train pycaret and find best model

    return None


if __name__ == "__main__":
    main()
