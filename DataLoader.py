from LoadGoogleMaps import GMData
from LoadRoutinedData import RDData
import pandas as pd
from Visualisations import EDA
import os

def load_data(data_source: str, begin_date: str, end_date: str, fraction: float, hours_offset: int, outputs_folder_name:str, verbose: bool = True, perform_eda: bool = True) -> pd.DataFrame:
    # Create output folder, if necessary.
    os.makedirs(f"output/{outputs_folder_name}", exist_ok=True)
    
    if data_source == "google_maps":
        df = GMData().df
    elif data_source == "routined":
        df = RDData().df
    else:
        raise ValueError(
            'Invalid input for DATA_SOURCE. Make sure that the value of DATA_SOURCE is either "google_maps" or "routine_d".'
        )

    if hours_offset > 0:
        if verbose:
            print(
                f"Message (data loader): Since HOUR_OFFSET > 0, we offset the timestamps with {hours_offset} hours."
            )
        df.timestamp = df.timestamp + pd.Timedelta(hours=int(hours_offset))

    df = filter_data(df, begin_date, end_date, fraction)

    if verbose:
        print(
            f"Message (data loader): Loaded {data_source} data from {begin_date} to {end_date} with a fraction of {fraction}. Length of data: {len(df)}"
        )

        print(
            f"Message (data loader): First record in dataset is from {str(df.iloc[0].timestamp).split('.')[0]} and last record is from {str(df.iloc[-1].timestamp).split('.')[0]}"
        )

    if perform_eda:
        # Perform EDA
        if verbose:
            print(f"Message (data loader): Performing EDA, saving plots at output/{outputs_folder_name}")

        e = EDA(data=df, outputs_folder_name=outputs_folder_name)
        e.records_per_day()

    return df, e.fig


def filter_data(df: bool, begin_date: str, end_date: str, fraction: float) -> pd.DataFrame:
    # First, set the timestamp column as index.
    df = df.set_index('timestamp')

    # Then we use .loc to filter the dataset between two dates. This is INclusive!
    df = df.loc[begin_date:end_date]

    # Then make timestamp column instead of index
    df = df.reset_index(names=["timestamp"])

    return downsample(
        df, fraction, # This is the fraction that is used in the sample function. Fraction of 0.1 will return 10% of the original df.
    ).sort_values(by="timestamp")


def downsample(df: bool, fraction: float) -> pd.DataFrame:
    # If you want to downsample the dataframe (because of memory issues), use this function.
    # A fraction of 0.5 will result in a dataframe half the size.
    if fraction < 1:
        return df.sample(frac=fraction)
    else:
        return df
