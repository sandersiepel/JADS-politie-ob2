from LoadGoogleMaps import GMData
from LoadRoutinedData import RDData
from datetime import datetime
import pandas as pd


def load_data(data_source, begin_date, end_date, fraction, hours_offset, verbose=True):
    if data_source == "google_maps":
        df = GMData().df
    elif data_source == "routine_d":
        df = RDData().df
    else:
        raise ValueError(
            'Invalid input for DATA_SOURCE. Make sure that the value of DATA_SOURCE is either "google_maps" or "routine_d".'
        )

    df = filter_data(df, begin_date, end_date, fraction)

    if hours_offset > 0:
        if verbose:
            print(
                f"Message (data loader): Since HOUR_OFFSET > 0, we offset the timestamps with {hours_offset} hours."
            )
        df.timestamp = df.timestamp + pd.Timedelta(hours=int(hours_offset))

    if verbose:
        print(
            f"Message (data loader): Loaded {data_source} data from {begin_date} to {end_date} with a fraction of {fraction}. Length of data: {len(df)}"
        )

        print(
            f"Message (data loader): First record in dataset is from {str(df.iloc[0].timestamp).split('.')[0]} and last record is from {str(df.iloc[-1].timestamp).split('.')[0]}"
        )

    return df


def filter_data(df, begin_date, end_date, fraction):
    return downsample(
        df[(df["timestamp"] > begin_date) & (df["timestamp"] <= end_date)],
        # This is the fraction that is used in the sample function. Fraction of 0.1 will return 10% of the original df.
        fraction,
    ).sort_values(by="timestamp")


def downsample(df, fraction):
    # If you want to downsample the dataframe (because of memory issues), use this function.
    # A fraction of 0.5 will result in a dataframe half the size.
    if fraction < 1:
        return df.sample(frac=fraction)
    else:
        return df
