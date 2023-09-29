import pandas as pd


def transform_start_end_times(df, outputs_folder_name:str, fill_gaps: bool = False) -> pd.DataFrame:
    # First we select the timestamp, location and cluster columns.
    df = df[["timestamp", "location", "cluster"]].copy()

    df["location_id"] = pd.factorize(df["location"])[0]
    df["group"] = (df["location"] != df["location"].shift()).cumsum()

    df = (
        df.groupby(["group", "location", "location_id"])
        .agg({"timestamp": ["min", "max"]})
        .reset_index()
    )
    df.columns = ["group", "location", "location_id", "begin_time", "end_time"]

    # Drop the 'group' column
    df = df.drop(columns="group")

    # Change the location labels to only include address.
    # df["location"] = (
    #     df["location"].str.split(",").str[1]
    #     + ", "
    #     + df["location"].str.split(",").str[0]
    # )

    if fill_gaps:
        df = fill_start_end_time_gaps(df)

    # df should now have the columns: location (string label), location_id (unique id for each unique location), begin_time and end_time.
    df.to_excel(f"output/{outputs_folder_name}/start_end_time_df.xlsx")

    print(
        f"Message (data transformer): First record in dataset starts at {str(df.iloc[0].begin_time)} and last record ends at {str(df.iloc[-1].end_time)}"
    )

    return df


def fill_start_end_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    def _create_new_rows(df: pd.DataFrame) -> list:
        new_rows = []
        for i in range(len(df) - 1):
            if df.iloc[i]["end_time"] != df.iloc[i + 1]["begin_time"]:
                new_row = {
                    "location": df.iloc[i]["location"], # Location can also be set to "unknown".
                    "begin_time": df.iloc[i]["end_time"] + pd.Timedelta(seconds=1), # We add one second to the begin and end times aren't overlapping
                    "end_time": df.iloc[i + 1]["begin_time"] - pd.Timedelta(seconds=1),
                    "location_id": -1,
                }
                new_rows.append(new_row)
        return new_rows

    # apply the function to the dataframe
    new_rows = pd.DataFrame(_create_new_rows(df))

    # Concatenate the new rows with the original dataframe
    df = pd.concat([df, new_rows], ignore_index=True)

    print(f"Message (resampling, filling gaps): Filled {len(new_rows)} gaps.")

    # Sort the dataframe by begin_time, reset index so it becomes only increasing from 0 again.
    return df.sort_values("begin_time").reset_index(drop=True)


def resample_df(df: pd.DataFrame, outputs_folder_name:str) -> pd.DataFrame:
    if not set(["begin_time", "end_time", "location_id"]).issubset(set(df.columns)):
        raise ValueError(
            "Make sure that df contains columns 'begin_time', 'end_time' and location_id'."
        )
    
    # In our dataset we have data points labeled as noise. Remove them and replace by last known (ie valid) value.
    # Replace 'none' with NaN to make use of fillna method
    df['location'] = df['location'].replace(' no cluster,  Noise', "Unknown") # pd.NA

    # Forward fill missing values
    # df['location'].ffill(inplace=True)

    # V2 of the code to also include the time intervals until the end_date of the last row:
    df = df.set_index('begin_time')

    # Create a new DataFrame with a continuous time range covering the desired period
    time_range = pd.date_range(start=df.index.min().normalize(), end=df['end_time'].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), freq='10T')

    # Merge the original DataFrame with the continuous time range using merge_asof
    resampled_df = pd.merge_asof(pd.DataFrame(index=time_range), df.reset_index(), left_index=True, right_on='begin_time', direction='backward')

    df = (
        resampled_df.reset_index()
        .drop(["begin_time", "location_id", "end_time"], axis=1)
        .rename(columns={"index": "timestamp"})
        .dropna(axis=0)
    )

    df.to_excel(f"output/{outputs_folder_name}/resampled_df_10_min.xlsx")

    return df

def add_temporal_features(df: pd.DataFrame):
    # Input: df with a 'timestamp' column and a location column. 
    
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["window_block"] = ((df['timestamp'].dt.minute * 60 + df['timestamp'].dt.second) // 600).astype(int)

    return df