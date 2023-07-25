import pandas as pd


def transform_start_end_times(df, fill_gaps=False):
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
    df["location"] = (
        df["location"].str.split(",").str[1]
        + ", "
        + df["location"].str.split(",").str[0]
    )

    if fill_gaps:
        df = fill_start_end_time_gaps(df)

    # df should now have the columns: location (string label), location_id (unique id for each unique location), begin_time and end_time.
    df.to_excel("output/start_end_time_df.xlsx")

    return df


def fill_start_end_time_gaps(df):
    def _create_new_rows(df):
        new_rows = []
        for i in range(len(df) - 1):
            if df.iloc[i]["end_time"] != df.iloc[i + 1]["begin_time"]:
                new_row = {
                    "location": "unknown",
                    "begin_time": df.iloc[i]["end_time"] + pd.Timedelta(seconds=1),
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


def resample_df(df):
    if not set(["begin_time", "end_time", "location_id"]).issubset(set(df.columns)):
        raise ValueError(
            "Make sure that df contains columns 'begin_time', 'end_time' and location_id'."
        )

    df = df.set_index("begin_time")
    df = df.resample("10T").ffill()
    df = (
        df.reset_index()
        .drop(["location_id", "end_time"], axis=1)
        .rename(columns={"begin_time": "time"})
        .dropna(axis=0)
    )

    df.to_excel("output/resampled_df_10_min.xlsx")

    return df
