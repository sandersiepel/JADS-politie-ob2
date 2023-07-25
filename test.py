import DataLoader as DL
import DataTransformer as DT
from LoadRoutinedData import RDData
import pandas as pd

# Initialize parameters.
data_source = "routined"  # Can be either 'google_maps' or 'routined'.
# HOURS_OFFSET is used to offset the timestamps to account for timezone differences. For google maps, timestamp comes in GMT+0
# which means that we need to offset it by 2 hours to make it GMT+2 (Dutch timezone). Value must be INT!
hours_offset = 2
# BEGIN_DATE and END_DATE are used to filter the data for your analysis.
begin_date = "2023-05-10"
end_date = "2023-07-24"  # End date is EXclusive!
# FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
fraction = 1
# For the heatmap visualization we specify a separate BEGIN_DAY and END_DAY (must be between BEGIN_DATE and END_DATE).
# For readiness purposes, it it suggested to select between 2 and 14 days.
heatmap_begin_date = "2023-07-10"
heatmap_end_date = "2023-07-20"  # End date is INclusive!



def resample_df(df: pd.DataFrame) -> pd.DataFrame:
    if not set(["begin_time", "end_time", "location_id"]).issubset(set(df.columns)):
        raise ValueError(
            "Make sure that df contains columns 'begin_time', 'end_time' and location_id'."
        )

    # V2 of the code to also include the time intervals until the end_date of the last row:
    df = df.set_index('begin_time')

    # Create a new DataFrame with a continuous time range covering the desired period
    time_range = pd.date_range(start=df.index.min().normalize(), end=df['end_time'].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), freq='10T')

    # Merge the original DataFrame with the continuous time range using merge_asof
    resampled_df = pd.merge_asof(pd.DataFrame(index=time_range), df.reset_index(), left_index=True, right_on='begin_time', direction='backward')

    df = (
        resampled_df.reset_index()
        .drop(["begin_time", "location_id", "end_time"], axis=1)
        .rename(columns={"index": "time"})
        .dropna(axis=0)
    )

    df.to_excel("output/resampled_df_10_min.xlsx")

    return df

df = pd.read_excel("output/start_end_time_df.xlsx", index_col=[0])
print(f"Df start: \n{df.head(2)}, df end: \n{df.tail(2)} \n\n")

df = resample_df(df)
print(f"Df start: \n{df.head(2)}, df end: \n{df.tail(2)}")

