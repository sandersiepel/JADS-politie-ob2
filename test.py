import DataLoader as DL
import DataTransformer as DT

# Initialize parameters.
DATA_SOURCE = "google_maps"  # Can be either 'google_maps' or 'routine_d'.
# HOURS_OFFSET is used to offset the timestamps to account for timezone differences. For google maps, timestamp comes in GMT+0
# which means that we need to offset it by 2 hours to make it GMT+2 (Dutch timezone). Value must be INT!
HOURS_OFFSET = 2
# BEGIN_DATE and END_DATE are used to filter the data for your analysis.
BEGIN_DATE = "2023-01-01"
END_DATE = "2023-07-19"  # End date is EXclusive!
# FRACTION is used to make the DataFrame smaller. Final df = df * fraction. This solves memory issues, but a value of 1 is preferred.
FRACTION = 1
# For the heatmap visualization we specify a separate BEGIN_DAY and END_DAY (must be between BEGIN_DATE and END_DATE).
# For readiness purposes, it it suggested to select between 2 and 14 days.
HEATMAP_BEGIN_DATE = "2023-07-01"
HEATMAP_END_DATE = "2023-07-17"  # End date is INclusive!


# df = DL.load_data(
#     DATA_SOURCE, BEGIN_DATE, END_DATE, FRACTION, HOURS_OFFSET, verbose=True
# )

# print(df.head())

def myfunc():
    return (3, 5)

a, b = myfunc()

print(type(a), type(b))
