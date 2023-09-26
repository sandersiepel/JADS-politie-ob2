import pandas as pd
import numpy as np

df_raw = pd.read_excel("output/politiedemo-100-30-1/resampled_df_10_min.xlsx", index_col=[0])
start_date = pd.to_datetime(f"2023-05-10 00:00:00")
end_date = pd.to_datetime(f"2023-05-14 23:50:00")

df = df_raw[df_raw["timestamp"].between(start_date, end_date)]

# self.original_data = self.df.location.values.reshape(
#             self.n_days, self.n_per_day
#         )  # Keep version of original dataset

print(df.head())

data = np.reshape(df.location.values.tolist(), (5, 144))


# import plotly.express as px
# fig = px.imshow(data,
#                 labels=dict(x="Time", y="Day", color="Productivity"),
#                 x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
#                 y=['Morning', 'Afternoon', 'Evening']
#                )
# fig.update_xaxes(side="top")
# fig.show()

# x_ticks = np.arange(0, 144, (144 / 24)) 
# print(x_ticks)
# x_tick_labels = [f"{i:02d}:{j:02d}" for i in range(0, 24) for j in range(0, 60, 60)]
# print(x_tick_labels)

import xarray as xr