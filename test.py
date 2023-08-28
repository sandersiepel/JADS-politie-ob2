import pandas as pd

# from Visualisations import HeatmapVisualizer, ModelPerformanceVisualizer

# ModelPerformanceVisualizer(
# 	scores=None,
# 	pkl_file_name="model_performances"
# )
from Visualisations import HeatmapVisualizer
import pandas as pd
import json

# df = pd.read_excel("output/martijn-100-30-1/resampled_df_10_min.xlsx")
# h = HeatmapVisualizer(begin_day="2023-01-02", end_day="2023-01-10", df=df, name="heatmap", title="blabla")

df = pd.read_excel("output/martijn-100-30-1/resampled_df_10_min.xlsx", index_col=[0])

df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.dayofweek
df["hour"] = df["timestamp"].dt.hour
df["window_block"] = ((df['timestamp'].dt.minute * 60 + df['timestamp'].dt.second) // 600).astype(int)

from collections import defaultdict
res = defaultdict(dict)
final_res = defaultdict(dict)
THRESHOLD = 30
this_day_values = []

for idx, row in df.iterrows():
    m = (row['weekday'], row['hour'])

    if not 'data' in res[m]:
        res[m]['data'] = {}

    try:
        # Calculate the probability of having this location in this moment
        prob = res[m]['data'][row['location']] / sum(res[m]['data'].values())
        # print(f"For {m}, prob of finding {row['location']} in {res[m]['data']} is: {prob}")
    except KeyError:
        # The location is not in the dict, so we default to 0. This happens when the location hasn't been visited before in this moment. 
        # print(f"For {m}, couldnt find {row['location']} in {res[m]['data']}, so prob is: 0")
        prob = 0
    
    this_day_values.append(prob)

    # Now we update the res dict with data of current window block
    if row['location'] in res[m]['data']:
        res[m]['data'][row['location']] += 1
    else:
        res[m]['data'][row['location']] = 1

    # If we now exceeded THRESHOLD, remove the last entry (like a rolling window approach)
    if sum(res[m]['data'].values()) >= THRESHOLD:
        # Minus one for last entry
        res[m]['data'][res[m]['meta']['last_location']] -= 1
        # print(f"Threshold exceeded, removing 1 from {res[m]['meta']['last_location']}")

    res[m]['meta'] = {'last_location':row['location']}

    # Now we check if this window block is the last one of the day. If so, save score for this day and reset variables for next day.
    if row['hour'] == 23 and row['window_block'] == 5:
        avg = sum(this_day_values)/len(this_day_values)
        # print(f"End of the day, values for this day: {this_day_values}, which is an avg of: {avg}")
        final_res[row['timestamp'].date()] = avg
        this_day_values = []

import yaml
print(yaml.dump(res, default_flow_style=False))