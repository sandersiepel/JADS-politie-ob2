import pandas as pd

# Sample DataFrame
data = {'time': ['2023-01-01 03:20:00', '2023-01-01 03:30:00', '2023-01-01 03:40:00', '2023-01-01 03:50:00',
                 '2023-01-01 04:00:00', '2023-01-01 04:10:00', '2023-01-01 04:20:00', '2023-01-01 04:30:00',
                 '2023-01-01 04:40:00', '2023-01-01 04:50:00', '2023-01-01 05:00:00'],
        'location': ['A', 'A', 'B', 'C', 'A', 'A', 'A', 'B', 'B', 'B', 'B']}

df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'])
df.sort_values('time', inplace=True)

# 1. Feature Engineering
df['hour_of_day'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['window_block'] = (df['time'].dt.hour * 60 + df['time'].dt.minute) // 20

# Calculate the most common location for each combination of hour_of_day, day_of_week, and window_block
most_common_locations = df.groupby(['hour_of_day', 'day_of_week', 'window_block'])['location'].apply(lambda x: x.value_counts().idxmax()).reset_index()

result_df = pd.merge(df,most_common_locations, on=['hour_of_day', 'day_of_week', 'window_block'], how='left')

print(result_df)