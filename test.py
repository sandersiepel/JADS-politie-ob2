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
print(df.timestamp.min().date())