import pickle
import itertools
import numpy as np
import json

with open('output/model_performances.pkl', 'rb') as f:
    scores = pickle.load(f)

# print (json.dumps(scores, indent=2, default=str))

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for the heatmap
heatmap_data = []

for training_size, forecast_scores in scores.items():
    training_days = int(training_size.split("_")[-1])
    avg_values = [np.mean(score_list) for score_list in forecast_scores.values()]
    heatmap_data.append([training_days, *avg_values])

# Create a DataFrame from the data
import pandas as pd
df = pd.DataFrame(heatmap_data, columns=["Training Days", *forecast_scores.keys()])

print(df.tail())
print(df.columns)

df = df.set_index('Training Days')
sns.heatmap(df.T.round(3), cmap="Blues")
plt.show()