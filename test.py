import DataLoader as DL
import DataTransformer as DT
from LoadRoutinedData import RDData
import pandas as pd
import pickle
import json
import numpy as np
from collections import defaultdict
import sys

min_n_training_days=1
max_n_training_days=21
min_n_testing_days=1
max_n_testing_days=14

with open('model_performances.pkl', 'rb') as f:
    res = pickle.load(f)

# print (json.dumps(res, indent=2, default=str))

new_res = defaultdict(dict)
for i in range(min_n_training_days, max_n_training_days +1):
    # For each training day in range [0, n_training_days] we loop over its keys and we average 
    
    # we have 14 x new_res[i], from 1 to 14, each of which contains 19 elements with the validation data

    for t_day in range(min_n_testing_days, max_n_testing_days): # loop 14 times for each testing day

        # For each testing day, we want to access the 19 elements
        accs = []
        for y in range(len(res[i])): # 19 loops
            val_data = res[i][y]
            accs.append(np.array(val_data['performance_metrics_per_day'][t_day]['acc']))

        new_res[f"training_day_{i}"][f"day_{t_day}"] = round(np.average(accs), 3)


import matplotlib.pyplot as plt

def create_heatmap(data_dict):
    # Extract training day labels
    training_days = list(data_dict.keys())
    
    # Extract day labels
    day_labels = list(data_dict[training_days[0]].keys())
    
    # Create a 2D array to store the data
    data_array = [[data_dict[training_day][day] for day in day_labels] for training_day in training_days]
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.pcolor(data_array, cmap='YlGnBu')
    
    # Add colorbar
    cbar = plt.colorbar(heatmap)
    
    # Set axis labels and title
    ax.set_xticks(range(len(day_labels)))
    ax.set_xticklabels(day_labels)
    ax.set_yticks(range(len(training_days)))
    ax.set_yticklabels(training_days)
    plt.xlabel('Days into the future')
    plt.ylabel('Number of days used for training')
    plt.title('Accuracy scores for predicting future locations')
    
    # Show the plot
    plt.show()

create_heatmap(new_res)