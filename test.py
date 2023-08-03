import pickle
import json

with open('output/model_performances.pkl', 'rb') as f:
    scores = pickle.load(f)

# print (json.dumps(scores, indent=2, default=str))

import matplotlib.pyplot as plt

# Extract data from the scores dictionary
training_sizes = []
performance_scores = []

for training_size, forecast_scores in scores.items():
    training_sizes.append(int(training_size.split("_")[-1]))
    performance_scores.append(sum(sum(score_list) / len(score_list) for score_list in forecast_scores.values()) / len(forecast_scores))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, performance_scores, marker='o')
plt.title("Performance vs. Number of Training Days")
plt.xlabel("Number of Training Days")
plt.ylabel("Performance Score")
plt.grid(True)
plt.xticks(training_sizes)
plt.tight_layout()

plt.show()