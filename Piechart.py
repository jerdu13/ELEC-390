import pandas as pd
import matplotlib.pyplot as plt

# Combine walking data
walking_data_1 = pd.read_csv('ryans-dataset/walking-data/walking-combined.csv')
walking_data_2 = pd.read_csv('Louie\'s-Dataset/combined_walking.csv')
# walking_data_3 = pd.read_csv('walking_data_3.csv')
walking_data = pd.concat([walking_data_1, walking_data_2], ignore_index=True)
walking_data.to_csv('walking_data.csv', index=False)

# Combine jump data
jump_data_1 = pd.read_csv('ryans-dataset/jumping-data/jumping-combined.csv')
jump_data_2 = pd.read_csv('Louie\'s-Dataset/combined-jumping.csv')
# jump_data_3 = pd.read_csv('jump_data_3.csv')
jump_data = pd.concat([jump_data_1, jump_data_2], ignore_index=True)
jump_data.to_csv('jump_data.csv', index=False)

# Plot pie chart for walking and jump data
walking_count = len(walking_data)
jump_count = len(jump_data)
total_count = walking_count + jump_count
walking_percent = (walking_count / total_count) * 100
jump_percent = (jump_count / total_count) * 100
labels = ['Walking', 'Jumping']
sizes = [walking_percent, jump_percent]
colors = ['green', 'blue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Walking vs Jumping Data')
plt.show()