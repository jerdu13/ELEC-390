import csv 
import matplotlib.pyplot as plt
import numpy as np

# List of filenames to read
filenames = ['Louie\'s-Dataset/combined_walking.csv', 'Louie\'s-Dataset/combined-jumping.csv', 'ryans-dataset/walking-data/walking-combined.csv','ryans-dataset/jumping-data/jumping-combined.csv', 'jeremys-dataset/jeremy-walking-combined.csv', 'jeremys-dataset/jeremy-jumping-combined.csv']  # Add the additional filenames to the list

# Initialize list for storing max absolute accelerations
max_accels = []

# Loop over each file
for filename in filenames:
    abs_accels = []  # Initialize list of absolute accelerations for this file
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            accel_str = row[4].strip()
            if accel_str:
                accel = abs(float(accel_str))
                abs_accels.append(accel)
        max_accels.append(max(abs_accels))

# Set up the bar graph
index = np.arange(len(filenames))
bar_width = 0.35
opacity = 0.8

# Abbreviated names for x-axis labels
abbreviated_names = ['Louie\'s walking dataset', 'Louie\'s jumping dataset', 'ryans walking dataset', 'ryans jumping dataset', 'jeremys walking dataset', 'jeremys jumping dataset']  # Update the abbreviated names

fig, ax = plt.subplots()
rects1 = ax.bar(index, max_accels, bar_width, alpha=opacity, color='b', label='Max Acceleration')

ax.set_xlabel('File')
ax.set_ylabel('Acceleration (m/s^2)')
ax.set_title('Max Absolute Accelerations')
ax.set_xticks(index)
ax.set_xticklabels(abbreviated_names, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()
