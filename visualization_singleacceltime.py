# Code for visualizing SINGLE acceleration (absolute) vs. time plots for a specified data input.
# Plots different colours based on action and placement (if columns populated)

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

filename = sys.argv[1]

# read in CSV file specified when calling function
dataset = pd.read_csv(filename)

if len(dataset.columns) > 5:
    isLabeled = True
else:
    isLabeled = False

# extract columns from set (time is assumed equal intervals)
accel_x = dataset.iloc[:,1]
accel_y = dataset.iloc[:,2]
accel_z = dataset.iloc[:,3]
abs_accel = dataset.iloc[:,4]

if isLabeled:
    action = dataset.iloc[:,5]
    placement = dataset.iloc[:,6]

# initialize an array for time
# estimate time increments to be 0.01 seconds
time = [0] * len(abs_accel)
for i in range(len(abs_accel)):
    time[i] = 0.01*i
time = pd.Series(time)

# initialize figure and axes objects
fig, ax = plt.subplots(figsize=(10,5))
legends = ['Walking - Body', 'Walking - Hand', 'Jumping - Body', 'Jumping - Hand']
colours = ['blue','cyan','red','orange']


# create the plot
# for each action
if isLabeled:
    m = 0
    for j in range(2): 
        # for each placement
        for k in range(2):
            ax.scatter(time[(action==j) & (placement==k)],abs_accel[(action==j) & (placement==k)],c=colours[m], marker=".", s=2)
            m+=1
else:
    ax.scatter(time,abs_accel,c='green',marker=".", s=2)

ax.set_xlabel('Time (s)', fontsize=8)
ax.set_ylabel('Absolute Acceleration ($m/s^2$)', fontsize=8)
ax.autoscale_view(tight=None, scaley=True)
ax.set_xbound(lower=0,upper=time[len(time)-1])
file_name = os.path.basename(filename)
titlestring = 'Absolute Acceleration vs. Time, ' + os.path.splitext(file_name)[0]
ax.set_title(titlestring, fontsize=12)
if isLabeled:
    ax.legend(legends, fontsize=8)

plt.grid()
plt.show()