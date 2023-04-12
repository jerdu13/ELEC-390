# Code for visualizing two absolute acceleration vs. time plots on one figure
# user specifies both input files when calling script


import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

first_filename = sys.argv[1]
second_filename = sys.argv[2]

# read in CSV file specified when calling function
first_dataset = pd.read_csv(first_filename)
second_dataset = pd.read_csv(second_filename)


# extract columns from first set (time is assumed equal intervals)
first_abs_accel = first_dataset.iloc[:,4]
# extract columns from second set
second_abs_accel = second_dataset.iloc[:,4]

# initialize an array for time
# estimate time increments to be 0.01 seconds
time1 = [0] * len(first_abs_accel)
for i in range(len(first_abs_accel)):
    time1[i] = 0.01*i
time1 = pd.Series(time1)

time2 = [0] * len(second_abs_accel)
for i in range(len(second_abs_accel)):
    time2[i] = 0.01*i
time2 = pd.Series(time2)


# initialize figure and axes objects
fig, ax = plt.subplots(figsize=(10,5))
colours = ['blue','red']


# create the plot for first dataset
ax.plot(time1,first_abs_accel,c=colours[0], linewidth=1, alpha=0.5)

# create the plot for second dataset
ax.plot(time2,second_abs_accel,c=colours[1], linewidth=1,alpha=0.5)

ax.set_xlabel('Time (s)', fontsize=8)
ax.set_ylabel('Absolute Acceleration ($m/s^2$)', fontsize=8)
ax.autoscale_view(tight=None, scaley=True, scalex=True)
ax.set_xbound(lower=0)
f_file_name = os.path.basename(first_filename)
s_file_name = os.path.basename(second_filename)
titlestring = 'Acceleration vs. Time comparison: ' + os.path.splitext(f_file_name)[0] + ' and ' + os.path.splitext(s_file_name)[0]
ax.set_title(titlestring, fontsize=10)
legends = [os.path.splitext(f_file_name)[0], os.path.splitext(s_file_name)[0]]
ax.legend(legends, fontsize=8)

plt.grid()
plt.show()