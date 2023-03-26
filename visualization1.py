# Code for visualizing single accel vs. time plots for a specified data input.
# STILL WIP - want to make more modular so we can use any file as input, and auto set titles

import h5py
import matplotlib.pyplot as plt
import pandas as pd

# for csv file importing (okay for visualization purposes?)
dataset = pd.read_csv('ryans-dataset\\jumping-data\\jumping-combined.csv')

# # for hdf5 file importing
# h5file = h5py.File('elec390.h5', 'r')

# print(h5file.keys())

# group = h5file['Ryan']

# dset = group['ryan-dataset']

# num = (dset[2, 5])

# extract columns from set (time is assumed equal intervals)
accel_x = dataset.iloc[:,1]
accel_y = dataset.iloc[:,2]
accel_z = dataset.iloc[:,3]
abs_accel = dataset.iloc[:,4]
action = dataset.iloc[:,5]
placement = dataset.iloc[:,6]

# initialize an array for time
time = [0] * len(accel_x)
for i in range(len(accel_x)):
    time[i] = 0.01*i
time = pd.Series(time)

fig, ax = plt.subplots(figsize=(10,5))
colours = ['blue','green']
legends = ['Body', 'Hand']

# creat the plot
for j in range(len(legends)):
    ax.plot(time[placement==j],accel_y[placement==j],c=colours[j], linewidth=1)

ax.set_xlabel('Time (s)', fontsize=8)
ax.set_ylabel('Acceleration in y ($m/s^2$)', fontsize=8)
ax.set_xbound(lower=0,upper=time[len(time)-1])
ax.set_title('Acceleration in y vs. Time, Ryan Jumping Data', fontsize=12)
ax.legend(legends, fontsize=8)

plt.grid()
plt.show()