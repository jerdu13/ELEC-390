import numpy as np
import pandas as pd

jd = "jeremy-data-combined.csv"

dset = pd.read_csv(jd)

sampling_rate = 100     # 100 data points collected each second by Phyphox
window_size = 5*sampling_rate
ratio = 0.9

rows = dset.shape[0] # number of rows in combined csv
num_windows = int(rows/window_size)

# initialize np array for segmented data
segments = np.zeros((num_windows, window_size, dset.shape[1])) # 3-D array. Can be thought of as an array of windows, where each window has 100 rows and 5 columns

for i in range(num_windows):
    segments[i] = dset.iloc[i * window_size:(i + 1) * window_size].values

# shuffle segmented data
np.random.shuffle(segments)

train_size = int(num_windows*ratio)

train_data = segments[:train_size]  # first 90% for train
test_data = segments[train_size:]   # remaining 10% for test
