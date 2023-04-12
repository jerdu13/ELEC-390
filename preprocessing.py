# Script to preprocess data and export to new csv files
# This will be the step done between raw data collection (the input) and shuffling/splitting

import sys
import os
import pandas as pd
from sklearn import preprocessing

filename = sys.argv[1]

raw_dataset = pd.read_csv(filename)

# Fill in NaN datapoints if there are any
if raw_dataset.isna().sum().sum():
    print('Dataset input contains missing (NaN) value(s). \nCorrection will be done using linear interpolation.')
    # linearly interpolate missing data
    raw_dataset = raw_dataset.interpolate(method='linear')
else:
    print('Dataset input contains no missing (NaN) values. No correction required.')

# separate data and labels - don't want to normalize labels
raw_dataset_labels = raw_dataset.iloc[:,5:]
raw_dataset_data = raw_dataset.iloc[:, 1:5]
raw_dataset_times = raw_dataset.iloc[:, 0]

# reduce noise using moving average filter
window_size = 31
norm_dataset_data = raw_dataset_data.rolling(window_size).mean()

# adjust the rows of the dataset (both data and labels) to drop (window_size - 1) rows at beginning
# must do this since rolling av filter removes these rows
norm_dataset_data = norm_dataset_data.truncate(before=(window_size-1))
raw_dataset_labels = raw_dataset_labels.truncate(before=(window_size-1))
raw_dataset_times = raw_dataset_times.truncate(before=(window_size-1))


### export data at current step for feature extraction (Step 5) ###
pieces = [raw_dataset_times, norm_dataset_data, raw_dataset_labels]
norm_dataset = pd.concat(pieces, axis=1)

new_path = 'feature-extraction-ready\\normalized-' + filename.rsplit('\\', 1)[-1]
norm_dataset.to_csv(new_path, index=False)

print('Data normalized and exported to feature-extraction-ready folder.')
###

# detect and remove outliers



# normalize



