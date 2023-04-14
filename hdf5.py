# ELEC 390 Project - Group 48 - Step 2: Data Storing
# Code to store individual raw datasets in their respective groups
# Also includes lines from 'train-test.py', for 

import h5py
import numpy as np
import nexusformat.nexus as nx
import pandas as pd

# create a new HDF5 file
h5file = h5py.File('elec390.h5', 'w')

# create groups under the root directory
h5JD = h5file.create_group('Jeremy')
h5RM = h5file.create_group('Ryan')
h5LT = h5file.create_group('Louie')

h5shuffled = h5file.create_group('Dataset')
h5shuffled_train = h5shuffled.create_group('Train') # Sub-groups
h5shuffled_test = h5shuffled.create_group('Test')

# create a dataset: ex_dset = h5group.create_dataset(name, shape, dtype, chunks, compression = 'gzip', scaleoffset = True, shuffle = True)

# load in individual csvs to numpy array (all are floats)
# headers cannot be stored to HDF5 file
RM_csv = np.genfromtxt('ryans-dataset\\ryan-data-combined.csv', dtype=float, delimiter=',', skip_header=1)
JD_csv = np.genfromtxt('jeremys-dataset\\jeremy-data-combined.csv', dtype=float, delimiter=',', skip_header=1)
LT_csv = np.genfromtxt('louies-dataset\\louie-data-combined.csv', dtype=float, delimiter=',', skip_header=1)


# transfer numpy arrays to groups
RM_dset = h5RM.create_dataset('ryan-dataset', data=RM_csv)
JD_dset = h5JD.create_dataset('jeremy-dataset', data=JD_csv)
LT_dset = h5LT.create_dataset('louie-dataset', data=LT_csv)


# # train and test splitting and shuffling - excerpt from train-test.py
dset1 = pd.read_csv("shuffling-splitting-ready\\normalized-jeremy-data-combined.csv")
dset2 = pd.read_csv("shuffling-splitting-ready\\normalized-louie-data-combined.csv")
dset3 = pd.read_csv("shuffling-splitting-ready\\normalized-ryan-data-combined.csv")

pieces = [dset1, dset2, dset3]
dset = pd.concat(pieces)

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
np.random.seed(42)
np.random.shuffle(segments)

train_size = int(num_windows*ratio)

train_data = segments[:train_size]  # first 90% for train
test_data = segments[train_size:]   # remaining 10% for test

# store train and test data in appropriate locations
train_dset = h5shuffled_train.create_dataset('train-dataset', data=train_data)
test_dset = h5shuffled_test.create_dataset('test-dataset', data=test_data)

# print the hdf5 organization nicely usine Nexus
f = nx.nxload('elec390.h5')
print(f.tree)

h5file.close()
