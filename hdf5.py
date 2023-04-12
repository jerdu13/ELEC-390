import h5py
import numpy as np
import nexusformat.nexus as nx
import pandas as pd

h5file = h5py.File('elec390.h5', 'w')

h5JD = h5file.create_group('Jeremy')
h5RM = h5file.create_group('Ryan')
h5LT = h5file.create_group('Louie')

h5shuffled = h5file.create_group('Dataset')
h5shuffled_train = h5shuffled.create_group('Train') # Sub-groups
h5shuffled_test = h5shuffled.create_group('Test')

# Create a dataset: ex_dset = h5group.create_dataset(name, shape, dtype, chunks, compression = 'gzip', scaleoffset = True, shuffle = True)

# load in Ryan csv to numpy array (all are floats)
RM_csv = np.genfromtxt('ryans-dataset\\ryan-data-combined.csv', dtype=float, delimiter=',', skip_header=1)
RM_csv2 = np.genfromtxt('ryans-dataset\\walking-data\\walking-combined.csv', dtype=float, delimiter=',', skip_header=1)
# transfer numpy array to Ryan group
RM_dset = h5RM.create_dataset('ryan-dataset', data=RM_csv)
RM_dset2 = h5RM.create_dataset('ryan-dataset2', data=RM_csv2)


RM_data = np.array(RM_dset[1,:])
RM_data_combined = np.append(RM_data, RM_dset2[1, :])



RM_dset_shuffled = h5RM.create_dataset(name="RM_Data_shuffled", data=RM_data_combined)

# train and test split
jd = "jeremy-data-combined.csv"

dset = pd.read_csv(jd)

sampling_rate = 100
window_size = 5*sampling_rate
num_rows = dset.shape[0]
ratio = 0.9

num_windows = int(num_rows//window_size)
segments = np.zeros((num_windows, window_size, dset.shape[1]))

for i in range(num_windows):
    segments[i] = dset.iloc[i * window_size:(i + 1) * window_size].values

np.random.shuffle(segments)
train_size = int(ratio*num_windows)
train_data = segments[:train_size] # 90%
test_data = segments[train_size:]  # 10%

train_dset = h5shuffled_train.create_dataset('train', data=train_data)
test_dset = h5shuffled_test.create_dataset('test', data=test_data)

# print the hdf5 organization nicely
f = nx.nxload('elec390.h5')
print(f.tree)

h5file.close()
