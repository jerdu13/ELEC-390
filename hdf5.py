import h5py
import numpy as np
import nexusformat.nexus as nx

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

# transfer numpy array to Ryan group
RM_dset = h5RM.create_dataset('ryan-dataset', data=RM_csv)


# print the hdf5 organization nicely
f = nx.nxload('elec390.h5')
print(f.tree)

h5file.close()
