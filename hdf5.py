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

f = nx.nxload('elec390.h5')
print(f.tree)

h5file.close()
