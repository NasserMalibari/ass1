import scipy.io as sio
from os.path import dirname, join as pjoin
import os
import numpy as np


full_data = []
i = 0
# loop through all test files:
data_dir = pjoin(os.getcwd(), 'train', 'train')
for filename in os.listdir(data_dir):

    # uncomment if you only want a subset of all iterations
    # if i == 25:
    #     break
    
    i += 1

    mat_fname = pjoin(data_dir, filename)

    y = mat_fname[-5]

    # load in .mat file
    mat_contents = sio.loadmat(mat_fname)
    data = mat_contents["CSI"]
    
    # find t
    t = len(data[:,0,0,0])

    # flatten
    data = data.reshape(t, 30, 18)

    # split into t sub-arrays
    data_split = [data[i,:,:] for i in range(t)]
    
    data_and_target = np.array([data_split, y], dtype = object)
    full_data.append(data_and_target)

full_data = np.array(full_data, dtype = object)

