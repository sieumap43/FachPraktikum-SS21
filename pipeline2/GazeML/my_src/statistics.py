import cv2 as cv
from glob import glob
import os
import h5py

f = h5py.File('../datasets/MPIIGaze.h5', 'r')
print("f.keys()", f.keys())
print("f['train'].keys()", f['train'].keys())
print("f['train']['p00'].keys()", f['train']['p00'].keys())

# statistics
train = f['train']
n_train = 0

for key in train.keys():
    n_train += train['p00']['eye'].shape[0]
print("Number of instances:", n_train)