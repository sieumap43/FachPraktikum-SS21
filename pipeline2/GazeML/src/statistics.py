"""
Author: An Trieu
Created: 17.07.2021
"""
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import numpy as np
import os
import util.gazemap as gm
import h5py

f = h5py.File('../datasets/MPIIGaze.h5', 'r')

num_train = 0
num_test = 0

for key in f['train'].keys():
    num_train += f['train'][key]['eye'].shape[0]

for key in f['test'].keys():
    num_test += f['test'][key]['eye'].shape[0]

print("Numeber of training instances", num_train)
print("Number of test instances: ", num_test)
print('Eye shape: ', f['train']['p00']['eye'][0].shape)

train_gazes = f['train']['p00']['gaze']

fig, axes = plt.subplots(nrows=2)
axes[0].hist(train_gazes[:, 0])
axes[0].set_xlabel('radians')
axes[0].set_title('histogram of vertical angles of p00')
axes[1].hist(train_gazes[:,1])
axes[1].set_xlabel('radians')
axes[1].set_title('histogram of horizontal angles of p00')

plt.tight_layout()
fig.show()
# f.close()
