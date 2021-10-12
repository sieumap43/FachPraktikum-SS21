import os
import glob
import shutil
import subprocess
import numpy as np
import h5py

my_src_dir = os.getcwd()

root_dir = os.path.abspath('../')
src_dir = os.path.abspath('../src')

# crop eyes and create a h5 file from the original columbia images
subprocess.run(['python', 'columbia_crop_eyes.py',
    '--src_dir', 'datasets/columbia_original_face', '--dst_file', 'datasets/columbia_exp_11.h5',
    '--width', '150', '--height', '90', '--channel', 'color',
    '--scale', '0.5', '--flip_right', 'True'])

# train
os.chdir(src_dir)
subprocess.run(['python', 'dpg_train_exp_11.py',
    '--log_dir', 'outputs_exp_11', '--epoch', '50',
    '--train_path', '../datasets/columbia_exp_11_0.h5', '--test_path', '../datasets/columbia_exp_11_0.h5'])
