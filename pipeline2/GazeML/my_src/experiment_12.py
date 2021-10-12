import os
import glob
import shutil
import subprocess
import numpy as np
import h5py

my_src_dir = os.getcwd()

os.chdir('../')
root_dir = os.getcwd()
src_dir = os.path.abspath('src/')

# copy all files within current directory to a destination directory
def copy_all(ext='.jpg', dst_dir=''):
    files = glob.glob('*' + ext)
    for f in files:
        shutil.copy(f, dst_dir)

# create h5 files (one h5 file for each fold)
from math import floor
n_folds = 5
fold_size = floor(56 / n_folds)

os.chdir('datasets/')
datasets_dir = os.getcwd()
original_dir = os.path.abspath('columbia_gr')

data_dir = 'columbia_gr'
data_file = 'columbia_exp_12.h5'

# convert to h5
os.chdir(my_src_dir)
subprocess.run(['python', 'convert_img2h5.py', '--src_dir', data_dir,
    '--dst_file', data_file, '--ids', '0'])
    
# train
os.chdir(src_dir)
subprocess.run(['python', 'dpg_train_exp_12.py',
        '--oh' , '64', '--ow', '64',
        '--epoch', '50', '--log_dir', 'outputs_exp_12',
            '--train_path', '../datasets/columbia_exp_12_0.h5', '--test_path', '../datasets/columbia_exp_12_0.h5'])