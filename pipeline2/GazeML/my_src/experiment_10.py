"""
Author: An Trieu
Date: 20.08.2021
"""

import os
import glob
import shutil
import subprocess
import numpy as np
import h5py

################################################################
# Metadata
################################################################
# function to remove existing directory
def remove_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)

os.chdir('../../')
root_dir = os.getcwd()
gr_dir = os.path.join(root_dir, 'gaze_redirection') # gaze redirector dir
ge_dir = os.path.join(root_dir, 'GazeML') # gaze estimator dir


# k-fold cross-validation params
from math import floor
n_folds = 5
fold_size = floor(56 / n_folds)

################################################################
# Train Gaze Redirector
################################################################
os.chdir(gr_dir)
for fold_id in range(n_folds):
    log_dir = os.path.join(gr_dir, 'log_%d' % (fold_id))

    # train
    subprocess.run(['python', 'main.py', '--mode', 'train', '--data_path', './dataset/all/', '--log_dir', log_dir,
        '--batch_size', '32', '--vgg_path', './vgg_16.ckpt', '--ids', str(fold_id)])

    # redirect
    subprocess.run(['python', 'main.py', '--mode', 'eval', '--data_path', './dataset/0P', '--log_dir', log_dir,
        '--batch_size', '64', '--ids', str(fold_id)])

    # copy generated images
    gen_dir = os.path.join(log_dir, 'eval', 'genes_%d' % (fold_id))
    dst_gen_dir = os.path.join(ge_dir, 'datasets')
    if not(os.path.exists(os.path.join(dst_gen_dir, 'genes_%d' % (fold_id)))):
        shutil.move(gen_dir, dst_gen_dir)

# copy original images
dst_orig_dir = os.path.join(ge_dir, 'datasets', 'all')
remove_if_exists(dst_orig_dir)

os.chdir(os.path.join(gr_dir, 'dataset'))
print("Number of copied original images:", len(os.listdir('all')))
shutil.copytree('all', dst_orig_dir)

os.chdir(os.path.join(ge_dir, 'datasets'))
remove_if_exists('columbia_gr')
os.rename('all', 'columbia_gr')

# flip the horizontal angle of right eyes
os.chdir('columbia_gr')
img_names = glob('*.jpg')
n = 0 # sequence number to prevent deleting duplicates
for img_name in img_names:
    if 'R' in img_name:
        img_label = img_name.split('_')
        # print(img_label)
        for i, label in enumerate(img_label):
            if 'H' in label:
                yaw = int(label.strip('H'))
                yaw *= -1
                img_label[i] = '%dH' %(yaw)
            if '.jpg' in label:
                new_label = label.strip('.jpg') + '_%d.jpg' %(n)
                img_label[i] = new_label
                n += 1
        os.rename(img_name, '_'.join(img_label))
print("Number of received original images:", len(os.listdir()))


################################################################
# Train Gaze Estimator
################################################################

my_src_dir = os.getcwd()

os.chdir('../')
root_dir = os.getcwd()
src_dir = os.path.abspath('src/')

# copy all files within current directory to a destination directory
def copy_all(ext='.jpg', dst_dir=''):
    files = glob.glob('*' + ext)
    for f in files:
        shutil.copy(f, dst_dir)

os.chdir('datasets/')
datasets_dir = os.getcwd()
original_dir = os.path.abspath('columbia_gr')

for fold_id in range(n_folds):
    """
    Create a directory storing original and generated images
    Convert this directory into a h5 file
    """
    os.chdir(datasets_dir)
    aug_dir = 'columbia_exp_10_%d' % (fold_id)
    aug_file = 'columbia_exp_10.h5'
    gen_dir = os.path.abspath('genes_%d' % (fold_id))
    remove_if_exists(aug_dir)
    os.mkdir(aug_dir)
    aug_dir = os.path.abspath(aug_dir)

    # copy generated images
    os.chdir(gen_dir)
    copy_all(ext='.jpg', dst_dir=aug_dir)

    # copy original images
    os.chdir(original_dir)
    copy_all(ext='.jpg', dst_dir=aug_dir)

    # convert to h5
    os.chdir(my_src_dir)
    subprocess.run(['python', 'convert_img2h5.py', '--src_dir', aug_dir,
        '--dst_file', aug_file, '--ids', str(fold_id)])
    
# train
os.chdir(src_dir)
subprocess.run(['python', 'dpg_train_exp_10.py',
        '--oh' , '64', '--ow', '64',
        '--epoch', '50', '--log_dir', 'outputs_exp_10'])