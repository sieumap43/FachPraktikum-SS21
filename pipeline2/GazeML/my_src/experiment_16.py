"""
Author: An Trieu
Date: 20.09.2021
"""

from genericpath import exists
import cv2 as cv
import subprocess
import os
import numpy as np
import shutil
from glob import glob


################################################################
# Metadata
################################################################
# function to remove existing directory
def remove_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)


my_src_dir = os.getcwd()
src_dir = os.path.abspath('../src/')
os.chdir('../../')
root_dir = os.getcwd()
gr_dir = os.path.join(root_dir, 'gaze_redirection') # gaze redirector dir
ge_dir = os.path.join(root_dir, 'GazeML') # gaze estimator dir

# k-fold cross-validation params
from math import floor
n_folds = 5
n_people = 15
fold_ids = ['%04d'%(i) for i in range(n_folds)]
fold_size = floor(n_people / n_folds)

################################################################
# Prepare dataset
################################################################
print("Preparing dataset ...")
# create flip directory
os.chdir(os.path.join(gr_dir, 'dataset'))
flipped_dir = os.path.join(gr_dir, 'dataset', 'mpiigaze_flipped')
remove_if_exists(flipped_dir)
os.mkdir(flipped_dir)

# copy left eyes and flip right eyes
os.chdir('mpiigaze')
img_names = glob('*.jpg')
for name in img_names:
    if 'R' in name:
        img = cv.imread(name)
        img = cv.flip(img, 1)
        cv.imwrite(os.path.join(flipped_dir, name), img)
    else:
        shutil.copy(name, flipped_dir)
assert len(glob(os.path.join(flipped_dir, '*.jpg'))) == len(img_names)

# create directory to augment
os.chdir(gr_dir)
data_dir = os.path.join(gr_dir, 'dataset', 'mpiigaze_lite')
remove_if_exists(data_dir)
os.mkdir(data_dir)

# copy images from flipped dir to to-be-augmented dir
def filter_label(label_name, label_list):
    return next(filter(lambda x: label_name in x, label_list))

os.chdir(flipped_dir)
img_names = glob('*.jpg')
for name in img_names:
    label = name[:-4].split('_')
    yaw = filter_label('H', label)
    yaw = float(yaw.strip('H'))

    pitch = filter_label('V', label)
    pitch = float(pitch.strip('V'))

    if abs(yaw) <= 5 and abs(pitch) <= 5:
        shutil.copy(name, data_dir) 


################################################################
# Train Gaze Redirector
################################################################
print("Training Gaze Redirector ...")
os.chdir(gr_dir)
for fold_id in range(n_folds):
    log_dir = os.path.join(gr_dir, 'log_mpiigaze_%d' % (fold_id))

    # train
    gen_dir = os.path.join(log_dir, 'eval', 'genes_%d' % (fold_id))
    remove_if_exists(gen_dir)
    subprocess.run(['python', 'main.py', '--mode', 'train', '--data_path', flipped_dir, '--log_dir', log_dir,
        '--batch_size', '32', '--vgg_path', './vgg_16.ckpt', '--ids', str(fold_id), '--epoch', '150',
        '--fold_size', '%d'%(fold_size),'--n_people', '%d'%(n_people)])

    # # redirect
    subprocess.run(['python', 'main.py', '--mode', 'eval', '--data_path', data_dir, '--log_dir', log_dir,
        '--batch_size', '64', '--ids', str(fold_id),
        '--fold_size', '%d'%(fold_size), '--n_people', '%d'%(n_people)])

    # copy generated images
    dst_gen_dir = os.path.join(ge_dir, 'datasets')
    copied_dir = os.path.join(dst_gen_dir, 'genes_%d' % (fold_id))
    remove_if_exists(copied_dir)
    shutil.move(gen_dir, dst_gen_dir)
    

# copy original images
dst_orig_dir = os.path.join(ge_dir, 'datasets', 'mpiigaze')
remove_if_exists(dst_orig_dir)

os.chdir(os.path.join(gr_dir, 'dataset'))
print("Number of copied original images:", len(os.listdir('mpiigaze')))
shutil.copytree('mpiigaze', dst_orig_dir)

os.chdir(os.path.join(ge_dir, 'datasets'))
remove_if_exists('mpiigaze_gr')
os.rename('mpiigaze', 'mpiigaze_gr')

# flip the horizontal angle of right eyes
os.chdir('mpiigaze_gr')
img_names = glob('*.jpg')
n = 0 # sequence number to prevent deleting duplicates
for img_name in img_names:
    if 'R' in img_name:
        img_label = img_name.split('_')
        # print(img_label)
        for i, label in enumerate(img_label):
            if 'H' in label:
                yaw = float(label.strip('H'))
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
print("Training Gaze Estimator ...")
# function to copy all files within current directory to a destination directory
def copy_all(ext='.jpg', dst_dir=''):
    files = glob('*' + ext)
    for f in files:
        shutil.copy(f, dst_dir)

# create h5 files (one h5 file for each fold)
os.chdir(os.path.join(ge_dir, 'datasets/'))
datasets_dir = os.getcwd()
original_dir = os.path.abspath('mpiigaze_gr')

for fold_id in range(n_folds):
    """
    Create a directory storing original and generated images
    Convert this directory to a h5 file
    """
    os.chdir(datasets_dir)
    aug_dir = 'mpiigaze_exp_16_%d' % (fold_id)
    aug_file = 'mpiigaze_exp_16.h5'
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
        '--dst_file', aug_file, '--ids', str(fold_id),
        '--n_people', '%d'%(n_people)])
    
# train Gaze Estimator
os.chdir(src_dir)
subprocess.run(['python', 'dpg_train_exp_16.py',
        '--oh' , '64', '--ow', '64', '--batch_size', '64',
        '--epoch', '50', '--log_dir', 'outputs_exp_16', '--n_folds', str(n_folds),
        '--fold_size', '%d'%(fold_size), '--n_people', '%d'%(n_people)])