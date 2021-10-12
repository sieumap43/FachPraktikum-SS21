import cv2 as cv
from glob import glob
import h5py
import numpy as np
import os
# https://github.com/open-mpi/ompi/issues/6535#issuecomment-640116873
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Parameters
ow, oh = (60, 36)
img_paths = glob('../datasets/MPIIGaze_aug/*.jpg')
img_paths.sort()

img_labels = [i.split('/')[-1] for i in img_paths]
img_labels = [i[:-4].split('_') for i in img_labels] # strip .jpg then strip _aug then split

hf_r = h5py.File('../datasets/MPIIGaze.h5', 'r')
hf_w = h5py.File('../datasets/MPIIGaze_aug.h5', 'w')

ow_div_2 = ow/2
oh_div_2 = oh/2

def get_image(img_path:str, imsize=(60,36)):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img[int(img.shape[0]/2 - oh_div_2):int(img.shape[0]/2 + oh_div_2),\
        int(img.shape[1]/2 - ow_div_2):int(img.shape[1]/2 + ow_div_2)]
    return img

pi_div_180 = np.pi / 180
def get_gaze(img_label: list):
    yaw = float(img_label[-2].strip('H')) * pi_div_180
    pitch = float(img_label[-1].strip('V')) * pi_div_180
    return [yaw, pitch]

def get_head(img_label: list):
    return [0,0]

# Copy and append train data
train_w = hf_w.create_group('train')
train_r = hf_r['train']

for key in train_r.keys():
    print("Processing", key)
    p = train_w.create_group(key)
    p_img_paths = [i for i,j in zip(img_paths, img_labels) if j[0] == key]
    p_img_labels = [i for i in img_labels if i[0] == key]

    # Append augmented eye patches
    eye = train_r[key]['eye']
    eye_aug = np.array(list(map(get_image, p_img_paths)))
    eye = np.concatenate([eye, eye_aug], axis=0)


    # Append gazes of augmented data
    gaze = train_r[key]['gaze']
    gaze_aug = np.array(list(map(get_gaze, p_img_labels)))
    gaze = np.concatenate([gaze, gaze_aug], axis=0)

    # Append head poses of augmented data
    head = train_r[key]['head']
    head_aug = np.array(list(map(get_head, p_img_labels)))
    head = np.concatenate([head, head_aug], axis=0)

    # Shuffle data
    index = np.arange(head.shape[0])
    np.random.shuffle(index)
    eye = eye[index]
    gaze = gaze[index]
    head = head[index]

    p.create_dataset('eye', data=eye)
    p.create_dataset('gaze', data=gaze)
    p.create_dataset('head', data=head)

# Copy test data 
# hf_r.copy('test', hf_w) # not needed anymore

hf_w.close()
hf_r.close()