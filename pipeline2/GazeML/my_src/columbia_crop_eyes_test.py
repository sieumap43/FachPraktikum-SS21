"""
Extract 150x90 (ow x oh) eye patches
Author: An Trieu
Date: 31.07.2021
"""

from glob import glob
import cv2 as cv
import numpy
import h5py
import os

import dlib
from imutils import face_utils

# img.shape = (oh, ow, channel)
# image params
(oh, ow) = (90, 150)

def crop(img, center, crop_size):
    def segment(middle, length):
        """
        Return the start and end indices given the middle index and array length
        """
        length = int(length)
        if length % 2 == 0:
            return (middle - int(length / 2), middle + int(length / 2))
        else:
            return (middle - int(length / 2), middle + int(length / 2) + 1)
            
    def is_inside_image(img, point):
        """
        Determine if the point lies within the image
        """
        for dim in range(2):
            if (point[dim] < 0) or (point[dim] > img.shape[dim]):
                return False
        return True
        
    x_start, x_end = segment(center[0], crop_size[0])
    y_start, y_end = segment(center[1], crop_size[1])
    if not(is_inside_image(img, (x_start, y_start)))\
    or not(is_inside_image(img, (x_end, y_end))):
        return None
    return img[x_start:x_end, y_start:y_end]

# load face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# list participants
ids_dirs = glob('../datasets/columbia_original_sampled/*')
assert ids_dirs == 56

# initialize destination data file
hf_w = h5py.File('columbia_150x90.h5', 'w')

for ids_dir in ids_dirs:

    ids = ids_dir.split('/')[-1].split('_')[0]
    ids_write = hf_w.create_group(ids)
    img_paths = glob(os.path.join(ids_dir, '*.jpg'))
    
    eye_data = []

    for img_path in img_paths:
        img = cv.imread(img_path)
        img_copy = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # detect face
        rects = detector(gray, 1)

        # detect landmarks
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        # extract eye landmarks
        eye_r = shape[36:42]
        eye_l = shape[42:48]

        # find min enclosing circles
        center_r, radius_r = cv.minEnclosingCircle(eye_r)
        center_l, radius_l = cv.minEnclosingCircle(eye_l)
        center_r = list(map(int, center_r))
        center_l = list(map(int, center_l))
        radius_r = int(radius_r)
        radius_l = int(radius_l)

        # crop eyes
        # img.shape = (oh, ow, c)
        # real image definition = (ow, oh)
        # center = (x, y) Hence remember to switch the dimension
        img_eye_r = crop(img_copy, (center_r[1], center_r[0]), (3*radius_r, 5*radius_r))
        img_eye_l = crop(img_copy, (center_l[1], center_l[0]), (3*radius_l, 5*radius_l))

        
hf_w.close()