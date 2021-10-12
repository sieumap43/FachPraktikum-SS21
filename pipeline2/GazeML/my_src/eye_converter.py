"""
Convert images to h5
Author: An Trieu
Date: 28.07.2021
"""
import cv2 as cv
import h5py
import numpy as np
import os
import dlib
from imutils import face_utils
from data_converter import DataConverter
# https://github.com/open-mpi/ompi/issues/6535#issuecomment-640116873
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class EyeConverter(DataConverter):
    def __init__(self, channel='gray', scale=1, flip_right=False, **kwargs):
        """Create queues and threads to read and preprocess data."""
        self.channel = channel
        self.scale = scale
        self.flip_right = flip_right

        # load face and landmark detectors
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # Call parent class constructor
        super().__init__(**kwargs)

    def crop(self, img, center, crop_size):
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

    def get_image(self, img_path:str):
        # load iamge
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if self.channel == 'gray':
            img_copy = gray.copy()
        elif self.channel == 'color':
            img_copy = img.copy()


        # detect face
        rects = self.detector(gray, 1)
        if len(rects) != 1: return None, None

        # detect landmarks
        shape = self.predictor(gray, rects[0])
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
        img_eye_r = self.crop(img_copy, (center_r[1], center_r[0]), (self.scale*3*radius_r, self.scale*5*radius_r))
        img_eye_l = self.crop(img_copy, (center_l[1], center_l[0]), (self.scale*3*radius_l, self.scale*5*radius_l))

        img_eye_r = cv.resize(img_eye_r, (self.ow, self.oh))
        img_eye_l = cv.resize(img_eye_l, (self.ow, self.oh))
        return img_eye_r, img_eye_l

    def get_data(self, img_path:str):
        """
        Get image, gaze and head pose
        """
        img_label = img_path.split('/')[-1].split('.')[0].split('_')

        img_r, img_l = self.get_image(img_path)
        if (img_r is None) or (img_l is None):
            return (None, None, None), (None, None, None)

        gaze = self.get_gaze(img_label)
        head = self.get_head(img_label)

        gaze_l = gaze
        gaze_r = gaze
        if self.flip_right:
            img_r = cv.flip(img_r, 1)
            gaze_r[0] = -1 * gaze_r[0]

        data_r = (img_r, gaze_r, head)
        data_l = (img_l, gaze_l, head)

        return (data_r, data_l)

    def concat_data(self, p_img_paths: list):
        """
        Concat eye patch arrays, gaze arrays and head pose arrays respectively
        """
        data = list(filter(lambda x: x[0][0] is not None and x[1][0] is not None,
            map(self.get_data, p_img_paths)))
        if len(data) == 0:
            return None, None, None

        data_r, data_l = zip(*data)
        eye_r, gaze_r, head_r = zip(*data_r)
        eye_l, gaze_l, head_l = zip(*data_l)

        eye = np.vstack([eye_r, eye_l])
        gaze = np.vstack([gaze_r, gaze_l])
        head = np.vstack([head_r, head_l])

        return eye, gaze, head


    def write_jpg(self, dst_dir:str, img_paths:list):
        """
        Write to images into directory dst_dir
        """
        img_labels = [i.split('/')[-1] for i in img_paths]
        img_labels = [i[:-4].split('_') for i in img_labels] # strip .jpg

        keys = np.array([label[0] for label in img_labels])
        keys = np.unique(keys)
        keys.sort()

        n = 0 # sequence number to prevent duplicates
        for key in keys:
            print("Processing", key)
            p_img_paths = [i for i,j in zip(img_paths, img_labels) if j[0] == key]

            for img_path in p_img_paths:
                img_label = img_path.split('/')[-1].split('.')[0].split('_')
            
                head_label = next(filter(lambda x: 'P' in x, img_label))
                head_label = int(head_label.strip('P'))
            
                yaw_label = next(filter(lambda x: 'H' in x, img_label))
                yaw_label = int(yaw_label.strip('H'))

                pitch_label = next(filter(lambda x: 'V' in x, img_label))
                pitch_label = int(pitch_label.strip('V'))

                # flip the label, the image is already flipped in get_image
                if self.flip_right: 
                    yaw_label *= -1

                data_r, data_l = self.get_data(img_path)
                if data_r[0] is None or data_l[0] is None:
                    continue
                cv.imwrite(os.path.join(
                    dst_dir,
                    '%s_2m_%d_P%d_H%d_V%d_%s.jpg' % (
                        img_label[0], n, head_label,
                        yaw_label, pitch_label, 'R')),
                    data_r[0]
                    )
                cv.imwrite(os.path.join(
                    dst_dir,
                    '%s_2m_%d_P%d_H%d_V%d_%s.jpg' % (
                        img_label[0], n, head_label,
                        yaw_label, pitch_label, 'L')),
                    data_l[0]
                    )
                n += 1