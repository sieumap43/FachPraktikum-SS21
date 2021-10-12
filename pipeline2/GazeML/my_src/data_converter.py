"""
Convert images to h5
Author: An Trieu
Date: 28.07.2021
"""


import cv2 as cv
import h5py
import numpy as np
import os

from numpy.core.fromnumeric import size
# https://github.com/open-mpi/ompi/issues/6535#issuecomment-640116873
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class DataConverter():
    def __init__(self, path='columbia.h5', mode='w-', imsize=(64,64), n_people=56):
        if mode == 'w-' and os.path.exists(path):
            raise  FileExistsError("%s exists! Choose a different name or write mode!" % (path))
        self.hf_w = h5py.File(path, mode)

        self.pi_div_180 = np.pi / 180.0 # for fast calculation

        self.ow, self.oh = imsize

        self.n_samples = 0
        
        self.n_people = n_people

    def get_image(self, img_path:str):
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (self.ow,self.oh))
        # img_name = img_path.split('/')[-1]
        # if 'R' in img_name: img = cv.flip(img, 1) # flip right eye patches
        return img

    def get_gaze(self, img_label: list):
        yaw_str = next(filter(lambda x: "H" in x, img_label))
        pitch_str = next(filter(lambda x: "V" in x, img_label))
        yaw = float(yaw_str.strip('H')) * self.pi_div_180
        pitch = float(pitch_str.strip('V')) * self.pi_div_180
        return [yaw, pitch]

    def get_head(self, img_label: list):
        # yaw_str = next(filter(lambda x: "P" in x, img_label))
        # yaw = float(yaw_str.strip('P')) * self.pi_div_180
        # pitch = 0
        # return [yaw, pitch]
        return [0, 0]

    def get_data(self, img_path:str):
        img_name = img_path.split('/')[-1]
        img_name_no_ext = os.path.splitext(img_name)[0]
        img_label = img_name_no_ext.split('_')

        img = self.get_image(img_path)
        if img is None:
            return (None, None, None)

        gaze = self.get_gaze(img_label)
        head = self.get_head(img_label)
        return (img, gaze, head)

    def concat_data(self, p_img_paths:list):
        # Append eye patches
        data = list(filter(lambda x: x[0] is not None,
                    map(self.get_data, p_img_paths)))


        if len(data) == 0: return None, None, None
        
        eye, gaze, head = zip(*data)

        eye = np.array(eye)
        gaze = np.array(gaze)
        head = np.array(gaze)

        return eye, gaze, head

    def write_data(self, subset_name, img_paths):
        """
        Write to the h5 file
        """
        img_labels = [i.split('/')[-1] for i in img_paths]
        img_labels = [i[:-4].split('_') for i in img_labels] # strip .jpg

        subset_w = self.hf_w.create_group(subset_name)
        keys = ["%04d"%(i+1) for i in range(self.n_people)] # p01 - p57

        print("Write %s data" % (subset_name))
        for key in keys:
            print("Processing", key)
            p = subset_w.create_group(key)
            p_img_paths = [i for i,j in zip(img_paths, img_labels) if j[0] == key]

            eye, gaze, head = self.concat_data(p_img_paths)
            if eye is None:
                continue

            # Shuffle data
            index = np.arange(head.shape[0])
            np.random.shuffle(index)
            eye = eye[index]
            gaze = gaze[index]
            head = head[index]

            self.n_samples += eye.shape[0]

            p.create_dataset('eye', data=eye)
            p.create_dataset('gaze', data=gaze)
            p.create_dataset('head', data=head)

        self.hf_w.create_dataset('data_shape', (self.n_samples, self.ow, self.oh))

    def __del__(self):
        self.hf_w.close()
        del self