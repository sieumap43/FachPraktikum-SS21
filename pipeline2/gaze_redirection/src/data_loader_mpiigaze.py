# Dataloader.

import numpy as np
np.random.seed(42)
import os
import tensorflow as tf


class ImageData(object):

    """ Dataloader for loading images. """

    def __init__(self, load_size, channels, data_path, ids):

        """ Init.

        Parameters
        ----------
        load_size: int, input image size.
        channels: int, number of channels.
        data_path: str, path of input images.
        ids: int, train/test split point.

        """

        self.load_size = load_size
        self.channels = channels
        self.ids = ids

        self.data_path = data_path
        file_names = [f for f in os.listdir(data_path)
                      if f.endswith('.jpg')]
        self.file_dict = dict()
        for f_name in file_names:
            # key = f_name.split('.')[0].split('_')[0]
            # side = f_name.split('.')[0].split('_')[-1]
            # key = key + '_' + side
            fields = f_name.strip('.jpg').split('_')
            identity = fields[0]
            head_pose = fields[2]
            side = fields[-1]
            key = '_'.join([identity, head_pose, side])
            if key not in self.file_dict.keys():
                self.file_dict[key] = []
                self.file_dict[key].append(f_name)
            else:
                self.file_dict[key].append(f_name)

        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.test_images_t = []
        self.test_angles_g = []

    def image_processing(
        self,
        filename,
        angles_r,
        labels,
        filename_t,
        angles_g
    ):
        """ Process input images.

        Parameters
        ----------
        filename: str, path of input image.
        angles_r: list, gaze direction of input image.
        labels: int, subject id. (deprecated!)
        filename_t: str, path of target image.
        angles_g: list, gaze direction of target image.

        Returns
        -------
        image: tensor, float32, normalized input image.
        angles_r: angels_r.
        labels: labels.
        image_t: tensor, float32, normalized target image.
        angles_g: angles_g.

        """

        def _to_image(file_name):

            """ Load image, normalize it and convert it into tf.tensor.

            Parameters
            ----------
            file_name: str, image path.

            Returns
            -------
            img: tf.tensor, tf.float32. Image tensor.

            """
            x = tf.read_file(file_name)
            img = tf.image.decode_jpeg(x, channels=self.channels)
            img = tf.image.resize_images(img, [self.load_size, self.load_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0

            return img
        image = _to_image(filename)
        image_t = _to_image(filename)
        
        return image, angles_r, labels, image_t, angles_g

    def preprocess(self):
        dhs = np.array([-2, -1, 1, 2]) / 15.0
        dvs = np.array([-1, 1]) / 10.0

        # Only augment a quarter of the dataset
        permute_keys = np.random.permutation(list(self.file_dict.keys()))
        chosen_keys = permute_keys[:int(len(self.file_dict.keys()) / 4)]

        for key in chosen_keys:

            idx = key.split('_')[0]
            flip = 1
            if key.split('_')[-1] == 'R':
                flip = -1

            for f_r in self.file_dict[key]:
                file_path = os.path.join(self.data_path, f_r)

                h_angle_r = flip * float(
                    f_r.split('_')[-2].split('H')[0]) / 15.0
                v_angle_r = float(
                    f_r.split('_')[-3].split('V')[0]) / 10.0

                add_train_data_once = True # only for debugging
                for dh in dhs:
                    for dv in dvs:
                        h_angle_g = h_angle_r + dh
                        v_angle_g = v_angle_r + dv
                        if (h_angle_g == h_angle_r and v_angle_g == v_angle_r): continue

                        file_path_t = f_r.split('_')
                        file_path_t[3] = str(np.round(v_angle_g, 4)) + 'V'
                        file_path_t[4] = str(np.round(h_angle_g, 4)) + 'H'
                        file_path_t = os.path.join(self.data_path, "_".join(file_path_t))

                        ##########
                        # Only for debugging
                        if add_train_data_once:
                            add_train_data_once = False
                            self.train_images.append(file_path)
                            self.train_angles_r.append([h_angle_r, v_angle_r])
                            self.train_labels.append(idx)
                            self.train_images_t.append(file_path_t)
                            self.train_angles_g.append([h_angle_g, v_angle_g])
                        ##########
                        self.test_images.append(file_path)
                        self.test_angles_r.append([h_angle_r, v_angle_r])
                        self.test_labels.append(idx)
                        self.test_images_t.append(file_path_t)
                        self.test_angles_g.append([h_angle_g, v_angle_g])

        print('Number of to-be-generated images:', len(self.test_images_t))
        print('\nFinished preprocessing the dataset...')
