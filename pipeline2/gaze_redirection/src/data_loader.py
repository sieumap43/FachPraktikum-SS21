# Dataloader.

import os
import tensorflow as tf
import numpy as np
np.random.seed(42)


class ImageData(object):

    """ Dataloader for loading images. """

    def __init__(self, load_size, channels, data_path, ids, fold_size, n_people):

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
        self.fold_size=fold_size # for k-fold cross-validation
        self.n_people = n_people

        self.data_path = data_path
        file_names = [f for f in os.listdir(data_path)
                      if f.endswith('.jpg')]
        self.file_dict = dict()
        for f_name in file_names:
            # key = f_name.split('.')[0].split('_')[0]
            # side = f_name.split('.')[0].split('_')[-1]
            # key = key + '_' + side
            fields = f_name[:-4].split('_')
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
        image_t = _to_image(filename_t)

        return image, angles_r, labels, image_t, angles_g

    def preprocess(self):
        """
        Experiment 16
        Train the gaze redirector on the training fold and generate new images from this
        training fold.
        The testing fold is ignored and used to test the gaze estimator in the subsequent
        part of the pipeline.
        """
        # validation ids
        val_ids = list(range(self.ids*self.fold_size + 1,
                            (self.ids+1)*self.fold_size + 1))
        
        # train ids
        train_ids = [j+1 for j in range(self.n_people) if j not in val_ids]

        for key in self.file_dict.keys():
            idx = int(key.split('_')[0]) # particpant number
            flip = 1
            if key.split('_')[-1] == 'R':
                flip = -1

            for f_r in self.file_dict[key]:

                file_path = os.path.join(self.data_path, f_r)

                h_angle_r = flip * float(
                    f_r.split('_')[-2].strip('H')) / 15.0
                v_angle_r = float(
                    f_r.split('_')[-3].strip('V')) / 10.0


                # images to be trained on
                for f_g in self.file_dict[key]:

                    file_path_t = os.path.join(self.data_path, f_g)

                    h_angle_g = flip * float(
                        f_g.split('_')[-2].strip('H')) / 15.0
                    v_angle_g = float(
                        f_g.split('_')[-3].strip('V')) / 10.0

                    if idx in train_ids:
                        self.train_images.append(file_path)
                        self.train_angles_r.append([h_angle_r, v_angle_r])
                        self.train_labels.append(key)
                        self.train_images_t.append(file_path_t)
                        self.train_angles_g.append([h_angle_g, v_angle_g])

                # images to be generated
                dh = np.random.normal(0, 5)
                dv = np.random.normal(0, 2)
                h_angle_g = h_angle_r + dh / 15.0
                v_angle_g = v_angle_r + dv / 10.0

                file_path_t = file_path # value irrelevant. Only for the code to run error free

                if idx in train_ids:
                    self.test_images.append(file_path)
                    self.test_angles_r.append([h_angle_r, v_angle_r])
                    self.test_labels.append(key)
                    self.test_images_t.append(file_path_t)
                    self.test_angles_g.append([h_angle_g, v_angle_g])

        print('\nFinished preprocessing the dataset...')