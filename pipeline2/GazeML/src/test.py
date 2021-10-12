import os
import sys
import glob
import tensorflow as tf
import numpy as np
import cv2 as cv
from models import DPG
tf.contrib.image

# with tf.Session() as sess:
#     hg = tf.train.import_meta_graph('../outputs/DPG_i150x90_f75x45_n32_m3_k8_p00/checkpoints/hourglass/model-26249.meta')
#     hg.restore(sess, tf.train.latest_checkpoint('../outputs/DPG_i150x90_f75x45_n32_m3_k8_p00/checkpoints/hourglass/'))
    
#     dn = tf.train.import_meta_graph('../outputs/DPG_i150x90_f75x45_n32_m3_k8_p00/checkpoints/densenet/model-26249.meta')
#     dn.restore(sess, tf.train.latest_checkpoint('../outputs/DPG_i150x90_f75x45_n32_m3_k8_p00/checkpoints/densenet/'))
class DataSource():
    def __init__(self):
        global _data_format
        # Check if GPU is available
        from tensorflow.python.client import device_lib
        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        gpu_available = False
        try:
            gpus = [d for d in device_lib.list_local_devices(config=session_config)
                    if d.device_type == 'GPU']
            gpu_available = len(gpus) > 0
        except:
            pass
        self.batch_size = 2
        self.data_format = 'NCHW' if gpu_available else 'NHWC'
        _data_format = self.data_format
        self.output_tensors = {
            'eye': tf.placeholder(tf.float32, [2, 36, 60, 1], name='eye')
        }
    def cleanup(self):
        pass
    def create_and_start_threads(self):
        pass

from tensorflow.python.client import device_lib
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
gpu_available = False
try:
    gpus = [d for d in device_lib.list_local_devices(config=session_config)
            if d.device_type == 'GPU']
    gpu_available = len(gpus) > 0
except:
    pass

# Initialize Tensorflow session
tf.logging.set_verbosity(tf.logging.INFO)
data_source = DataSource()
with tf.Session(config=session_config) as session:
    model = DPG(
            session, train_data={'videostream': data_source},
            first_layer_stride=2,
            num_modules=2,
            num_feature_maps=32,
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'combined_loss': ['hourglass', 'densenet'],
                    },
                    'metrics': ['gaze_mse', 'gaze_ang'],
                    'learning_rate': 0.0002,
                },
            ],
        )

    model.__del__()
