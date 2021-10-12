"""
Crop out eyes image from the full face image of the Columbia dataset
and convert them to h5
Default ouput image size: 150x90
Output image channel: grayscale (optional)

Author: An Trieu
Date: 01.08.2021
"""

import argparse
from glob import glob
import h5py
import os
from eye_converter import EyeConverter

parser = argparse.ArgumentParser()

# source data directory
parser.add_argument('--src_dir', type=str, default='datasets/columbia_original',
                    help='source directory')

# destination h5 file
parser.add_argument('--dst_file', type=str, default='datasets/columbia_cropped.h5',
                    help='destination h5 file')

# id of the cross-validation fold
parser.add_argument('--ids', type=int, default=0,
                    help='id of the cross-validation fold')

# image size
parser.add_argument('--width', type=int, default=150,
                    help='image width')

parser.add_argument('--height', type=int, default=90,
                    help='image height')

# image channel
parser.add_argument('--channel', type=str, default='gray',
                    help='image channel gray or color', choices=['color', 'gray'])

# output image scale
parser.add_argument('--scale', type=float, default=0.6,
                    help='output image dimension scale')

# flip right eye or not
parser.add_argument('--flip_right', type=bool, default=False,
                    help='flip all right eye images (and their yaw angles)')

# write mode
# w - create file, truncate if exists
# w- - create file, fail if exists
parser.add_argument('--mode', type=str, default='w',
                    help='h5 file write mode')


if __name__ == '__main__':
    root_dir = "/".join(os.getcwd().split('/')[:-1])

    # Parameters
    params = parser.parse_args()

    src_dir = os.path.join(root_dir, params.src_dir)
    if not(os.path.exists(src_dir)):
        raise FileExistsError("%s does not exist!" %(src_dir))

    img_paths = glob(os.path.join(src_dir, '*.jpg'))
    img_paths.sort()
    
    (ow, oh) = (params.width, params.height)

    # destination file
    dst_file_without_ext = params.dst_file.split('.')[0]
    dst_file = os.path.join(root_dir,
                            '%s_%d.h5'%(dst_file_without_ext, params.ids))
    
    converter = EyeConverter(path=dst_file, mode=params.mode, scale=params.scale,
                             imsize=(ow, oh), channel=params.channel,
                             flip_right=params.flip_right)

    # train and test have the same data because k-fold cross-validation is being employed
    converter.write_data(subset_name='train', img_paths=img_paths)
    # converter.write_data(subset_name='test', img_paths=img_paths)