"""
Convert eye jpg images from the Columbia dataset (src_dir) to h5 (dst_file)
Default output image size: 64x64
Output image channel: grayscale (fixed)

Author: An Trieu
Date: 01.08.2021
"""

import argparse
from glob import glob
import h5py
import os
from data_converter import DataConverter

parser = argparse.ArgumentParser()

# source data directory
parser.add_argument('--src_dir', type=str, default='columbia',
                    help='source directory (assuming the directory is in GazeML/datasets')

# destination h5 file
parser.add_argument('--dst_file', type=str, default='columbia.h5',
                    help='destination h5 file (assuming the file will be in GazeML/datasets')

# id of the cross-validation fold
parser.add_argument('--ids', type=int, default=0,
                    help='id of the cross-validation fold')

# image size
parser.add_argument('--width', type=int, default=64,
                    help='image width')

parser.add_argument('--height', type=int, default=64,
                    help='image height')

# metadata
parser.add_argument('--n_people', type=int, default=56,
                    help='number of participants')

# write mode
# w - create file, truncate if exists
# w- - create file, fail if exists
parser.add_argument('--mode', type=str, default='w',
                    help='h5 file write mode')


if __name__ == '__main__':
    datasets_dir = os.path.abspath('../datasets')

    # Parameters
    params = parser.parse_args()

    img_paths = glob(os.path.join(datasets_dir, params.src_dir, '*.jpg'))
    img_paths.sort()
    (ow, oh) = (params.width, params.height)

    # destination file
    dst_file_without_ext = params.dst_file.split('.')[0]
    dst_file = os.path.join(datasets_dir,
                            '%s_%d.h5'%(dst_file_without_ext, params.ids))

    converter = DataConverter(path=dst_file, mode=params.mode,
        imsize=(ow, oh), n_people=params.n_people)

    converter.write_data(subset_name='train', img_paths=img_paths)
    # converter.write_data(subset_name='test', img_paths=img_paths)