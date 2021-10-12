import os
import glob
import shutil
import subprocess
import numpy as np
import h5py

my_src_dir = os.getcwd()

root_dir = os.path.abspath('../')
src_dir = os.path.abspath('../src')
dst_dir = os.path.join(root_dir, 'datasets', 'columbia_exp_14')

# if os.path.exists(dst_dir):
#     shutil.rmtree(dst_dir)
# os.mkdir(dst_dir)

# # # crop eyes and create a h5 file from the original columbia images
# for k in range(7): # redundant participants numbers 57-60 will return None
#     procs_list = []
#     for i in range(k*8, (k+1)*8): # 56 participants in total
#         procs_list.append(subprocess.Popen(['python', 'columbia_crop_eyes_exp_14.py',
#             '--src_dir', 'datasets/columbia_original_face', '--dst_dir', 'datasets/columbia_exp_14',
#             '--width', '150', '--height', '90', '--channel', 'color',
#             '--scale', '0.5', '--flip_right', 'True',
#             '--participant', '%04d'%(i+1), '--dst_file', 'datasets/empty.h5']))
#     for proc in procs_list:
#         proc.wait()

# # # convert to h5
# data_dir = 'columbia_exp_14'
# data_file = 'columbia_exp_14.h5'

# os.chdir(my_src_dir)
# subprocess.run(['python', 'convert_img2h5.py', '--src_dir', data_dir,
#     '--dst_file', data_file, '--ids', '0',
#     '--width', '150', '--height', '90'])

# train
os.chdir(src_dir)
subprocess.run(['python', 'dpg_train_exp_14.py',
    '--log_dir', 'outputs_exp_14', '--epoch', '50',
    '--train_path', '../datasets/columbia_exp_14_0.h5', '--test_path', '../datasets/columbia_exp_14_0.h5',
    '--ow', '150', '--oh', '90'])
