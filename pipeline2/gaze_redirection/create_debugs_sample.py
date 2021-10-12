"""
Create a sample input folder for debugging
Author: An Trieu
"""

from shutil import copyfile
import os
from glob import glob

old_dir = 'mpiigaze'
new_dir = 'mpiigaze_debug'
file_paths = glob(os.path.join(old_dir, '*.jpg'))
if not(os.path.isdir(new_dir)):
	os.mkdir(new_dir)

for i in range(64):
	file_name = file_paths[i].split('/')[-1]
	new_file_path = os.path.join(new_dir, file_name)
	copyfile(file_paths[i], new_file_path)