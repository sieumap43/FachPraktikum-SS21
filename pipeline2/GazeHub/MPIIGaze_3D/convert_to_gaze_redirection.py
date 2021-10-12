import os
from glob import glob
import shutil
import numpy as np

label_paths = glob('./Label/p??.label')
img_dir = './Image'
output_dir = './mpiigaze'
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

for label_path in label_paths:
    with open(label_path) as f:
        labels = f.read().splitlines()[1:]
        labels = [i.strip().split(' ') for i in labels]
    img_paths = [i[0] for i in labels]
    whichEyes = [i[2] for i in labels]
    gaze_angles = np.array([list(map(float, i[5].split(','))) for i in labels])
    head_angles = np.array([list(map(float, i[6].split(','))) for i in labels])
    gaze_angles = (gaze_angles / np.pi * 180) #/ np.array([15, 10])
    head_angles = (head_angles / np.pi * 180) #/ np.array([15, 10])
    
    for i, img_path in enumerate(img_paths):
        ids = int(label_path.split('/')[-1].strip('p.label'))
        dist = 'NaNm' # distance to screen in the Columbia dataset. Unknown for MPIIGaze
        head = head_angles[i,0] # head yaw angle
        yaw = gaze_angles[i, 0] # yaw gaze angle
        pitch = gaze_angles[i, 1] # pitch gaze angle
        whichEye = 'L' if whichEyes[i] == 'left' else 'R'

        old_img_name = img_path.split('/')[-1]
        old_img_path = os.path.join(img_dir, img_path)
        new_img_path = os.path.join(output_dir, old_img_name)
        new_img_name = os.path.join(
            output_dir,
            '%04d_%s_%.2fP_%.2fV_%.2fH_%s.jpg' % (
            (ids+1), dist, head, yaw, pitch, whichEye)
        )

        shutil.copyfile(old_img_path, new_img_path)
        os.rename(new_img_path, new_img_name)
