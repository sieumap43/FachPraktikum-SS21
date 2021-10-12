import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

label_files = glob.glob('Label/*.label')
label_files.sort()

for label_file in label_files:
    with open(label_file) as f:
        labels = f.read().splitlines()[1:]
        labels = [i.strip().split(' ') for i in labels]
        gaze_angles = np.array([list(map(float, i[5].split(','))) for i in labels])
        head_angles = np.array([list(map(float, i[6].split(','))) for i in labels])

yaws_gaze = gaze_angles[:, 0]
pitches_gaze = gaze_angles[:, 1]
yaws_head = head_angles[:, 0]
pitches_head = head_angles[:, 1]


fig, axes = plt.subplots(nrows=4, figsize=(10,10))
axes[0].hist(yaws_gaze)
axes[0].set_xlabel('radians')
axes[0].set_title('histogram of gaze yaw')

axes[1].hist(pitches_gaze)
axes[1].set_xlabel('radians')
axes[1].set_title('histogram of gaze pitch')

axes[2].hist(yaws_head)
axes[2].set_xlabel('radians')
axes[2].set_title('histogram of head yaw')

axes[3].hist(pitches_head)
axes[3].set_xlabel('radians')
axes[3].set_title('histogram of head pitch')

plt.tight_layout()
fig.show()
# fig.savefig('Angles Histogram.jpg')
