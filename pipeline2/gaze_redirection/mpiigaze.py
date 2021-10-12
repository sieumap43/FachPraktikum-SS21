import numpy as np
import cv2
import h5py

def show(arr):
    cv2.imshow("image", arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hdf_path = "dataset/MPIIGaze.h5"
hdf5 = h5py.File(hdf_path, 'r')
print(hdf5.keys())
eye_image_shape = (36, 60)

oh,ow = eye_image_shape
i = 0
person_id = 'p%02d' % i
other_person_ids = ['p%02d' % j for j in range(15) if i != j]
keys_to_use=['train/' + s for s in other_person_ids]
key = keys_to_use[0]
entry = hdf5[key]
eye = entry['eye'][0]
show(eye)
