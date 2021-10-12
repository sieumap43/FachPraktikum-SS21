import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import cv2
import os

def show(img):
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# LPIPS score
f = open("LPIPS.txt", "r")
lines = f.read().splitlines()
f.close()

names = np.array([i.split(": ")[0] for i in lines])
lpips = np.array([float(i.split(": ")[1]) for i in lines])

# remove instance where the original and target angles are the same
angles = np.array([float(i.split("_")[2]) for i in names])
names = names[angles != 0]
lpips = lpips[angles != 0]
angles = angles[angles != 0]

# compute average
(angle_uniq, n_uniq) = np.unique(angles, return_counts=True)
lpips_avg = np.zeros(len(angle_uniq))
for idx,angle in enumerate(angle_uniq):
	lpips_avg[idx] = np.mean(lpips[angles == angle])



# Image Blurriness (IB) score
img_paths = [os.path.join("log/eval/genes", i) for i in names]
ib = np.zeros(len(img_paths))

# compute individual IB scores
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float)
for idx,img_path in enumerate(img_paths):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ib[idx] = 1/cv2.Laplacian(img, cv2.CV_64F).var()

# compute average IB scores
ib_avg = np.zeros(len(angle_uniq))
for idx, angle in enumerate(angle_uniq):
	ib_avg[idx] = np.mean(ib[angles == angle])

# Gaze Estimation Error


# plot
fig, axes = plt.subplots(2)
axes[0].plot(angle_uniq, lpips_avg)
axes[0].set_xlabel("Redirected Angle")
axes[0].set_ylabel("LPIPS")

axes[1].plot(angle_uniq, ib_avg)
axes[1].set_xlabel("Redirected Angle")
axes[1].set_ylabel("Image Blurriness")

fig.suptitle("Evaluation results (lower scores are better)")
fig.tight_layout()
fig.savefig("scores")
fig.show()
