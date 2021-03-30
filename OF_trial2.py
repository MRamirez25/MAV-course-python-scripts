import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time

imgs = []
folder = "AE4317_2019_datasets/sim_poles/20190121-160844"
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        scale = 100 # in percent
        width = int(img.shape[1] * scale / 100)
        height = int(img.shape[0] * scale / 100)
        img = cv2.resize(img, (width, height))
        imgs.append(img)
frame1 = imgs[0]
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
mags = []
angs = []
flows = []
diffs = []
durations = []
i = 1
h_crop_1 = 80
h_crop_2 = 160
v_mid = int(imgs[0].shape[1] / 2)
while(i<476):
    start = time.time()
    if i < 476:
        frame2 = imgs[i]
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowPyrLK(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flows.append(flow)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mags.append(mag)
    angs.append(ang)

    # This next part is inspired from https://www-ncbi-nlm-nih-gov.tudelft.idm.oclc.org/pmc/articles/PMC6604071/
    M_left = mag[h_crop_1:h_crop_2, 0:v_mid].sum()
    M_right = mag[h_crop_1:h_crop_2, v_mid:].sum()
    diff = M_left - M_right
    diffs.append(diff)
    prvs = next
    i += 1
    duration = time.time() - start
    durations.append(duration)

plt.plot(diffs)
plt.show()
print("Mean duration per loop", np.average(durations))

