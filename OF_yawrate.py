import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import re

imgs = []
folder = "imgs_logs/slightly_right_obs"
sorted_list = sorted(os.listdir(folder), key=lambda f: int(re.sub('\D', '', f)))
sorted_list = sorted_list[84:144]  # 67: for frontal obs, 74:144
for filename in sorted_list:
    img = cv2.imread(os.path.join(folder, filename))
    # print(filename)
    if img is not None:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        scale = 20 # set scale to resize in percent
        width = int(img.shape[1] * scale / 100)
        height = int(img.shape[0] * scale / 100)
        img = cv2.resize(img, (width, height))
        imgs.append(img)
#imgs.reverse()
frame1 = imgs[0]
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
mags = []
angs = []
flows = []
diffs = []
divs = []
durations = []
i = 1
crop_portion = 1/3
h_crop_1 = round(imgs[0].shape[0] * crop_portion) # Set upper boundary to 'crop off' top section
h_crop_2 = round(imgs[0].shape[0] * 1 - crop_portion) # Set bottom boundary to 'crop off' bottom section
v_mid = int(imgs[0].shape[1] / 2) # Calculating vertical middle of the image
# Dense optical flow calculated with opencv function, example here in second section:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
tot = len(sorted_list)
while(i<tot):
    start = time.time()
    if i < tot:
        frame2 = imgs[i]
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, pyr_scale=0.1, levels=3,
                                        winsize=10, iterations=2, poly_n=2, poly_sigma=3.0, flags=0)
    flows.append(flow)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mags.append(mag)
    angs.append(ang)

    # This next part is inspired from https://www-ncbi-nlm-nih-gov.tudelft.idm.oclc.org/pmc/articles/PMC6604071/
    M_left = mag[:, 0:v_mid].sum()
    M_right = mag[:, v_mid:].sum()
    diff = M_right - M_left
    diffs.append(diff)
    prvs = next
    i += 1
    duration = time.time() - start
    durations.append(duration)

    flowleft = flow[:, 0:v_mid]
    flowright = flow[:, v_mid:]

    deriv_ux = cv2.Sobel(flow[..., 0], cv2.CV_32FC1, 1, 0, 3)
    deriv_vy = cv2.Sobel(flow[..., 1], cv2.CV_32FC1, 0, 1, 3)

    div = sum(deriv_ux.flatten() + deriv_vy.flatten())
    divs.append(div)

of_diff_means = []
for i in range(len(diffs)):
    mean = np.mean(diffs[0:i+1])
    of_diff_means.append(mean)

plt.figure(0)
plt.ylabel("Optical flow difference [-]", fontsize=12)
plt.xlabel("Timestep [-]", fontsize=12)
plt.tight_layout()
plt.plot(diffs)
#plt.plot(of_diff_means)
plt.show()
plt.figure(1)
plt.ylabel("Divergence [-]", fontsize=12)
plt.xlabel("Timestep [-]", fontsize=12)
plt.plot(divs)
plt.show()
print("Mean duration per loop", np.average(durations))
short_diffs = diffs[15:105]
print(st.describe(diffs))
print(st.describe(short_diffs))

