import numpy as np
import cv2
import os

imgs = []
folder = "AE4317_2019_datasets/sim_poles/20190121-160844"
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        scale = 25 # set scale to resize in percent
        width = int(img.shape[1] * scale / 100)
        height = int(img.shape[0] * scale / 100)
        img = cv2.resize(img, (width, height))
        imgs.append(img)
frame1 = imgs[0]
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
mags = []
angs = []
flows = []
i = 1
while(i<476):
    if i < 476:
        frame2 = imgs[i]
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flows.append(flow)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mags.append(mag)
    angs.append(ang)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    k = cv2.waitKey(30) & 0xff
    if k == 27 or i >=476:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)

    prvs = next
    i += 1

