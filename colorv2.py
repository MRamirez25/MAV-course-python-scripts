import cv2 as cv
import numpy as np
import time
start = time.time()
ori = cv.imread("AE4317_2019_datasets/cyberzoo_aggressive_flight/20190121-144646/29748588.jpg")
img = ori.copy()
img = np.array(img)
blr = cv.blur(src=img,ksize=(7,7))
blr = cv.resize(blr,(blr.shape[1]//4,blr.shape[0]//4),interpolation = cv.INTER_AREA)
#can = cv.Canny(blr,15,40)
#cont,hier = cv.findContours(can,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
#cv.drawContours(can, cont, -1,(255,255,255),3)
h = blr.shape[0]
w = blr.shape[1]
mask = (blr[...,2] >= 106) & (blr[...,2] < 255) & (blr[...,1] >= 50 ) & (blr[...,1] < 91 ) & (blr[...,0] >= 60 ) & (blr[...,0] < 81)
blr[mask, :] = 0
blr[~(mask), :] = 255
# for i in range(h):
#     for j in range(w):
#         if blr[i,j,2] in range(106,255):
#             if blr[i,j,1] in range(50,91):
#                 if blr[i,j,0] in range(60,81):
#                     blr[i,j] = (0,0,0)
#                 else:
#                     blr[i,j] = (255,255,255)
#             else:
#                 blr[i,j] = (255,255,255)
#         else:
#             blr[i,j] = (255,255,255)
#cv.imshow("ori",ori)
#cv.imshow("blurred",blr)
#cv.waitKey(0)
#cv.destroyAllWindows()

s0 = mask[0:round(h/3), :].size - np.count_nonzero(mask[0:round(h/3), :])
s1 = mask[round(h/3):round(2*h/3), :].size - np.count_nonzero(mask[0:round(h/3), :])
s2 = mask[round(2*h/3):, :].size - np.count_nonzero(mask[0:round(h/3), :])

for i in range(round(h/3)):
    for j in range(w):
        if (blr[i,j,0] == 255 and blr[i,j,1] == 255 and blr[i,j,2] == 255):
            s0 = s0 + 1
        else:
            continue
for i in range(round(h/3),round(2*h/3)):
    for j in range(w):
        if (blr[i,j,0] == 255 and blr[i,j,1] == 255 and blr[i,j,2] == 255):
            s1 = s1 + 1
        else:
            continue
for i in range(round(2*h/3),h):
    for j in range(w):
        if (blr[i,j,0] == 255 and blr[i,j,1] == 255 and blr[i,j,2] == 255):
            s2 = s2 + 1
        else:
            continue
print(s0,s1,s2)
print("Time taken", time.time() - start)
if (s0 > s1 and s0 > s2):
    print("turn left")
elif (s1 > s0 and s1 >s2):
    print("go ahead")
elif (s2 > s0 and s2 > s1):
    print("turn right")