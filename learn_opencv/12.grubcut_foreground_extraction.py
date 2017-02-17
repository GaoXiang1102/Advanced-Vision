import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../data/image/guojing.jpg")
mask = np.zeros(img.shape[:2], np.uint8)

bgpModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect = (100, 50, 600, 500)

cv2.grabCut(img, mask, rect, bgpModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

cv2.imshow('foreground', img)
cv2.waitKey(0)
cv2.destroyAllWindows()