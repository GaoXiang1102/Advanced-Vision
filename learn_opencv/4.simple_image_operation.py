import cv2
import numpy as np

img = cv2.imread("../data/image/paris_1.jpg", cv2.IMREAD_COLOR)

img[100:200,100:200] = [255,255,255]

region = img[300:350, 400:450]
img[0:50, 0:50] = region

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()