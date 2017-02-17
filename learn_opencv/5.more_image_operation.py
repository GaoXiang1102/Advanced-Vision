import cv2
import numpy as np

img2 = cv2.imread("../data/image/paris_2.jpg", cv2.IMREAD_COLOR)
img3 = cv2.imread("../data/image/paris_3.jpg", cv2.IMREAD_COLOR)

reshaped_img2 = cv2.resize(img2, (400, 400))
reshaped_img3 = cv2.resize(img3, (400, 400))

# add = reshaped_img2 + reshaped_img3
# cv2.imshow('add', add)
# weighted = cv2.addWeighted(reshaped_img2, 0.5, reshaped_img3, 0.5, 0)
# cv2.imshow('image', weighted)

img2gray = cv2.cvtColor(reshaped_img3, cv2.COLOR_BGR2GRAY)

ret1, mask1 = cv2.threshold(img2gray, 80, 255, cv2.THRESH_BINARY)
ret2, mask2 = cv2.threshold(img2gray, 80, 255, cv2.THRESH_BINARY_INV)
foreground1 = cv2.bitwise_and(reshaped_img3, reshaped_img3, mask=mask1)
foreground2 = cv2.bitwise_and(reshaped_img3, reshaped_img3, mask=mask2)

cv2.imshow('original', reshaped_img3)
cv2.imshow('gray', img2gray)
cv2.imshow('mask1', mask1)
cv2.imshow('mask2', mask2)
cv2.imshow('foreground1', foreground1)
cv2.imshow('foreground2', foreground2)
cv2.waitKey(0)
cv2.destroyAllWindows()
