import cv2
import numpy as np

img = cv2.imread("../data/image/paris_1.jpg", cv2.IMREAD_COLOR)

cv2.line(img, (0,0), (150,150), (255,255,255), 5)
cv2.rectangle(img, (15,25), (200,150), (0,255,0), 2)
cv2.circle(img, (100,100), 50, (0,0,255), -1)

pts = np.array([[100,100], [500,100], [250,300]], np.int32)
cv2.polylines(img, [pts], True, (0,255,255), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV tuts', (200,200), font, 1, (200,255,255), 3)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
