import numpy as np
import cv2

cap = cv2.VideoCapture("../../data/video/day_2.avi")

fgbg1 = cv2.BackgroundSubtractorMOG()
fgbg2 = cv2.BackgroundSubtractorMOG2()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while True:
    ret, frame = cap.read()
    fgmask1 = fgbg1.apply(frame)
    fgmask2 = fgbg2.apply(frame)
    fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('original', frame)
    cv2.imshow('fg1', fgmask1)
    cv2.imshow('fg2', fgmask2)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
