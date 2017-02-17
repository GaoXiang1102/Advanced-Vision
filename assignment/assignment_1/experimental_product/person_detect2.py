import cv2
import numpy as np
import imutils
import Queue

# number of frame
num_frame = 0

# area of working at the desk
working_at_the_desk_area = (((325, 125)), (725, 525))
# area of standing beside the cabinet
standing_beside_the_file_cabinet_area = ((1025, 280), (1275, 700))

# statistical variable
num_of_frame_working_at_the_desk = 1
num_of_frame_with_no_one_in_the_office = 0
num_of_frame_beside_filing_cabinet = 0

# a queue to store history centroids to avoid the influence of noise and anomaly
centroids_history = Queue.Queue(maxsize = 5)

# define a function to do background subtraction
def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t0, t1)
    d2 = cv2.absdiff(t1, t2)
    return cv2.bitwise_or(d1, d2)


# clean up the image
def clean_up_image(image):
    fg = cv2.GaussianBlur(image, (35, 35), 0)
    fg = cv2.threshold(fg, 3, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))
    fg = cv2.GaussianBlur(fg, (65, 65), 0)
    fg = cv2.threshold(fg, 3, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.dilate(fg, np.ones((10, 10), np.uint8), iterations=10)
    fg = cv2.GaussianBlur(fg, (99, 99), 0)
    fg = cv2.threshold(fg, 3, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.erode(fg, np.ones((10, 10), np.uint8), iterations=10)
    fg = cv2.GaussianBlur(fg, (99, 99), 0)
    fg = cv2.threshold(fg, 3, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.erode(fg, np.ones((10, 10), np.uint8), iterations=10)
    fg = cv2.GaussianBlur(fg, (155, 155), 0)
    fg = cv2.threshold(fg, 3, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.dilate(fg, np.ones((10, 10), np.uint8), iterations=10)
    fg = cv2.GaussianBlur(fg, (55, 55), 0)
    fg = cv2.threshold(fg, 3, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.erode(fg, np.ones((10, 10), np.uint8), iterations=18)
    return fg

# judge if the person is working at the desk
def working_at_the_desk(cX, cY):
    return (cX >= working_at_the_desk_area[0][0]) & (cX <= working_at_the_desk_area[1][0])\
           & (cY >= working_at_the_desk_area[0][1]) & (cY <= working_at_the_desk_area[1][1])

# judge if the person is working at the file cabinet
def working_at_the_file_cabinet(cX, cY):
    return (cX >= standing_beside_the_file_cabinet_area[0][0]) & (cX <= standing_beside_the_file_cabinet_area[1][0])\
           & (cY >= standing_beside_the_file_cabinet_area[0][1]) & (cY <= standing_beside_the_file_cabinet_area[1][1])


# compute the average point of the history centroids
def get_average_centroid_point():
    if centroids_history.empty():
        return (0, 0)
    else:
        sum_x = 0
        sum_y = 0
        for centroid in centroids_history.queue:
            sum_x += centroid[0]
            sum_y += centroid[1]
        return (sum_x / centroids_history.qsize(), sum_y / centroids_history.qsize())


# compute the distance of two points
def get_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)



# read the video data
cam = cv2.VideoCapture("../../data/video/day_2.avi")

# read the first image
t_minus = cam.read()[1]
# convert the first image to grey scale
t_minus_gray = cv2.cvtColor(t_minus, cv2.COLOR_BGR2GRAY)
# read the second image
t = cam.read()[1]
# convert the second image to grey scale
t_gray = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
# read the third image
t_plus = cam.read()[1]
# convert the third image to grey scale
t_plus_gray = cv2.cvtColor(t_plus, cv2.COLOR_BGR2GRAY)


num_frame += 3


while True:

    # get the foreground image by background subtraction
    fg = diffImg(t_minus_gray, t_gray, t_plus_gray)

    # detect edges in foreground image
    edges = cv2.Canny(fg, 100, 100)

    # clean up image
    fg = clean_up_image(edges)

    # find all the contours in the cleaned up image
    contours = cv2.findContours(fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    # filter those contours with very small area
    cnts = []
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            cnts.append(contour)


    # if there are contours detected
    if (len(cnts) > 0):
        c = None
        cX = 0
        cY = 0

        # if only one contour is detected, it will be ok
        if len(cnts) == 1:
            c = cnts[0]
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        # if there are more than one contour detected, find the most probable contour
        # by selecting the shortest distance between the centroid of the contour to the
        # average point of the history centroids.
        else:
            shortest_distance = 5000
            for cnt in cnts:
                M = cv2.moments(cnt)
                _X = int(M["m10"] / M["m00"])
                _Y = int(M["m01"] / M["m00"])
                distance = get_distance((_X, _Y), get_average_centroid_point())
                if distance < shortest_distance:
                    shortest_distance = distance
                    c = cnt
                    cX = _X
                    cY = _Y

        print cv2.contourArea(c)

        # update the history centroids queue
        if not centroids_history.full():
            centroids_history.put((cX, cY))
        else:
            centroids_history.get()
            centroids_history.put((cX, cY))


        # do statistics and draw the centroid only when the history queue is full
        if centroids_history.full():
            if working_at_the_desk(cX, cY):
                num_of_frame_working_at_the_desk += 1
            if working_at_the_file_cabinet(cX, cY):
                num_of_frame_beside_filing_cabinet += 1

            cv2.drawContours(t, [c], -1, (0, 255, 0), 5)
            cv2.circle(t, (cX, cY), 10, (0, 0, 255), -1)
            cv2.putText(t, "center: ({:d},{:d})".format(cX, cY), (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)

    # no contours detected
    else:
        # if the history queue is not empty, pop the oldest centroid.
        # if there are still remaining centroids in the queue, calculate
        # the average point as the centroid of the current frame, and draw it.
        # if the queue is already empty, we are convinced that there is no one
        # in the office.
        if not centroids_history.empty():
            centroids_history.get()
            if not centroids_history.empty():
                if working_at_the_desk(cX, cY):
                    num_of_frame_working_at_the_desk += 1
                if working_at_the_file_cabinet(cX, cY):
                    num_of_frame_beside_filing_cabinet += 1
                cv2.circle(t, get_average_centroid_point(), 10, (0, 0, 255), -1)
                cv2.putText(t, "center: ({:d},{:d})".format(cX, cY), (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
        else:
            num_of_frame_with_no_one_in_the_office += 1


    cv2.putText(t, "working at the desk: {:d}".format(num_of_frame_working_at_the_desk), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 3)
    cv2.putText(t, "standing beside the file cabinet: {:d}".format(num_of_frame_beside_filing_cabinet), (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 255), 3)
    cv2.putText(t, "no one in the office: {:d}".format(num_of_frame_with_no_one_in_the_office), (900, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
    cv2.rectangle(t, working_at_the_desk_area[0], working_at_the_desk_area[1], (255, 0, 0), 5)
    cv2.rectangle(t, standing_beside_the_file_cabinet_area[0], standing_beside_the_file_cabinet_area[1], (100, 0, 255), 5)


    cv2.imshow('original', t)


    # moving the sliding window one frame forward
    t_minus = t
    t_minus_gray = t_gray
    t = t_plus
    t_gray = t_plus_gray
    t_plus = cam.read()[1]
    t_plus_gray = cv2.cvtColor(t_plus, cv2.COLOR_BGR2GRAY)

    num_frame += 1

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release the camera
cam.release()
cv2.destroyAllWindows()