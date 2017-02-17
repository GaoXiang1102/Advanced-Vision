import cv2
import numpy as np
import imutils
import time


# initialize frame number
num_frame = 0

# initialize the state (whether Bob is in his office) 1: in his office 0: not in his office
state = 1

# initialize a list of frames in which Bob enters his office
enter_times = []

# initialize a list of frames in which Bob exits his office
exit_times = []


# rectangle area for counting frames of working at the table
working_at_the_desk_area = (((325, 125)), (725, 525))
# rectangle area for counting frames of standing beside the file cabinet
standing_beside_the_file_cabinet_area = ((1000, 280), (1275, 700))


# statistical variable
num_of_frame_working_at_the_desk = 0
num_of_frame_with_no_one_in_the_office = 0
num_of_frame_beside_filing_cabinet = 0


# normalize the RGB image into RGS image (red_norm, green_norm, lightness)
def rgb_norm(rgb_image):
    red_channel = rgb_image[:,:,0].astype(float)
    green_channel = rgb_image[:,:,1].astype(float)
    blue_channel = rgb_image[:,:,2].astype(float)
    sum_channel = red_channel+green_channel+blue_channel
    lightness = sum_channel / 3.0
    red_norm = red_channel / (sum_channel+0.00000001) *255
    green_norm = green_channel / (sum_channel+0.00000001) *255
    return np.array([red_norm, green_norm, lightness], dtype=np.uint8)



#  do background_subtraction between two normalized RGS images
def background_substraction(bg_rgs, image_rgs, s_sub_threshold):
    bg_r, bg_g, bg_s = bg_rgs
    image_r, image_g, image_s = image_rgs
    s_sub = np.abs((bg_s-bg_s.mean()) - (image_s-image_s.mean())) > s_sub_threshold
    return s_sub



# clean up the image after background subtraction
def clean_up_image(image):
    fg = cv2.GaussianBlur(image, (21, 21), 0)
    fg = cv2.threshold(fg, 3, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((100, 100), np.uint8))
    return fg



# find the contour and centroid of the largest object detected
def find_the_largest_object(image):
    # find all the contours in the cleaned up image
    contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours[0] if imutils.is_cv2() else contours[1]

    # if there are contours detected
    if (len(cnts) > 0):
        contour_area = 0

        # if only one contour is detected, it will be ok
        if len(cnts) == 1:
            c = cnts[0]
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_area = cv2.contourArea(c)

        # if there are more than one contour detected, select the one with the largest area
        else:
            largest_contour_area = 0
            for cnt in cnts:
                M = cv2.moments(cnt)
                _X = int(M["m10"] / M["m00"])
                _Y = int(M["m01"] / M["m00"])
                contour_area = cv2.contourArea(cnt)
                if contour_area > largest_contour_area:
                    largest_contour_area = contour_area
                    c = cnt
                    cX = _X
                    cY = _Y
            contour_area = largest_contour_area
        print contour_area
        if contour_area <= 10000:
            return None
        return (c, (cX, cY))
    else:
        return None



# get the y value of the top-most point on a contour
def find_y_of_the_top_most_point_on_contour(contour):
    y_min = 5000
    for pnt in contour:
        if pnt[0][1] < y_min:
            y_min = pnt[0][1]
    return y_min



# mark the detected person with a rectangle
def mark_person(image, contour, centroid, rectangle_with):
    y_top = find_y_of_the_top_most_point_on_contour(contour)
    if y_top <= 80:
        y_top = centroid[1] - 300
    y_bottom = y_top + (centroid[1] - y_top) * 2

    x_left_most = centroid[0] - rectangle_with / 2
    x_right_most = centroid[0] + rectangle_with / 2


    cv2.rectangle(image, (x_left_most, y_top), (x_right_most, y_bottom), (0, 255, 0), 5)
    cv2.circle(image, centroid, 10, (0, 0, 255), -1)
    cv2.putText(image, "center: ({:d},{:d})".format(centroid[0], centroid[1]), (centroid[0] - 20, centroid[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)



# whether the person is working at the desk
def working_at_the_desk(cX, cY):
    return (cX >= working_at_the_desk_area[0][0]) & (cX <= working_at_the_desk_area[1][0])\
           & (cY >= working_at_the_desk_area[0][1]) & (cY <= working_at_the_desk_area[1][1])



# whether the person is standing at the file cabinet
def standing_beside_the_file_cabinet(cX, cY):
    return (cX >= standing_beside_the_file_cabinet_area[0][0]) & (cX <= standing_beside_the_file_cabinet_area[1][0])\
           & (cY >= standing_beside_the_file_cabinet_area[0][1]) & (cY <= standing_beside_the_file_cabinet_area[1][1])



# initialize the camera to read the video
cam = cv2.VideoCapture("../../data/video/day_2.avi")

# load two background image for background subtraction
bg1 = rgb_norm(cv2.imread("../../data/image/day_2/inspacecam163_2016_02_19_13_05_01.jpg"))
bg2 = rgb_norm(cv2.imread("../../data/image/day_2/inspacecam163_2016_02_19_13_45_06.jpg"))



while True:

    # load a frame from the the video
    success, image = cam.read()

    if success is not True:
        break

    image_rgs = rgb_norm(image)
    num_frame += 1

    # do background subtraction with two loaded background images
    background_sub1 = background_substraction(bg1, image_rgs, 40)
    background_sub2 = background_substraction(bg2, image_rgs, 40)

    # get the final background subtracted image
    background_sub = np.array(background_sub1*background_sub2*255, dtype=np.uint8)

    # clean up the image
    cleaned_image = clean_up_image(background_sub)

    # find the largest detected object
    largest_object = find_the_largest_object(cleaned_image)

    # find the person
    if largest_object != None:
        # if the state of the last frame is 0, Bob is entering the office
        if state == 0:
            enter_times.append(num_frame)
        # Bob is in his office, set the state to be 1
        state = 1
        contour = largest_object[0]
        centroid = largest_object[1]
        # mark the detected person with contour and centroid
        mark_person(image, contour, centroid, 200)

        if working_at_the_desk(centroid[0], centroid[1]):
            num_of_frame_working_at_the_desk += 1
        elif standing_beside_the_file_cabinet(centroid[0], centroid[1]):
            num_of_frame_beside_filing_cabinet += 1
    # not find the person
    else:
        # if the state of the last frame is 1, Bob is exiting his office
        if state == 1:
            exit_times.append(num_frame)
        # Bob is absent, set the state to be 0
        state = 0
        num_of_frame_with_no_one_in_the_office += 1

    cv2.putText(image, "working at the desk: {:d}".format(num_of_frame_working_at_the_desk), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
    cv2.putText(image, "standing beside the filing cabinet: {:d}".format(num_of_frame_beside_filing_cabinet), (400, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 255), 3)
    cv2.putText(image, "no one in the office: {:d}".format(num_of_frame_with_no_one_in_the_office), (900, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 3)
    cv2.putText(image, "the {:d}th frame".format(num_frame), (900, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    for i in range(len(exit_times)):
        cv2.putText(image, "the {:d}th time of leaving: frame {:d}".format(i + 1, exit_times[i]), (50, 500 + i * 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
    for i in range(len(enter_times)):
        cv2.putText(image, "the {:d}th time of entering: frame {:d}".format(i + 1, enter_times[i]), (600, 500 + i * 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

    cv2.imshow('original', image)
    # cv2.imshow('background_sub', background_sub)
    # cv2.imshow('cleaned_image', cleaned_image)


    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
