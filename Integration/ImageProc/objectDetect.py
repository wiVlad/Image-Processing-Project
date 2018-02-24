import cv2
import numpy as np
from collections import deque


camera = cv2.VideoCapture(0)
cv2.namedWindow("frame")
frame_stack = []
max_frames = 1         # Number of frames to mediate
j = 0
Detect = False

while True:
    if Detect:
        cv2.waitKey(1000)
        Detect = False

    # grab the current frame
    (grabbed, frame) = camera.read()

    cv2.waitKey(1)

    if not grabbed:
        break
    #cv2.imshow('frame',frame)
    j += 1
    img_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([5,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)              # TODO adjust the red mask values
    # red upper mask (170-180)
    lower_red = np.array([172,50,50])
    upper_red = np.array([179,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # join masks
    maskRed = mask0 + mask1

    # green mask (55-65)
    lower_green = np.array([40, 128, 0])
    upper_green = np.array([70, 255, 127])
    maskGreen = cv2.inRange(img_hsv, lower_green, upper_green)  # TODO adjust the green mask values

    # blue lower mask (107-117)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([120, 255, 255])
    maskBlue = cv2.inRange(img_hsv, lower_blue, upper_blue)  # TODO adjust the blue mask values

    # set my output img to zero everywhere except my mask
    output_img = frame.copy()
    output_img[np.where((maskRed==0)&(maskGreen==0)&(maskBlue==0))] = 0

    # or your HSV image, which I believe is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where((maskRed==0)&(maskGreen==0)&(maskBlue==0))] = 0
    mod_frame = output_hsv[:,:,1]

    if j>max_frames:
        frame_stack.append(mod_frame)
        med = np.median(frame_stack,axis=0)
        med = np.uint8(med)
        ret, med = cv2.threshold(med, 100, 255, cv2.THRESH_BINARY)     # TODO calibrate the binary threshold

        circles = cv2.HoughCircles(med, cv2.HOUGH_GRADIENT, 4, 800, param1=100, param2=100, minRadius=7, maxRadius=50)
        print(circles)
        if np.shape(circles): #TODO and (np.shape(circles)[1] == 1):
            Detect = True
            circles = np.uint16(np.around(circles))
            print(circles)
            for i in circles[0, :]:
                # draw the outer circle
                 cv2.circle(med, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                 cv2.circle(med, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imshow('frame', med)
        cv2.waitKey(1)
        frame_stack =[]
        j = 0
    frame_stack.append(mod_frame)

camera.release()
cv2.destroyAllWindows()