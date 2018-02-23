import numpy as np
import argparse
import cv2
import collections

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])
cv2.namedWindow("frame")
originalFrame = None
flagFrame = 0

while True:
    (grabbed, frameTest) = camera.read()
    hsv_frame = cv2.cvtColor(frameTest, cv2.COLOR_BGR2HSV)
    bw_frame = cv2.cvtColor(frameTest, cv2.COLOR_BGR2GRAY)
    if not grabbed:
        print ("error")
        break
    lb = np.array([0, 100, 0])
    ub = np.array([10, 256, 256])
    mask = cv2.inRange(hsv_frame, lb, ub)
    lb = np.array([160, 100, 0])
    ub = np.array([179, 256, 256])
    mask1 = cv2.inRange(hsv_frame, lb, ub)
    mask = mask | mask1

    sub = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
    sub1 = sub[:, :, 1]
    #sub = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    sub = cv2.medianBlur(sub1, 11)
    kernel = np.ones((5, 5), np.uint8)
    #sub = cv2.dilate(sub, kernel, iterations = 1)
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel)
    ret, sub = cv2.threshold(sub, 50, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, 4,
                                10, param1=100, param2=50, minRadius=50, maxRadius=100)
    # #sub = cv2.adaptiveThreshold(frameTest - originalFrame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if (np.shape(circles)):
        circles = np.around(circles)
        print(circles)
        for i in circles[0, :]:
            print(i)
            cv2.circle(sub, (i[0], i[1]), i[2], (150, 150, 0), 2)
    cv2.imshow("frame", sub)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()