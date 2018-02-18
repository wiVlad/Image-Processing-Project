
# import the necessary packages
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roiPts1 = []
roiPts2 = []
Detect = False




def detection(img):

    # Convert the given frame to HSV color space:
    hsv_img     = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Histogram calculation for thresholding(not neccesery in the real code)

    # hist        = cv.calcHist(hsv_img,[1],None,[256],[0 ,255])
    # plt.hist(img.ravel(),256,[0,256])
    # plt.plot(hist)
    # plt.show()

    # Thresholding by saturation values: (we take the saturation values to be in range [140,256])

    lb      = np.array([0 ,140 ,0])                                                                                     # TODO think how to automatically calculate the thresholding values (not really neccesery)
    ub      = np.array([179, 256 , 256])
    mask    = cv2.inRange(hsv_img,lb,ub)
    res     = cv2.bitwise_and(hsv_img,hsv_img,mask = mask)
    res_sat = res[:,:,1]                                  # isolate the saturation dimension
    #plt.imshow(res[:,:,1])
    #plt.show()

    # Otsu Thresholding:

    ret,mask2 = cv2.threshold(res_sat,0,255,cv2.THRESH_OTSU)
    #plt.imshow(mask2+hsv_img_sat)
    #plt.show()

    # Erosion & Dilation:

    kernel = np.ones((5,5),np.uint8)                                                                                    # TODO try other kernels
    open_img = cv2.morphologyEx(mask2,cv2.MORPH_CLOSE, kernel)
    #plt.imshow(open_img)
    #plt.show()


    # Median filtering:

    filt = cv2.medianBlur(open_img,11)
    #plt.imshow(filt)
    #plt.show()

    # Hough Transform

    circles     = cv2.HoughCircles(filt, cv2.HOUGH_GRADIENT, 4, 500, param1=100, param2=50, minRadius=30, maxRadius=100)
                                                                                                                        # TODO calibrate these parameters when using the real system

    print(circles)
    print(np.shape(circles))


    if (np.shape(circles)) and (np.shape(circles)[1]==2):          # The loop imposed to wait until exactly 2 circles are detected
                                                                                                                        #  TODO try to seperate the detection for each player individually
        circles = np.uint16(np.around(circles))
        x,y         = np.zeros((2,2))

        r			= np.zeros((2,1))
        circle_num = 0
        for i in circles[0, :]:

            x[circle_num] = i[0]
            y[circle_num] = i[1]
            r[circle_num] = i[2]
            # We assume the x coordinate of player 1 is smaller than player 2 so if the order is opposite we swap the values.
            if circle_num > 0:
                if x[circle_num] < x[circle_num - 1]:
                    x[circle_num] = x[circle_num-1]
                    y[circle_num] = y[circle_num-1]
                    r[circle_num] = r[circle_num - 1]
                    x[circle_num-1] = i[0]
                    y[circle_num-1] = i[1]
                    r[circle_num-1] = i[2]
            circle_num   += 1
            # draw the outer circle
            #cv.circle(filt, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            #cv.circle(filt, (i[0], i[1]), 2, (0, 0, 255), 3)

        #cv.imshow('detected circles',filt)
        print("number of circles")
        print(circle_num)
        return x,y,r
    else:                          # in case no circles is found or number of circles is different than 2 return empty cell in all fields
        x,y,r = [],[],[]
        return x,y,r


def main():


    global frame,roiPts1,roiPts2, Detect

    camera = cv2.VideoCapture(0)
    cv2.namedWindow("frame")

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

                                                                                                                        # TODO calibrate the termination criteria
    roiBox1 = None
    roiBox2 = None

    # keep looping over the frames
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # see if the ROI has been computed
        if roiBox1 is not None and roiBox2 is not None:
            # convert the current frame to the HSV color space
            # and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            backProj1 = cv2.calcBackProject([hsv], [1], roiHist1, [140, 256], 1)        
            backProj2 = cv2.calcBackProject([hsv], [1], roiHist2, [140, 256], 1)

            # apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them
            (r1, roiBox1) = cv2.CamShift(backProj1, roiBox1, termination)
            (r2, roiBox2) = cv2.CamShift(backProj2, roiBox2, termination)

            pts1 = np.int0(cv2.boxPoints(r1))
            pts2 = np.int0(cv2.boxPoints(r2))
            cv2.polylines(frame, [pts1], True, (0, 255, 0), 2)
            cv2.polylines(frame, [pts2], True, (0, 255, 0), 2)







        # show the frame and record if the user presses a key
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF



        if not Detect :


            orig = frame.copy()
            cv2.imshow("frame", frame)
            x,y,r = detection(frame)
            if not (r ==[]):     # The condition states that we wait until exactly 2 players are found.
                # Built the range of interest(roi) as the square that blocks the Hough circle.
                roiPts1 = [(int(x[0]-r[0]),int(y[0]-r[0])),(int(x[0]+r[0]),int(y[0]-r[0])),(int(x[0]-r[0]),int(y[0]+r[0])),(int(x[0]+r[0]),int(y[0]+r[0]))]
                roiPts2 = [(int(x[1] - r[1]),int( y[1] - r[1])), (int(x[1] + r[1]),int(y[1] - r[1])),(int(x[1] - r[1]),int(y[1] + r[1])),
                       (int(x[1] + r[1]), int(y[1] + r[1]))]
                roiPts1 = np.array(roiPts1)
                roiPts2 = np.array(roiPts2)

                s1 = roiPts1.sum(axis = 1)
                s2 = roiPts2.sum(axis = 1)
                tl1 = roiPts1[np.argmin(s1)]
                tl2 = roiPts2[np.argmin(s2)]
                br1 = roiPts1[np.argmax(s1)]
                br2 = roiPts2[np.argmax(s2)]

                # grab the ROI for the bounding box and convert it
                # to the HSV color space
                roi1 = orig[tl1[1]:br1[1], tl1[0]:br1[0]]
                roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
                roi2 = orig[tl2[1]:br2[1], tl2[0]:br2[0]]
                roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

                # compute a HSV histogram for the ROI and store the
                # bounding box
                roiHist1 = cv2.calcHist([roi1], [1], None, [128], [140, 256])
                roiHist1 = cv2.normalize(roiHist1, roiHist1, 0, 255, cv2.NORM_MINMAX)
                roiBox1 = (tl1[0], tl1[1], br1[0], br1[1])
                roiHist2 = cv2.calcHist([roi2], [1], None, [128], [140, 256])
                roiHist2 = cv2.normalize(roiHist2, roiHist2, 0, 255, cv2.NORM_MINMAX)
                roiBox2 = (tl2[0], tl2[1], br2[0], br2[1])

                Detect = True


        # if the 'q' key is pressed, stop the loop
        elif key == ord("q"):
            break


    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


main()

