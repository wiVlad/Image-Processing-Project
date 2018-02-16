# USAGE
# python track.py --video video/sample.mov

# import the necessary packages
import numpy as np
import argparse
import cv2


class DetectCamShift():
	# initialize the current frame of the video, along with the list of
	# ROI points along with whether or not this is input mode

	def detection(self, img):
		print("Trying to detect")
		# Convert the given frame to HSV color space:
		hsv_img     = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

		# Histogram calculation for thresholding(not neccesery in the real code)

		# hist        = cv.calcHist(hsv_img,[1],None,[256],[0 ,255])
		# plt.hist(img.ravel(),256,[0,256])
		# plt.plot(hist)
		# plt.show()

		# Thresholding by saturation values: (we take the saturation values to be in range [140,256])
		# TODO think how to automatically calculate the thresholding values (not really neccesery)
		lb      = np.array([0 ,140 ,0])
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
		# TODO try other kernels
		kernel = np.ones((5,5),np.uint8) 
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

		# print(circles)
		#print(np.shape(circles))

		# The loop imposed to wait until exactly 2 circles are detected
		if (np.shape(circles)) and (np.shape(circles)[1]==2):          
		#  TODO try to seperate the detection for each player individually
			circles = np.uint16(np.around(circles))
			x,y= np.zeros((2,2))
			r = np.zeros((2,1))
			circle_num = 0
			for i in circles[0, :]:
				print(i)
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
			print("X is")
			print(x)
			print("Y is")
			print(y)
			return x,y,r
	    # in case no circles is found or number of circles is different than 2 return empty cell in all fields
		else:
			x,y,r = [],[],[]
			print("Nothing!")
			return x,y,r


	def __init__(self, queue):
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-v", "--video",
			help = "path to the (optional) video file")
		args = vars(ap.parse_args())

		# grab the reference to the current frame, list of ROI
		# points and whether or not it is ROI selection mode
		self.frame = None
		self.roiPts1 = []
		self.roiPts2 = []
		self.inputMode = False
		self.Detect = False

		# if the video path was not supplied, grab the reference to the
		# camera
		if not args.get("video", False):
			camera = cv2.VideoCapture(0)

		# otherwise, load the video
		else:
			camera = cv2.VideoCapture(args["video"])

		# setup the mouse callback
		cv2.namedWindow("frame")
		#cv2.setMouseCallback("frame", self.selectROI)

		# initialize the termination criteria for cam shift, indicating
		# a maximum of ten iterations or movement by a least one pixel
		# along with the bounding box of the ROI
		termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
		
		roiBox1 = None
		roiBox2 = None

		# keep looping over the frames
		while True:
			# grab the current frame

			(grabbed, self.frame) = camera.read()
			

			# check to see if we have reached the end of the
			# video
			if not grabbed:
				break

			# if the see if the ROI has been computed
			if roiBox1 is not None and roiBox2 is not None:
				# convert the current frame to the HSV color space
				# and perform mean shift
				hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            
				backProj1 = cv2.calcBackProject([hsv], [1], roiHist1, [140, 256], 1)        
				backProj2 = cv2.calcBackProject([hsv], [1], roiHist2, [140, 256], 1)
				
				# apply cam shift to the back projection, convert the
				# points to a bounding box, and then draw them
				(r1, roiBox1) = cv2.CamShift(backProj1, roiBox1, termination)
				(r2, roiBox2) = cv2.CamShift(backProj2, roiBox2, termination)
				pts1 = np.int0(cv2.boxPoints(r1))
				pts2 = np.int0(cv2.boxPoints(r2))
				cv2.polylines(self.frame, [pts1], True, (0, 255, 0), 2)
				cv2.polylines(self.frame, [pts2], True, (0, 255, 0), 2)

				center_x1 = (pts1[0][0]+pts1[1][0]+pts1[2][0]+pts1[3][0])/4
				center_y1 = (pts1[0][1]+pts1[1][1]+pts1[2][1]+pts1[3][1])/4
				
				center_x2 = (pts2[0][0]+pts2[1][0]+pts2[2][0]+pts2[3][0])/4
				center_y2 = (pts2[0][1]+pts2[1][1]+pts2[2][1]+pts2[3][1])/4

				queue.put((1,center_x1,center_y1))
				queue.put((2,center_x2,center_y2))
				print("CamShift Player 1 X is %d, Y is %d"%(center_x1, center_y1))
				print("CamShift Player 2 X is %d, Y is %d"%(center_x2, center_y2))

			
			# show the frame and record if the user presses a key
			key = cv2.waitKey(1) & 0xFF
			cv2.imshow("frame", self.frame)

			if not self.Detect:
				orig = self.frame.copy()
				#cv2.imshow("frame", self.frame)
				x,y,r = self.detection(self.frame)
				if not (r ==[]):
					# The condition states that we wait until exactly 2 players are found.
					# Built the range of interest(roi) as the square that blocks the Hough circle.
					self.roiPts1 = [(int(x[0]-r[0]),int(y[0]-r[0])),(int(x[0]+r[0]),int(y[0]-r[0])),(int(x[0]-r[0]),int(y[0]+r[0])),(int(x[0]+r[0]),int(y[0]+r[0]))]
					self.roiPts2 = [(int(x[1] - r[1]),int( y[1] - r[1])), (int(x[1] + r[1]),int(y[1] - r[1])),(int(x[1] - r[1]),int(y[1] + r[1])),
					       (int(x[1] + r[1]), int(y[1] + r[1]))]
					self.roiPts1 = np.array(self.roiPts1)
					self.roiPts2 = np.array(self.roiPts2)


					self.roiPts1[self.roiPts1 < 0] = 0
					self.roiPts2[self.roiPts2 < 0] = 0
					print("ROI PTS")
					print(self.roiPts1)
					print(self.roiPts2)


					s1 = self.roiPts1.sum(axis = 1)
					s2 = self.roiPts2.sum(axis = 1)
					tl1 = self.roiPts1[np.argmin(s1)]
					tl2 = self.roiPts2[np.argmin(s2)]
					br1 = self.roiPts1[np.argmax(s1)]
					br2 = self.roiPts2[np.argmax(s2)]

					# grab the ROI for the bounding box and convert it
					# to the HSV color space
					roi1 = orig[tl1[1]:br1[1], tl1[0]:br1[0]]
					roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
					roi2 = orig[tl2[1]:br2[1], tl2[0]:br2[0]]
					roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

					print("ROI1")
					print(roi1)
					print("ROI2")
					print(roi2)
					# compute a HSV histogram for the ROI and store the
					# bounding box
					roiHist1 = cv2.calcHist([roi1], [1], None, [256], [140, 255], False)
					roiHist1 = cv2.normalize(roiHist1, roiHist1, 0, 255, cv2.NORM_MINMAX)
					roiBox1 = (tl1[0], tl1[1], br1[0], br1[1])
					roiHist2 = cv2.calcHist([roi2], [1], None, [256], [140, 255], False)
					roiHist2 = cv2.normalize(roiHist2, roiHist2, 0, 255, cv2.NORM_MINMAX)
					roiBox2 = (tl2[0], tl2[1], br2[0], br2[1])

					self.Detect = True


			# if the 'q' key is pressed, stop the loop
			elif key == ord("q"):
				break

			#queue.put((0,0,0))

			

		# cleanup the camera and close any open windows
		camera.release()
		cv2.destroyAllWindows()