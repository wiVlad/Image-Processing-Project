# USAGE
# python track.py --video video/sample.mov

# import the necessary packages
import numpy as np
import argparse
import cv2
import collections
import imutils
import math 


class DetectMeanShift():
	# initialize the current frame of the video, along with the list of
	# ROI points along with whether or not this is input mode

	def detection(self, img):
		print("Trying to detect")
		# Convert the given frame to HSV color space:
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Histogram calculation for thresholding(not neccesery in the real code)

		# hist        = cv.calcHist(hsv_img,[1],None,[256],[0 ,255])
		# plt.hist(img.ravel(),256,[0,256])
		# plt.plot(hist)
		# plt.show()

		# Thresholding by saturation values: (we take the saturation values to be in range [140,256])
		# TODO think how to automatically calculate the thresholding values (not really neccesery)
		lb = np.array([0, 100, 0])
		ub = np.array([50, 256, 128])
		mask = cv2.inRange(hsv_img, lb, ub)
		res = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
		# isolate the saturation dimension
		res_sat = res[:, :, 1]
		# plt.imshow(res[:,:,1])
		# plt.show()

		# Otsu Thresholding:

		ret, mask2 = cv2.threshold(res_sat, 0, 255, cv2.THRESH_OTSU)
		# plt.imshow(mask2+hsv_img_sat)
		# plt.show()

		# Erosion & Dilation:
		# TODO try other kernels
		kernel = np.ones((5, 5), np.uint8)
		open_img = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
		# plt.imshow(open_img)
		# plt.show()

		# Median filtering:

		filt = cv2.medianBlur(open_img, 11)
		# plt.imshow(filt)
		# plt.show()

		# Hough Transform

		circles = cv2.HoughCircles(filt, cv2.HOUGH_GRADIENT, 4,
								   500, param1=100, param2=50, minRadius=30, maxRadius=50)
		# TODO calibrate these parameters when using the real system

		# print(circles)
		# print(np.shape(circles))

		# The loop imposed to wait until exactly 2 circles are detected
		if (np.shape(circles)) and (np.shape(circles)[1] == 2):
			#  TODO try to seperate the detection for each player individually
			circles = np.uint16(np.around(circles))
			x, y = np.zeros((2, 2))
			r = np.zeros((2, 1))
			circle_num = 0
			for i in circles[0, :]:
				# print(i)
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
				circle_num += 1
				# draw the outer circle
				#cv.circle(filt, (i[0], i[1]), i[2], (0, 255, 0), 2)
				# draw the center of the circle
				#cv.circle(filt, (i[0], i[1]), 2, (0, 0, 255), 3)
						#cv.imshow('detected circles',filt)

			return x, y, r
		# in case no circles is found or number of circles is different than 2 return empty cell in all fields
		else:
			x, y, r = [], [], []
			print("Nothing!")
			return x, y, r


	def detectGesture(self, crop_img, player):

		if(player == "right"):
			crop_img = imutils.rotate(crop_img, -90)
			returnID = 1
		if(player == "left"):
			crop_img = imutils.rotate(crop_img, 90)
			returnID = 2
		
		# convert to grayscale
		grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

		# applying gaussian blur
		value = (5, 5)
		blurred = cv2.GaussianBlur(grey, value, 0)
		# cv2.imshow('blurred', blurred)

		# thresholdin: Otsu's Binarization method
		_, thresh1 = cv2.threshold(blurred, 215, 255,
								cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

		# show thresholded image
		# cv2.imshow('Thresholded', thresh1)

		image, contours, hierarchy = cv2.findContours(thresh1.copy(),
													cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		# # find contour with max area
		cnt = max(contours, key=lambda x: cv2.contourArea(x))

		# create bounding rectangle around the contour (can skip below two lines)
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

		# finding convex hull
		hull = cv2.convexHull(cnt)

		# drawing contours
		drawing = np.zeros(crop_img.shape, np.uint8)
		cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
		cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

		# finding convex hull
		hull = cv2.convexHull(cnt, returnPoints=False)

		# finding convexity defects
		defects = cv2.convexityDefects(cnt, hull)
		count_defects = 0
		cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

		# applying Cosine Rule to find angle for all defects (between fingers)
		# with angle > 90 degrees and ignore defects
		if (defects is not None):
			for i in range(defects.shape[0]):
				s, e, f, d = defects[i, 0]

				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])

				# find length of all sides of triangle
				a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
				b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
				c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

				# apply cosine rule here
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

				# ignore angles > 90 and highlight rest with red dots
				if angle <= 90:
					count_defects += 1
					cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
				#dist = cv2.pointPolygonTest(cnt,far,True)

				# draw a line from start to end i.e. the convex points (finger tips)
				# (can skip this part)
				cv2.line(crop_img, start, end, [0, 255, 0], 2)
				#cv2.circle(crop_img,far,5,[0,0,255],-1)

			all_img = np.hstack((drawing, crop_img))
			# define actions required
			if count_defects == 1:
				cv2.putText(all_img, "NONE", (30, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200))
			if count_defects == 2:
				cv2.putText(all_img, "Getting 3 Fingers", (30, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200))
			elif count_defects == 3:
				cv2.putText(all_img, "Getting 4 Fingers", (30, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200))
			else:
				# print("NONE")
				cv2.putText(all_img, "NONE", (30, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200))

			# show appropriate images in windows
			# cv2.imshow('Gesture', img)
			# all_img = np.hstack((drawing, crop_img))
			cv2.imshow('Contours', all_img)
			if count_defects == 3:
				return 1
			elif count_defects == 2:
				return 2
			else: 
				return 0

	def __init__(self, queue):

		# grab the reference to the current frame, list of ROI
		# points and whether or not it is ROI selection mode
		self.frame = None
		self.roiPts1 = []
		self.roiPts2 = []
		self.inputMode = False
		self.Detect = False
		self.fist_radius_add = 50

		self.MedQueueX1 = collections.deque([])
		self.MedQueueX2 = collections.deque([])
		self.MedQueueY1 = collections.deque([])
		self.MedQueueY2 = collections.deque([])

		self.MedNum = 9


		globalFrameCounter = 0
		# if the video path was not supplied, grab the reference to the
		# camera
		camera = cv2.VideoCapture(0)
		camera.set(3, 800)
		camera.set(4, 448)
		# setup the mouse callback
		# cv2.namedWindow("frame")
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
			globalFrameCounter += 1
			# check to see if we have reached the end of the
			# video
			if not grabbed:
				break

			timer = cv2.getTickCount()
			#  if the ROI has been computed
			if roiBox1 is not None and roiBox2 is not None:
				# convert the current frame to the HSV color space
				# and perform mean shift

				gestureFrame = self.frame.copy() 

				hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

				backProj1 = cv2.calcBackProject(
					[hsv], [1], roiHist1, [140, 256], 1)
				backProj2 = cv2.calcBackProject(
					[hsv], [1], roiHist2, [140, 256], 1)

				# apply cam shift to the back projection, convert the
				# points to a bounding box, and then draw them
				(r1, roiBox1) = cv2.meanShift(backProj1, roiBox1, termination)
				(r2, roiBox2) = cv2.meanShift(backProj2, roiBox2, termination)

				x1, y1, w1, h1 = roiBox1
				x2, y2, w2, h2 = roiBox2

				cv2.rectangle(self.frame, (x1, y1), (x1+w1, y1+h1), 255, 2)
				cv2.rectangle(self.frame, (x2, y2), (x2+w2, y2+h2), 255, 2)

				center_x1 = round(x1 + w1/2)
				center_y1 = round(y1 + h1/2)

				center_x2 = round(x2 + w2/2)
				center_y2 = round(y2 + h2/2)

				frameSize = 125
				minX1 = np.max([center_x1-frameSize, 0])
				minY1 = np.max([center_y1-frameSize, 0])

				minX2 = np.max([center_x2-frameSize, 0])
				minY2 = np.max([center_y2-frameSize, 0])

				handFrameLeft = gestureFrame[minY1:center_y1 +
                                    frameSize, minX1:center_x1+frameSize]

				minX2 = np.max([x2-50, 0])
				minY2 = np.max([y2-50,0])
				handFrameRight = gestureFrame[minY2:center_y2 +
                                    frameSize, minX2:center_x2+frameSize]

				
				# cv2.imshow("Hand Right", handFrameRight)
				if(globalFrameCounter % 20 == 0):
					if(self.detectGesture(handFrameRight, "right") == 1):
						queue.put((5,0,0))
					if(self.detectGesture(handFrameRight, "right") == 2):
						queue.put((7,0,0))

				if((globalFrameCounter+5)% 20 == 0):
					if(self.detectGesture(handFrameLeft, "left") == 1):
						queue.put((6,0,0))
					if(self.detectGesture(handFrameLeft, "left") == 2):
						queue.put((8,0,0))

				# if(globalFrameCounter % 20 == 0):
				# 	if(self.detectGesture(handFrameLeft, "left")):
				# 		queue.put((6,0,0))


				self.MedQueueX1.append(center_x1)
				self.MedQueueX2.append(center_x2)
				self.MedQueueY1.append(center_y1)
				self.MedQueueY2.append(center_y2)

				med_x1 = np.median(self.MedQueueX1)
				med_x2 = np.median(self.MedQueueX2)
				med_y1 = np.median(self.MedQueueY1)
				med_y2 = np.median(self.MedQueueY2)

				queue.put((1, med_x1, med_y1))
				queue.put((2, med_x2, med_y2))
				if len(self.MedQueueX1) == self.MedNum:
					self.MedQueueX1.popleft()
					self.MedQueueX2.popleft()
					self.MedQueueY1.popleft()
					self.MedQueueY2.popleft()

				# queue.put((1, med_x1, center_y1))
				# queue.put((2, med_x2, center_y2))
				# print("CamShift Player 1 X is %d, Y is %d" %
				# 	  (center_x1, center_y1))
				# print("CamShift Player 2 X is %d, Y is %d" %
				# 	  (center_x2, center_y2))

			# show the frame and record if the user presses a key
			

			# Calculate Frames per second (FPS)
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			cv2.putText(self.frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

			key = cv2.waitKey(1) & 0xFF
			cv2.imshow("frame", self.frame)

			if not self.Detect:
				orig = self.frame.copy()
				x, y, r = self.detection(self.frame)
				if not (r == []):
					r += self.fist_radius_add
					# The condition states that we wait until exactly 2 players are found.
					# Built the range of interest(roi) as the square that blocks the Hough circle.
					self.roiPts1 = [(int(x[0]-r[0]), int(y[0]-r[0])), (int(x[0]+r[0]), int(y[0]-r[0])),
									(int(x[0]-r[0]), int(y[0]+r[0])), (int(x[0]+r[0]), int(y[0]+r[0]))]
					self.roiPts2 = [(int(x[1] - r[1]), int(y[1] - r[1])), (int(x[1] + r[1]), int(y[1] - r[1])), (int(x[1] - r[1]), int(y[1] + r[1])),
									(int(x[1] + r[1]), int(y[1] + r[1]))]
					self.roiPts1 = np.array(self.roiPts1)
					self.roiPts2 = np.array(self.roiPts2)

					# maxY = np.shape(orig)(3)

					self.roiPts1[self.roiPts1 < 0] = 0
					self.roiPts2[self.roiPts2 < 0] = 0

					[Y_size, X_size, _] = np.shape(orig)

					self.roiPts1[:, 0][self.roiPts1[:, 0] > X_size] = X_size
					self.roiPts1[:, 1][self.roiPts1[:, 1] > Y_size] = Y_size

					self.roiPts2[:, 0][self.roiPts2[:, 0] > X_size] = X_size
					self.roiPts2[:, 1][self.roiPts2[:, 1] > Y_size] = Y_size

					s1 = self.roiPts1.sum(axis=1)
					s2 = self.roiPts2.sum(axis=1)
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

					# compute a HSV histogram for the ROI and store the
					# bounding box
					roiHist1 = cv2.calcHist([roi1], [1], None, [256], [140, 255], False)
					roiHist1 = cv2.normalize(roiHist1, roiHist1, 0, 255, cv2.NORM_MINMAX)
					roiBox1 = (tl1[0], tl1[1], int(r[0]), int(r[0]))
					roiHist2 = cv2.calcHist([roi2], [1], None, [256], [140, 255], False)
					roiHist2 = cv2.normalize(roiHist2, roiHist2, 0, 255, cv2.NORM_MINMAX)
					roiBox2 = (tl2[0], tl2[1], int(r[1]), int(r[1]))

					self.Detect = True


			# if the 'q' key is pressed, stop the loop
			elif key == ord("q"):
				break


		# cleanup the camera and close any open windows
		camera.release()
		cv2.destroyAllWindows()
