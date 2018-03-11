# import the necessary packages
import numpy as np
import argparse
import cv2
import collections
import imutils
import math 


WIDTH = 800
HEIGHT = 448
GESTURES_FREQ = 20
HAND_FRAME_SIZE = 100
PAD_SIZE = 15

G_WIDTH = 1300
G_HEIGHT = 600
class DetectMeanShift():
	# initialize the current frame of the video, along with the list of
	# ROI points along with whether or not this is input mode

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

		self.MedNum = 5

		self.frameR = np.zeros(
			(HAND_FRAME_SIZE*2, HAND_FRAME_SIZE*4+3*PAD_SIZE, 3), dtype='uint8')
		
		self.frameL = np.zeros(
			(HAND_FRAME_SIZE*2, HAND_FRAME_SIZE*4+3*PAD_SIZE, 3), dtype='uint8')

		self.mainFrame = np.zeros(
			(G_HEIGHT,G_WIDTH, 3), dtype='uint8')

		self.Hands = np.zeros(
			(HAND_FRAME_SIZE*4+PAD_SIZE*5, HAND_FRAME_SIZE*4+3*PAD_SIZE, 3), dtype='uint8')

		# self.GUI = np.zeros()

		globalFrameCounter = 0

		global temp_circles, j, ff, red_counter, Locate, red_undet_counter, UNDETECT_THRESH, i
		UNDETECT_THRESH = 4

		red_undet_counter = UNDETECT_THRESH 
		red_counter = 0
		temp_circles = []
		j = []
		ff = False
		Locate = False
		# if the video path was not supplied, grab the reference to the
		# camera
		camera = cv2.VideoCapture(0)
		camera.set(3, WIDTH)
		camera.set(4, HEIGHT)
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

				
				minX1 = np.max([center_x1-HAND_FRAME_SIZE, 0])
				minY1 = np.max([center_y1-HAND_FRAME_SIZE, 0])

				minX2 = np.max([center_x2-HAND_FRAME_SIZE, 0])
				minY2 = np.max([center_y2-HAND_FRAME_SIZE, 0])


				
				# cv2.imshow("Hand Right", handFrameRight)
				if(globalFrameCounter % GESTURES_FREQ == 0):
					handFrameRight = gestureFrame[minY2:center_y2 +
										HAND_FRAME_SIZE, minX2:center_x2+HAND_FRAME_SIZE]
					try:
						GestDetect, self.frameR = self.detectGesture(handFrameRight, "right")
						if(GestDetect == 1):
							queue.put((5,0,0))
						if(GestDetect == 2):
							queue.put((7,0,0))
					except expression as identifier:
						pass
					# cv2.imshow("TEST",self.frameR)
	
				if((globalFrameCounter+5) % GESTURES_FREQ == 0):	
					handFrameLeft = gestureFrame[minY1:center_y1 +
										HAND_FRAME_SIZE, minX1:center_x1+HAND_FRAME_SIZE]
					try:
						GestDetect, self.frameL = self.detectGesture(handFrameLeft, "left")
						if(GestDetect == 1):
							queue.put((6, 0, 0))
						if(GestDetect == 2):
							queue.put((8, 0, 0))
					except expression as identifier:
						pass
					# cv2.imshow("TEST", self.frameL)

				if(globalFrameCounter % 5 == 0):
					(i, newLocation) = self.red_detection(self.frame)
					if((newLocation) and (i[0]!=0)):
						(xx,yy,rr) = i
						cv2.putText(self.frame, "SENDING PKAK", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
						print("SENDING PKAK")
						print(i)
						queue.put((3,xx,yy))
					if((not newLocation) and (i[0]!=0)):
						cv2.putText(self.frame, "SENDING PKAK", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
						print("REMOVE PKAK")
						queue.put((3,0,0))

				if self.Detect:
					Padding = np.zeros((PAD_SIZE, self.frameL.shape[1], 3), dtype='uint8')
					PaddingT1 = np.zeros((PAD_SIZE*2, self.frameL.shape[1], 3), dtype='uint8')
					PaddingT2 = np.zeros((PAD_SIZE*2, self.frameL.shape[1], 3), dtype='uint8')
					cv2.putText(PaddingT1, "Left Hand:", (50, 25),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200))
					cv2.putText(PaddingT2, "Right Hand:", (50, 25),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200))
					self.Hands = np.vstack(
						(PaddingT1, self.frameL, PaddingT2, self.frameR, Padding))
					# cv2.imshow("TEST", self.Hands)




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

				PLayer1IsStatic = True
				PLayer2IsStatic = True
				for i in range(0, len(self.MedQueueX1)):
					if(self.MedQueueX1[i] != center_x1 or self.MedQueueY1[i] != center_y1):
						PLayer1IsStatic = False
					if (self.MedQueueX2[i] != center_x2 or self.MedQueueY2[i] != center_y2):
						PLayer2IsStatic = False
						if (not PLayer1IsStatic) and (not PLayer2IsStatic):
							break
							
				# if PLayer2IsStatic or PLayer1IsStatic:
				# 	print("either player 1 or player 2 are static - detection is needed")
				# else:
				# 	print("detection not needed")
				# either player 1 or player 2 are static - detection is needed
				self.Detect = (not (PLayer1IsStatic or PLayer2IsStatic))
				# print(self.Detect)
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
			# cv2.imshow("frame", self.frame)

			VPad = np.zeros(
				(G_HEIGHT - self.Hands.shape[0], self.Hands.shape[1], 3), dtype='uint8')
			frameL = np.vstack((self.Hands,VPad))
			
			VPad = np.zeros((G_HEIGHT - self.frame.shape[0], self.frame.shape[1], 3), dtype='uint8')
			frameR= np.vstack((self.frame,VPad))

			frameF = np.hstack((frameL,frameR))
			# cv2.imshow("frame1", self.mainFrame)
			cv2.imshow("frame1", frameF)

			if not self.Detect:
				print("")
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
			print("Detected!!")
			return x, y, r

		# in case no circles is found or number of circles is different than 2 return empty cell in all fields
		else:
			x, y, r = [], [], []
			return x, y, r

	def detectGesture(self, img, player, verbose=False):
		# Rotate image
		if(player == "right"):
			img = imutils.rotate(img, -90)
			returnID = 1
		elif(player == "left"):
			img = imutils.rotate(img, 90)
			returnID = 2

		# print(img.shape)

		# Padding if needed:
		if(img.shape[0] != HAND_FRAME_SIZE*2):
			PadSize = 2*HAND_FRAME_SIZE - img.shape[0]
			Padding = np.zeros((PadSize, img.shape[1], 3), dtype='uint8')
			img = np.vstack((img,Padding))

		if(img.shape[1] != HAND_FRAME_SIZE*2):
			PadSize = 2*HAND_FRAME_SIZE - img.shape[1]
			Padding = np.zeros((img.shape[0], PadSize, 3), dtype='uint8')
			img = np.hstack((img,Padding))

		# convert to grayscale
		grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# applying gaussian blur
		value = (5, 5)
		blurred = cv2.GaussianBlur(grey, value, 0)
		if verbose:
			cv2.imshow('Blurred', blurred)

		# thresholdin: Otsu's Binarization method
		_, thresh = cv2.threshold(
			blurred, 215, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		if verbose:
			cv2.imshow('Thresholded', thresh)

		# find contour with max area
		# Returns array with (x,y) tuples of contour outline 
		image, contours, hierarchy = cv2.findContours(
			thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cnt = max(contours, key=lambda x: cv2.contourArea(x))

		# create bounding rectangle around the contour
		# x, y, w, h = cv2.boundingRect(cnt)
		# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 0)

		# finding convex hull
		hull = cv2.convexHull(cnt)

		# drawing contours
		drawing = np.zeros(img.shape, np.uint8)
		cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
		cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

		# finding convex hull
		hull = cv2.convexHull(cnt, returnPoints=False)

		# finding convexity defects
		defects = cv2.convexityDefects(cnt, hull)
		count_defects = 0
		cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)

		# applying Cosine Rule to find angle for all defects (between fingers)
		# with angle > 90 degrees and ignore defects
		# TODO ignore outlier defects
		if (defects is not None):
			for i in range(defects.shape[0]):
				s, e, f, d = defects[i, 0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])

				# Calculate middle of base
				mX = int((cnt[s][0][0]+cnt[e][0][0])/2)
				mY = int((cnt[s][0][1]+cnt[e][0][1])/2)
				middle = tuple((mX, mY))

				# find length of all sides of triangle
				a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
				b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
				c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

				# apply cosine rule and get orientation of the defect
				dX, dY = (mX - far[0], mY - far[1])
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
				alpha = math.atan2(dX, dY)
				dist = math.sqrt(dX**2+dY**2)
				verbose = True
				# ignore angles > 90 and orientations not pointing "Up"
				if (angle <= 90) and ((alpha < -2.3) or (alpha > 2.3)) and (dist > 30) and (dist < 70):
					count_defects += 1
					cv2.circle(img, far, 1, [0, 0, 255], -1)
					# draw a line from start to end i.e. the convex points (finger tips)
					cv2.line(img, start, end, [200, 255, 100], 2)
					if verbose:
						cv2.line(img, middle, far, [10, 200, 100], 2)
						str = "{}".format(alpha)
						cv2.putText(img, str, middle,
											cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

			Padding = np.zeros((img.shape[0], PAD_SIZE, 3), dtype='uint8')
			all_img = np.hstack((Padding, drawing, Padding, img,Padding))
			# print("DRAWIND SHAPE:")
			# print(all_img.shape)
			# if verbose:
				# cv2.imshow('Contours', all_img)

			# define actions required
			if count_defects == 2:
				cv2.putText(all_img, "Getting 3 Fingers", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
				return 2, all_img
			elif count_defects == 3:
				cv2.putText(all_img, "Getting 4 Fingers", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
				return 1, all_img
			else:
				cv2.putText(all_img, "NONE", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
				return 0, all_img

			# show appropriate images in windows
			# cv2.imshow('Gesture', img)
			# all_img = np.hstack((drawing, img))

			# if count_defects == 3:
			# elif count_defects == 2:
			# else:

	def red_detection(self, frame):
		global ff, Locate, j, temp_circles, red_counter, red_undet_counter, i
		Flag = 0
		THRESH = 30  # Object movement detection threshold
		FRAME_THRESH = 3

		img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# red lower mask (0-10)
		lower_red = np.array([0, 50, 50])
		upper_red = np.array([3, 255, 255])
		# TODO adjust the red mask values
		mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
		# red upper mask (170-180)
		lower_red = np.array([172, 50, 50])
		upper_red = np.array([179, 255, 255])
		mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
		# join masks
		maskRed = mask0 + mask1

		# or your HSV image, which I believe is what you want
		output_hsv = img_hsv.copy()
		output_hsv[np.where((maskRed == 0))] = 0
		mod_frame = output_hsv[:, :, 1]  # TODO choose the best color channel
		mod_frame = np.uint8(mod_frame)

		# TODO calibrate the binary threshold
		ret, med = cv2.threshold(mod_frame, 170, 255, cv2.THRESH_BINARY)
		circles = cv2.HoughCircles(
			med, cv2.HOUGH_GRADIENT, 4, 100, param1=100, param2=70, minRadius=4, maxRadius=15)

		if (np.shape(circles)) and (np.shape(circles)[1] == 1):
			circles = np.uint16(np.around(circles))
			i = circles[0, 0, :]

			cv2.circle(med, (i[0], i[1]), i[2], (0, 255, 0), 2)
			cv2.circle(med, (i[0], i[1]), 2, (0, 0, 255), 3)

			if ff:
				if Locate:
					if (abs(np.int8(i[0] - j[0])) > THRESH) or (abs(np.int8(i[1] - j[1])) > THRESH):
						red_undet_counter -= 1
					if red_undet_counter < 0:
						Locate = False
						red_counter = 0
						print("Falge is AP!")
						Flag = 1
				else:
					if not (abs(np.int8(i[0]-j[0])) > THRESH or abs(np.int8(i[1]-j[1])) > THRESH):
						red_counter += 1
						print('xxx')
					if red_counter > FRAME_THRESH:
						Locate = True
						red_undet_counter = UNDETECT_THRESH
						Flag = 1
						print("Falge is AP!")

		else:
			if Locate:
				if not(np.shape(circles)):
					red_undet_counter -= 1
				if red_undet_counter < 0:
					Locate = False
					Flag = 1
					print("Falge is AP!")
					red_counter = 0

		if (np.shape(circles)):
			temp_circles = np.uint16(np.around(circles))
			j = temp_circles[0, 0, :]
			ff = True

		if Flag:
			return(i, Locate)
		else:
			return((0, 0, 0), Locate)
