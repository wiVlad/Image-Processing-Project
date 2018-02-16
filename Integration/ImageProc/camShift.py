# USAGE
# python track.py --video video/sample.mov

# import the necessary packages
import numpy as np
import argparse
import cv2


class CamShift():
	# initialize the current frame of the video, along with the list of
	# ROI points along with whether or not this is input mode


	def selectROI(self, event, x, y, flags, param):
		# grab the reference to the current frame, list of ROI
		# points and whether or not it is ROI selection mode
		#global frame, roiPts, inputMode

		# if we are in ROI selection mode, the mouse was clicked,
		# and we do not already have four points, then update the
		# list of ROI points with the (x, y) location of the click
		# and draw the circle
		if self.inputMode and event == cv2.EVENT_LBUTTONDOWN and (len(self.roiPts1) < 4 or len(self.roiPts2) < 4) :
			if len(self.roiPts1)>=4:
				self.roiPts2.append((x, y))
				cv2.circle(self.frame, (x, y), 4, (0, 255, 0), 2)
				cv2.imshow("frame", self.frame)
			else: 
				self.roiPts1.append((x, y))
				cv2.circle(self.frame, (x, y), 4, (0, 255, 0), 2)
				cv2.imshow("frame", self.frame)

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
		Detect = False

		# if the video path was not supplied, grab the reference to the
		# camera
		if not args.get("video", False):
			camera = cv2.VideoCapture(0)

		# otherwise, load the video
		else:
			camera = cv2.VideoCapture(args["video"])

		# setup the mouse callback
		cv2.namedWindow("frame")
		cv2.setMouseCallback("frame", self.selectROI)

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
				backProj1 = cv2.calcBackProject([hsv], [0], roiHist1, [0, 180], 1)
				backProj2 = cv2.calcBackProject([hsv], [0], roiHist2, [0, 180], 1)

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
			cv2.imshow("frame", self.frame)
			key = cv2.waitKey(1) & 0xFF

			# handle if the 'i' key is pressed, then go into ROI
			# selection mode
			if key == ord("x") and (len(self.roiPts1) < 4 or len(self.roiPts12) < 4) :
				# indicate that we are in input mode and clone the
				# frame
				self.inputMode = True
				orig = self.frame.copy()

				# keep looping until 4 reference ROI points have
				# been selected; press any key to exit ROI selction
				# mode once 4 points have been selected
				while len(self.roiPts1) < 4:
					cv2.imshow("frame", self.frame)
					cv2.waitKey(0)
				while len(self.roiPts2)<4:
					cv2.imshow("frame", self.frame)
					cv2.waitKey(0)

				self.roiPts1 = np.array(self.roiPts1)
				self.roiPts2 = np.array(self.roiPts2)
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

				#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

				# compute a HSV histogram for the ROI and store the
				# bounding box
				roiHist1 = cv2.calcHist([roi1], [0], None, [64], [0, 180])
				roiHist1 = cv2.normalize(roiHist1, roiHist1, 0, 255, cv2.NORM_MINMAX)
				roiBox1 = (tl1[0], tl1[1], br1[0], br1[1])
				roiHist2 = cv2.calcHist([roi2], [0], None, [64], [0, 180])
				roiHist2 = cv2.normalize(roiHist2, roiHist2, 0, 255, cv2.NORM_MINMAX)
				roiBox2 = (tl2[0], tl2[1], br2[0], br2[1])


			# if the 'q' key is pressed, stop the loop
			elif key == ord("q"):
				break

			#queue.put((0,0,0))

			

		# cleanup the camera and close any open windows
		camera.release()
		cv2.destroyAllWindows()