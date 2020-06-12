import cv2
import numpy as np

class LaneDetection:

	def __init__(self, image):
		#input image
		self.image = image

		#Pos slope is right lane and Neg slope is left lane
		self.neg_parameters = []
		self.pos_parameters = []

		#List containing the lines corresponding to the lanes on the road
		self.lines = []

	def get_ROI(self, image):
		H, W = image.shape[:2]


				    #Left bottom,  Left top,   	   Mid-left,   		Mid-Right,  	    Right top,      Right bottom	
		vertices = np.array([[80, H-35], [80, H*0.85], [(W/2)-70, (H/2)+20], [(W/2)+30, (H/2)+20], [W-190, H*0.85], [W-190, H-35]], np.int32)

		mask = np.zeros_like(image)
		cv2.fillPoly(mask, [vertices], 255)
		masked = cv2.bitwise_and(image, mask)

		return masked

	#slope_threshold dictates the max angle of the slope desired
	#Used to remove flat lines parallel to x axis within a threshold
	def get_slope(self, lines, slope_threshold=0.1):

		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			parameters = np.polyfit((x1,x2), (y1,y2), 1)
			slope = parameters[0]
			intercept = parameters[1]

			if abs(slope) >= slope_threshold:
				if slope < 0: #left lane
					self.neg_parameters.append((slope, intercept))
				if slope > 0: #right lane
					self.pos_parameters.append((slope, intercept))

		#Takes average of all corresponding slopes to get slope a line
		pos_parameters = np.average(self.pos_parameters, axis=0)
		neg_parameters = np.average(self.neg_parameters, axis=0)

		return pos_parameters, neg_parameters

	#finds the lines for the lanes 
	def find_lines(self):

		parameters = [self.pos_parameters, self.neg_parameters]
		H, W = self.image.shape[:2]

		#y coords for line: y1 is bottom of frame, and y2 is 3/5 of frame height
		#Determines length of the lines
		y1 = H
		y2 = int(y1 * (3/4))

		for parameter in parameters:
			(slope, intercept) = parameter[0]

			#Get the corresponding x intercepts of the y coord
			#using the slope equation y=mx+b
			x1 = int((y1 - intercept) / slope)
			x2 = int((y2 - intercept) / slope)

			self.lines.append(np.array([x1, y1, x2, y2]))

		return self.lines

	def draw_lines(self):

		line_image = np.zeros_like(self.image) 

		if self.lines is not None:
			for line in self.lines:
				x1, y1, x2, y2 = line.reshape(4)
				cv2.line(line_image, (x1,y1), (x2, y2), (255,0,0), 10)

		return line_image

