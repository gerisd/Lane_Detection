from LaneDetection import LaneDetection
import cv2
import numpy as np

image = r"./Images/rd2.jpg"

img = cv2.imread(image)

#Get height and width of image
H, W = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
canny = cv2.Canny(blur, threshold1=50, threshold2=150)

#Threshold image
retval, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)      
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Initalize Lane Detection object
lanedetection = LaneDetection(img)

#Get ROI
imgROI = lanedetection.get_ROI(canny)

#Find lines in ROI
lines = cv2.HoughLinesP(imgROI, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 

#Find average line parameters and draw them to image
pos_slope, neg_slope = lanedetection.get_slope(lines)
image_lines = lanedetection.find_lines()
lanes = lanedetection.draw_lines()

#combine lanes with original image
laned_image = cv2.addWeighted(img, 0.8, lanes, 1, 1)

#Display
#cv2.imshow('Original Image', img)
cv2.imshow('Final Image', laned_image)

key = cv2.waitKey(0) & 0xFF

if key == ord('q'):
	cv2.destroyAllWindows()
