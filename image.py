import cv2
import logging
import numpy as np

class Image():
	def __init__():
		self.original = None
		self.sudoku_size = 9 #standard
		self.lines = self.get_lines()
		self.number_pics = []

	def read_image(self, path):
		"""reads an image"""
		try:
			self.original = cv2.imread(path)
		except:
			logging.info("Error in reading the image!")

	def get_lines(self):
		"""finds lines in the image and returns a copy of the image with found lines"""
		# create a copy of original picture
		color_image = np.copy(self.original)
		# remove colors channel
		grey_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
		# apply threshold to distinguish lines better
		ret, thresh = cv2.threshold(grey_image,200,255,0)
		# define edges
		edges = cv2.Canny(thresh,127,255, apertureSize=3)
		# get lines from all the edges
		lines = cv2.HoughLines(edges,1,np.pi / 180,200)
		return lines
		


		




