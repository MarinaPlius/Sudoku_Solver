import cv2
import logging
import numpy as np


WHITE = 255
BLACK = 0
THRESHOLD = 200
THRESHOLD_CANNY = 127
RHO = 1
THETA = np.pi/180


class Image():
	def __init__(self, image, sudoku_size=9):
		self.original = image
		self.sudoku_size = sudoku_size

	def get_edge_coordinates(self):
		"""finds lines in the image and returns a copy of the image with found lines"""

		# create a copy of original picture
		color_image = np.copy(self.original)

		# remove colors channel
		grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

		# apply threshold to distinguish lines better
		ret, thresh = cv2.threshold(grey_image, THRESHOLD, WHITE, BLACK)

		# define edges
		edges = cv2.Canny(thresh, THRESHOLD_CANNY, WHITE)

		# get lines from all the edges
		lines = cv2.HoughLines(edges, RHO, THETA, THRESHOLD)

		# lists to store coordinates of edges
		X, Y = [], []

		# calculate coordinates
		for line in lines:
		    for rho,theta in line:
		        a = np.cos(theta)
		        b = np.sin(theta)
		        x0 = a*rho
		        y0 = b*rho
		        X.append(int(x0))
        		Y.append(int(y0))
		X = self.reduce_lines(X)
		Y = self.reduce_lines(Y)

	def reduce_lines(coordinates):
	    """eliminates line duplicates"""
	    num_lines = self.sudoku_size + 1
	    Z = sorted(list(set(Z)))
	    while len(Z) > num_lines:
	        element_to_remove = Z[0]
	        smallest_distance = abs(Z[0] - Z[1])
	        for i in range(len(Z))[1:]:
	            if distance(Z[i], Z[i-1]) < smallest_distance:
	                element_to_remove = Z[i]
	                smallest_distance = distance(Z[i], Z[i-1])
	        Z.remove(element_to_remove)
	    return Z
		


		




