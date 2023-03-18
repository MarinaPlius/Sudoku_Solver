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
		self.list_of_number_pictures = self.get_list_of_number_pictures()

	def get_list_of_number_pictures(self):
		"""finds lines in the image and returns a copy of the image with found lines"""
		def reduce_lines(coordinates):
			"""eliminates line duplicates"""
			num_lines = self.sudoku_size + 1
			coordinates = sorted(list(set(coordinates)))
			while len(coordinates) > num_lines:
				element_to_remove = coordinates[0]
				smallest_distance = abs(coordinates[0] - coordinates[1])
				for i in range(len(coordinates))[1:]:
					if abs(coordinates[i] - coordinates[i-1]) < smallest_distance:
						element_to_remove = coordinates[i]
						smallest_distance = abs(coordinates[i] - coordinates[i-1])
				coordinates.remove(element_to_remove)
			return coordinates

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

		X = reduce_lines(X)
		Y = reduce_lines(Y)

		separated_pics = []
		for y in range(len(Y) -1): 
			row = []
			for x in range(len(X)-1):
				pic = self.original[Y[y]:Y[y+1], X[x]:X[x+1]]
				row.append(pic)
			separated_pics.append(row)

		return separated_pics


		


		




