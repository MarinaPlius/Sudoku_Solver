import cv2
import logging
import numpy as np
from copy import deepcopy
from tensorflow.keras.models import load_model
import pytesseract


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
		self.empty_cells = self.find_empty_cells()
	
	def reduce_lines(self, coordinates):
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

	def get_list_of_number_pictures(self):
		"""finds lines in the image
		returns list of cropped image, where each element is an image of an empty cell or a number"""

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

		separated_pics = []
		for y in range(len(Y) -1): 
			row = []
			for x in range(len(X)-1):
				pic = self.original[Y[y]:Y[y+1], X[x]:X[x+1]]
				row.append(pic)
			separated_pics.append(row)

		return separated_pics
	
	def preprocess_cell(self, img, mnist=False, resize=False, clean_remains=False):
		"""
		cleans a number image
		"""
		if mnist:
			# remove colors channel
			img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			threshold = 117
			max_value= 255

			# inverse to black
			img = 255 - img
			ret1, img = cv2.threshold(img, threshold, max_value,
				cv2.THRESH_BINARY)
			
		if resize:
			img = cv2.resize(img,(28,28),interpolation = cv2.INTER_CUBIC)

		# clean remains of lines
		if clean_remains:
			for i in range(28):
				for j in range(5):
					if mnist:
						img[i][j] = 0 # top of the image
					else:
						img[i][j] = 255
			for i in range(5):
				for j in range(28):
					if mnist:
						img[i][j] = 0 # left side
					else:
						img[i][j] = 255

		return img

	def find_empty_cells(self, inverse=False):
		"""
		finds cells which do not have a number in them and fills them with 0s
		"""
		if inverse:
			black, white = 255, 0
		else:
			black, white = 0, 255
		
		# create empty ndarray
		cells = np.ones(shape=(self.sudoku_size - 1, self.sudoku_size - 1))

		for i in range(self.sudoku_size - 1):
			for j in range(self.sudoku_size - 1):
				if np.sum((self.list_of_number_pictures[i][j]) == black) < 10:
					cells[i][j] = 0

		return cells


	def use_mnist_model(self):
		"""
		predicts which number is on the image by using a model trained on MNIST dataset
		returns a matrix of sudoku size with integers
		"""

		# load the model
		number_recognizer_MNIST = load_model('models/MNIST_digits_recognition.h5', compile=False)

		# create empty ndarray
		numbers_mnist = np.ones(shape=(self.sudoku_size - 1, self.sudoku_size - 1))

		pics = deepcopy(self.list_of_number_pictures)
		for i in range(self.sudoku_size - 1):
			for j in range(self.sudoku_size - 1):
				pics[i][j] = self.preprocess_cell(pics[i][j], mnist=True, resize=True, clean_remains=True)
				if self.empty_cells[i][j] != 0:
					numbers_mnist[i][j] = np.argmax(number_recognizer_MNIST.predict([[pics[i][j].reshape(28,28,1)]]))

		return numbers_mnist

	def use_model_trained_on_printed_images(self):

		# load the model
		number_recognizer = load_model('models/Printed_digits_recognition.h5', compile=False)

		# create empty ndarray
		numbers = np.ones(shape=(self.sudoku_size - 1, self.sudoku_size - 1))

		pics = deepcopy(self.list_of_number_pictures)

		# predict numbers with the model
		for i in range(self.sudoku_size - 1):
			for j in range(self.sudoku_size - 1):
				preprocessed_img = self.preprocess_cell(pics[i][j], mnist=False, resize=True, clean_remains=True)
				prediction = number_recognizer.predict([[preprocessed_img.reshape(28,28,3)]])
				numbers[i][j] = np.argmax(prediction)

		return numbers

	def use_pytesseract(self):

		# create empty ndarray
		numbers_tesseract = np.ones(shape=(self.sudoku_size - 1, self.sudoku_size - 1))

		for i in range(self.sudoku_size - 1):
			for j in range(self.sudoku_size - 1):
				if self.empty_cells[i][j] != 0:
					# recognise a number 
					result = pytesseract.image_to_string(self.list_of_number_pictures[i][j], config='--psm 7 -c tessedit_char_whitelist=0123456789.%')
					try:
						if len(result) > 1:
							result = result[-1]
						numbers_tesseract[i][j] = int(result)
					# in case of an empty cell   
					except:
						numbers_tesseract[i][j] = 0
		return numbers_tesseract
            







		


		




