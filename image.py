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
		

	def get_list_of_number_pictures(self):
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

	def use_mnist_model(self):

		def preprocess_for_MNIST(img):
			"""
			inverses and resizes a pic to 28*28 size,
			cleans from noize
			returns clean small nice pic
			"""

			# remove colors channel
			gray_pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


			threshold = 117
			max_value= 255

			# inverse to black
			inverse = 255 - gray_pic
			ret1, thresh1 = cv2.threshold(inverse, threshold, max_value,
				cv2.THRESH_BINARY)

			resize = cv2.resize(thresh1,(28,28),interpolation = cv2.INTER_CUBIC)
			# clean remains of lines
			for i in range(28):
				for j in range(5):
					resize[i][j] = 0 # top of the image
			for i in range(5):
				for j in range(28):
					resize[i][j] = 0 # left side

			return resize

		# load the model
		number_recognizer_MNIST = load_model('models/MNIST_digits_recognition.h5', compile=False)

		# create empty ndarray
		numbers_mnist = np.ones(shape=(self.sudoku_size - 1, self.sudoku_size - 1))

		# clean, resize and inverse all the pics to use them with the model
		pics = deepcopy(self.list_of_number_pictures)
		for i in range(self.sudoku_size - 1):
			for j in range(self.sudoku_size - 1):
				pics[i][j] = preprocess_for_MNIST(pics[i][j])
				if np.sum((pics[i][j]) == 255) < 10:
					numbers_mnist[i][j] = 0
				else:
					numbers_mnist[i][j] = np.argmax(number_recognizer_MNIST.predict([[pics[i][j].reshape(28,28,1)]]))

		return numbers_mnist

	def use_model_trained_on_printed_images(self):

		# load the model
		number_recognizer = load_model('models/Printed_digits_recognition.h5', compile=False)

		# create empty ndarray
		numbers = np.ones(shape=(self.sudoku_size - 1, self.sudoku_size - 1))

		# predict numbers with the model
		for i in range(self.sudoku_size - 1):
			for j in range(self.sudoku_size - 1):
				if numbers[i][j] == 1:
					# resize the image
					resized_pic = cv2.resize(self.list_of_number_pictures[i][j],(28,28),interpolation = cv2.INTER_CUBIC)
					# clean remains of lines
					for k in range(28):
						for l in range(5):
							for m in range(3):
								resized_pic[k][l][m] = 255 # top
					for k in range(5):
						for l in range(28):
							for m in range(3): # left side
								resized_pic[k][l][m] = 255
					prediction = number_recognizer.predict([[resized_pic.reshape(28,28,3)]])
					numbers[i][j] = np.argmax(prediction)

		return numbers

	def use_pytesseract(self):

		# create empty ndarray
		numbers_tesseract = np.ones(shape=(self.sudoku_size - 1, self.sudoku_size - 1))

		for i in range(self.sudoku_size - 1):
			for j in range(self.sudoku_size - 1):
				if sudoku_numbers[i][j] == 1:
					# recognise a number 
					result = pytesseract.image_to_string(self.list_of_number_pictures[i][j], config='--psm 7 -c tessedit_char_whitelist=0123456789.%')
					try:
						if len(result) > 1:
							result = result[-1]
						numbers_tesseract[i][j] = int(result)
					# in case of an empty cell   
					except:
						numbers_tesseract[i][j] = 0
            







		


		




