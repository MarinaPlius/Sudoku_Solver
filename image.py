import cv2
import numpy as np
from copy import deepcopy
from tensorflow.keras.models import load_model
import pytesseract
from collections import Counter


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
		self.horizontal_lines, self.vertical_lines = self.get_lines()
		self.font_size =  abs(self.horizontal_lines[0] - self.horizontal_lines[1]) // 14 # formula for the best size
		self.list_of_number_pictures = self.get_list_of_number_pictures()
		self.empty_cells = self.find_empty_cells()
		self.predictions = self.use_ensemble_model()
		self.sudoku_solution = self.get_sudoku_solution(np.copy(self.predictions))
		self.image_with_solution = self.draw_solution()
	
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

	def get_lines(self):
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

		return X, Y

	def get_list_of_number_pictures(self):
		X = self.horizontal_lines
		Y = self.vertical_lines

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
		cells = np.ones(shape=(self.sudoku_size, self.sudoku_size))

		for i in range(self.sudoku_size):
			for j in range(self.sudoku_size):
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
		numbers_mnist = np.ones(shape=(self.sudoku_size, self.sudoku_size))

		pics = deepcopy(self.list_of_number_pictures)
		for i in range(self.sudoku_size):
			for j in range(self.sudoku_size):
				pics[i][j] = self.preprocess_cell(pics[i][j], mnist=True, resize=True, clean_remains=True)
				if self.empty_cells[i][j] != 0:
					numbers_mnist[i][j] = np.argmax(number_recognizer_MNIST.predict([[pics[i][j].reshape(28,28,1)]]))

		return numbers_mnist

	def use_model_trained_on_printed_images(self):

		# load the model
		number_recognizer = load_model('models/Printed_digits_recognition.h5', compile=False)

		# create empty ndarray
		numbers = np.ones(shape=(self.sudoku_size, self.sudoku_size))

		pics = deepcopy(self.list_of_number_pictures)

		# predict numbers with the model
		for i in range(self.sudoku_size):
			for j in range(self.sudoku_size):
				preprocessed_img = self.preprocess_cell(pics[i][j], mnist=False, resize=True, clean_remains=True)
				prediction = number_recognizer.predict([[preprocessed_img.reshape(28,28,3)]])
				numbers[i][j] = np.argmax(prediction)

		return numbers

	def use_pytesseract(self):

		pics = deepcopy(self.list_of_number_pictures)

		# create empty ndarray
		numbers_tesseract = np.zeros(shape=(self.sudoku_size, self.sudoku_size))

		for i in range(self.sudoku_size):
			for j in range(self.sudoku_size):
				if self.empty_cells[i][j] != 0:
					# recognise a number 
					result = pytesseract.image_to_string(self.preprocess_cell(pics[i][j], mnist=True, resize=True, clean_remains=True), config='--psm 7 -c tessedit_char_whitelist=0123456789.%')
					try:
						numbers_tesseract[i][j] = int(result[0])
					# in case of an empty cell   
					except:
						continue
		return numbers_tesseract
	
	def use_ensemble_model(self):
		"""
		gets predictions from all the models and return the most frequent predictions
		additional weight given to tesseract model in case of all the predictions differ
		"""
		#get predictions
		predictions_mnist = self.use_mnist_model()
		predictions_print_img = self.use_model_trained_on_printed_images()
		predictions_tesseract = self.use_pytesseract()

		# create empty ndarray
		numbers = np.zeros(shape=(self.sudoku_size, self.sudoku_size))

		#find the most frequent
		for i in range(self.sudoku_size):
			for j in range(self.sudoku_size):
				preds = [predictions_mnist[i][j], predictions_print_img[i][j],predictions_tesseract[i][j]]
				if len(set(preds)) == 3:
					numbers[i][j] = preds[2]
				else:
					occurence_count = Counter(preds)
					numbers[i][j] = occurence_count.most_common(1)[0][0]

		return numbers
	
	
	def get_sudoku_solution(self, grid):
		# sudoku solver
		def find_next_cell_to_fill(grid, i, j):
			for x in range(i,9):
					for y in range(j,9):
							if grid[x][y] == 0:
									return x,y
			for x in range(0,9):
					for y in range(0,9):
							if grid[x][y] == 0:
									return x,y
			return -1,-1

		def is_valid(grid, i, j, e):
			rowOk = all([e != grid[i][x] for x in range(9)])
			if rowOk:
					columnOk = all([e != grid[x][j] for x in range(9)])
					if columnOk:
							# finding the top left x,y co-ordinates of the section containing the i,j cell
							secTopX, secTopY = 3 *(i//3), 3 *(j//3) #floored quotient should be used here. 
							for x in range(secTopX, secTopX+3):
									for y in range(secTopY, secTopY+3):
											if grid[x][y] == e:
													return False
							return True
			return False

		def solve_sudoku(grid, i=0, j=0):
			i,j = find_next_cell_to_fill(grid, i, j)
			if i == -1:
				return True
			for e in range(1,10):
				if is_valid(grid,i,j,e):
					grid[i][j] = e
					if solve_sudoku(grid, i, j):
						return True
					# undo the current cell for backtracking
					grid[i][j] = 0
			return False
		
		solve_sudoku(grid, i=0, j=0)
		return grid
	
	def draw_solution(self):
		"""
		creates a copy of the original image and draws solution on it
		returns a new image
		"""
		new_image = self.original.copy()
		font = cv2.FONT_HERSHEY_PLAIN
		green = (0, 255, 0)
		thickness = 3
		X = self.horizontal_lines
		Y = self.vertical_lines

		# draw numbers
		for i in range(self.sudoku_size):
			for j in range(self.sudoku_size):
				if self.predictions[i][j] == 0:
					cv2.putText(new_image,str(int(self.sudoku_solution[i][j])),(X[j]+5, Y[i+1]), font, self.font_size, green, thickness, cv2.LINE_AA)

		return new_image
		










            







		


		




