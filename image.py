import cv2
import logging

class Image():
	def __init__():
		self.original = None

	def read_image(path):
		try:
			self.original = cv2.imread(path)
		except:
			logging.info("Error in reading the image!")


