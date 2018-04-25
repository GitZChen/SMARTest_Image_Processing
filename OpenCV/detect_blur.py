'''
This file was created as a test ground for automatic blur detection in photos. 
The intention is to detect images that are not focused properly in real time
as the camera is running and either
 1. disable the shutter button in the app, or
 2. display a warning to the user
so that we will not get a blurred input picture.

Much of the code here are reused from  hiv_image_processing.py  .
'''

# import the necessary packages
import cv2
import numpy as np

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def adjust_gamma(image, gamma=2):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	if gamma == 1.0:
		return image
	
	invGamma = 1.0 / gamma
	table = [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
	table = np.array(table).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def CLAHE_adj_lightness(image):
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
	cl = clahe.apply(l)
	cl = adjust_contrast(cl)
	
	limg = cv2.merge((cl,a,b))
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return final

def preprocess(image):
	image = CLAHE_adj_lightness(image)
	#cv2.imshow('after_clahe1', image)
	
	image = adjust_gamma(image)
	#cv2.imshow('after_gamma', image)
	
	image = CLAHE_adj_lightness(image)
	#cv2.imshow('after_clahe2', image)
	
	#image = adjust_brightness(image)
	
	return image


# loop over the input images
for imagePath in ["TestPhotos/PSF9	_square.jpg"]:
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	image = cv2.imread(imagePath)
	#image = np.zeros((600, 800), dtype="uint8")
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = image
	fm = variance_of_laplacian(gray)
	print(fm)
	text = "Not Blurry"

	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm < 100:
		text = "Blurry"

	# show the image
	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)
	cv2.imshow("Image", image)
	key = cv2.waitKey(0)
