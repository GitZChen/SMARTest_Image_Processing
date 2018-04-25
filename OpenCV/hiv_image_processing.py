import cv2
import numpy as np
import mahotas as mh
import sys
import math

######################### Generic Image Processing Functions #########################



### Lighting Adjustment Algorithms ###

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

def adjust_brightness(image, brightness=180):
	# brightness is btw [0,255] and is the result brightness value
	return adjust_brightness_contrast(image, brightness, 1)

def adjust_contrast(image, contrast_factor=1.15):
	return adjust_brightness_contrast(image, image.mean(), contrast_factor)

def adjust_brightness_contrast(image, brightness=180, contrast_factor=1.15):
	img_32 = np.asarray(image, dtype="int32")
	brightness_offset = brightness - image.mean()*contrast_factor
	adjusted_img_32 = (img_32*contrast_factor + brightness_offset).clip(0, 255)
	
	return np.asarray(adjusted_img_32, dtype="uint8")

def CLAHE_adjust(image):
	# Actually improves contrast more than adjusting lightness
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
	cl = clahe.apply(l)
	cl = adjust_contrast(cl)
	
	limg = cv2.merge((cl,a,b))
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return final



### General Algorithms ###

def filter_out_blue(image):
	blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
	hsv_img = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
	
	lower_blue = np.array([160//2,10,35])
	upper_blue = np.array([270//2,255,255])

	mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
	blue_img = cv2.bitwise_and(image, image, mask = mask)
	
	return blue_img, mask

def is_blurry(image, threshold=200):
	return get_blurriness(image) < threshold

def get_blurriness(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()



### Edge Detection Algorithms ###

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

def detect_edges_canny(image):
	# smoothes out image with bilateral blur filter before applying canny edge detection
	blurred = cv2.bilateralFilter(image,9,15,15)
	edged = cv2.Canny(blurred, 35, 125) # orig: (75, 200); L2gradient = True?
	#edged = auto_canny(blurred)
	#cv2.imshow("Edged_b2", edged)
	return edged

def cart2pol(x, y):
	theta = np.arctan2(y, x)
	rho = np.hypot(x, y)
	return (theta, rho)

def PST(I,LPF=0.21,Phase_strength=0.48,Warp_strength=12.14, Threshold_min=-1, Threshold_max=0.0019, Morph_flag=1):
	# I: image
	# Gaussian Low Pass Filter
	#	LPF = 0.21
	# PST parameters:
	# 	Phase_strength = 0.48 
	#	Warp_strength = 12.14
	# Thresholding parameters (for post processing after the edge is computed)
	#	Threshold_min = -1
	#	Threshold_max = 0.0019
	# To compute analog edge, set Morph_flag = 0 and to compute digital edge, set Morph_flag = 1
	# 	Morph_flag = 1 
	I_initial = I
	if(len(I.shape) == 3):
		I = I.mean(axis=2)
	
	L = 0.5
	x = np.linspace(-L, L, I.shape[0])
	y = np.linspace(-L, L, I.shape[1])
	[X1, Y1] = (np.meshgrid(x, y))
	X = X1.T
	Y = Y1.T
	[THETA,RHO] = cart2pol(X,Y)
	
	# Apply localization kernel to the original image to reduce noise
	Image_orig_f=((np.fft.fft2(I)))  
	expo = np.fft.fftshift(np.exp(-np.power((np.divide(RHO, math.sqrt((LPF**2)/np.log(2)))),2)))
	Image_orig_filtered=np.real(np.fft.ifft2((np.multiply(Image_orig_f,expo))))
	# Constructing the PST Kernel
	PST_Kernel_1=np.multiply(np.dot(RHO,Warp_strength), np.arctan(np.dot(RHO,Warp_strength)))-0.5*np.log(1+np.power(np.dot(RHO,Warp_strength),2))
	PST_Kernel=PST_Kernel_1/np.max(PST_Kernel_1)*Phase_strength
	# Apply the PST Kernel
	temp=np.multiply(np.fft.fftshift(np.exp(-1j*PST_Kernel)),np.fft.fft2(Image_orig_filtered))
	Image_orig_filtered_PST=np.fft.ifft2(temp)

	# Calculate phase of the transformed image
	PHI_features=np.angle(Image_orig_filtered_PST)

	if Morph_flag == 0:
		out=PHI_features
		return out
	else:
		#   find image sharp transitions by thresholding the phase
		features = np.zeros((PHI_features.shape[0],PHI_features.shape[1]))
		features[PHI_features> Threshold_max] = 1 # Bi-threshold decision
		features[PHI_features< Threshold_min] = 1 # as the output phase has both positive and negative values
		features[I<(np.amax(I)/20)] = 0 # Removing edges in the very dark areas of the image (noise)

		# apply binary morphological operations to clean the transformed image 
		out = features
		out = mh.thin(out, 1)
		out = mh.bwperim(out, 4)
		out = mh.thin(out, 1)
		out = mh.erode(out, np.ones((1, 1))); 
		
		Overlay = mh.overlay(I, out)
		return (out, Overlay)

def adaptive_threshold_mean(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	threshold_img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
		cv2.THRESH_BINARY_INV,13,2)
	#cv2.imshow('threshold_img1', threshold_img)
	
	return threshold_img

''' threshold_mean has better performance for our use case
def adaptive_threshold_gaussian(image):
	threshold_img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		cv2.THRESH_BINARY,11,2)
	#cv2.imshow('threshold_img2', threshold_img)
	return threshold_img
'''

def opening(image, kernel_size=3):
	# closing should be performed after cv2.THRESH_BINARY_INV
	# kernel_size of 2 and 3 are acceptable
	# 2 is more detail and more noise
	# 3 is just right.
	kernel = np.ones((kernel_size,kernel_size),np.uint8)
	opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	return opening

def closing(image, kernel_size=3):
	# closing should be performed after cv2.THRESH_BINARY
	# kernel_size of 2 and 3 are acceptable
	# 2 is more detail and more noise
	# 3 is just right.
	kernel = np.ones((kernel_size,kernel_size),np.uint8)
	closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	return closing

def hough_circle_detection(image):
	pass


############################## Procedural Functions  ##############################

def read_n_resize(image_path):
	image = cv2.imread(image_path)
	image = resize(image)
	#cv2.imshow('image_in', image)
	
	return image

def resize(image):
	if image.shape[0]/image.shape[1] == 3/4:
		return cv2.resize(image,(400,300), interpolation = cv2.INTER_CUBIC)
	elif image.shape[0]/image.shape[1] == 4/3:
		return cv2.resize(image,(300,400), interpolation = cv2.INTER_CUBIC)
	else:
		return cv2.resize(image,(int(image.shape[1]/image.shape[0]*300),300), interpolation = cv2.INTER_CUBIC)

def preprocess(image):
	#cv2.imshow('original', image)
	#image = CLAHE_adjust(image)
	#cv2.imshow('after_clahe1', image)
	
	image = adjust_gamma(image)
	#cv2.imshow('after_gamma', image)
	
	image = adjust_contrast(image, 1.25)
	#cv2.imshow('after_contrast', image)
	
	#image = CLAHE_adjust(image)
	#cv2.imshow('after_clahe2', image)
	
	return image

def print_usage_and_quit():
	print_usage()
	sys.exit(1)

def print_usage():
	print("Usage: python hivImageProcessing.py [-i]/[--image]/[-v]/[--video] [FILENAME]")

def detect_edges(image):
	return detect_edges_canny(image)


############################## Main Routine ##############################

# index : [0-5], length = 6
test_img_file_names = ['HIV_Negative_Android_3162017.jpg', 'HIV_Positive_0516_Residue.JPG',
					   'HIV_Positive_0516.JPG', 'HIV_Positive_Sample.jpg', 'PSF8_square.jpg',
					   'PSF9_square.jpg']
# index : [0-10], length = 11
test_img_file_names2 = ['IMG_20180406_135708.jpg', 'IMG_20180406_135715.jpg',
					   'IMG_20180406_135830.jpg', 'IMG_20180406_140915.jpg', 'IMG_20180406_140916.jpg',
					   'IMG_20180406_140928.jpg', 'IMG_20180406_140933.jpg', 'IMG_20180406_142924.jpg',
					   'IMG_20180406_142932.jpg', 'IMG_20180406_144902.jpg', 'IMG_20180406_144906.jpg']

isDebugging = True # This is set manually

if __name__ == "__main__":
	
	# Determine input image path based on cmd line args
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		if isDebugging:
			isVideo = False
			img_name = 'TestPhotos_HIV1/'+test_img_file_names[5]
			#img_name = 'TestPhotos_OnePlusOne/'+test_img_file_names2[0]
		else:
			print_usage_and_quit()
	elif sys.argv[1].lower() in ["-v", "--video"]:
		isVideo = True
		if len(sys.argv) == 2:
			video_file = 0
		else:
			video_file = sys.argv[2]
	elif sys.argv[1].lower() in ["-i", "--image"] :
		isVideo = False
		if len(sys.argv) == 3:
			img = cv2.imread(sys.argv[2])
		else:
			print_usage_and_quit()
	else:
		print_usage_and_quit()
	
	if isVideo:
		
		cap = cv2.VideoCapture(video_file)
		if (cap.isOpened()== False): 
			print("Error opening video stream or file")

		while cap.isOpened():
			ret, frame = cap.read()
			if ret == True:
				resized_frame = resize(frame)
				cv2.imshow('Frame',resized_frame)
				preprocessed_frame = preprocess(resized_frame)
				cv2.imshow("preprocessed_frame",preprocessed_frame)
				
				#PEdge, PFrame = PST(preprocessed_frame)
				NPEdge, NPFrame = PST(resized_frame)
				#cv2.imshow('PEdge',PEdge*255.0)
				cv2.imshow('NPEdge',NPEdge*255.0)
				#cv2.imshow('PFrame',PFrame)
				#cv2.imshow('NPFrame',NPFrame)
				
				threshold_frame = adaptive_threshold_mean(resized_frame)
				#cv2.imshow('threshold_frame', threshold_frame)
				opened_frame = opening(threshold_frame)
				cv2.imshow('opened_frame', opened_frame)
				
				if cv2.waitKey(30) >= 0:
					break
			else: 
				break
		
		cap.release()
		cv2.destroyAllWindows()
		
	else:
		
		img = read_n_resize(img_name)
		cv2.imshow('img', img)
	
		pst_img = PST(img)[1]
		cv2.imshow('PSTd_IMG', pst_img)
		
		#img = preprocess(img)
		#cv2.imshow('img_preprocessed', img)
		
		threshold_img = adaptive_threshold_mean(img)
		#cv2.imshow('threshold_img', threshold_img)
		opened_img = opening(threshold_img)
		cv2.imshow('opening', opened_img)
		
		#edge, overlayed_img = PST(img)
		#cv2.imshow('PST_EDGE', np.reshape(np.repeat(edge*255.0,3), edge.shape+(3,)))
		#cv2.imshow('PSTd_IMG', overlayed_img)
		
		# Detect edges
		#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#cv2.imshow("Edged", detect_edges(gray))
		#detect_edges(gray)
		
		cv2.waitKey(0)
		#res = np.hstack((img,equ)) #stacking images side-by-side
		#cv2.imwrite('res.png',res)

# End Main Routine



#################### Code below saved for potential future use. ####################

'''Equalize Histogram
grayed_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grayed_img_equ = cv2.equalizeHist(grayed_img)
cv2.imshow('grayed_img',grayed_img)
cv2.imshow('grayed_img_equ',grayed_img_equ)
#'''

''' Split channels, equalize histogram, and regroup
b,g,r = cv2.split(img)
b_eq = cv2.equalizeHist(b)
cv2.imshow('equ_img', cv2.merge((b_eq,g,r)))
#'''

	