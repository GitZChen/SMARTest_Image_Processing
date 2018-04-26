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

def get_lightness(image):
	lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l = cv2.split(lab_img)[0]
	
	return int(l.mean())
	
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

def write_text(image, text, fontColor=(255,255,255)):
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,image.shape[1]-20)
	fontScale              = 1
	lineType               = 2
	
	new_image = image.copy()
	cv2.putText(new_image, text, 
		bottomLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)
	
	return new_image


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

def adaptive_threshold_mean(image, inverse = True):
	if len(image.shape) > 2:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	if inverse:
		mode = cv2.THRESH_BINARY_INV
	else:
		mode = cv2.THRESH_BINARY
		
	threshold_img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,mode,13,2)
	#cv2.imshow('threshold_img1', threshold_img)
	
	return threshold_img

''' threshold_mean has better performance for our use case
def adaptive_threshold_gaussian(image):
	threshold_img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		cv2.THRESH_BINARY,11,2)
	#cv2.imshow('threshold_img2', threshold_img)
	return threshold_img
'''

def erode(image, kernel_size=3):
	kernel = np.ones((kernel_size,kernel_size),np.uint8)
	eroded = cv2.erode(image,kernel,iterations = 1)
	return eroded

def morph_open(image, kernel_size=3):
	# closing should be performed after cv2.THRESH_BINARY_INV
	# kernel_size of 2 and 3 are acceptable
	# 2 is more detail and more noise
	# 3 is just right.
	kernel = np.ones((kernel_size,kernel_size),np.uint8)
	opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	return opening

def morph_close(image, kernel_size=3):
	# closing should be performed after cv2.THRESH_BINARY
	# kernel_size of 2 and 3 are acceptable
	# 2 is more detail and more noise
	# 3 is just right.
	kernel = np.ones((kernel_size,kernel_size),np.uint8)
	closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	return closing

def hough_circle_detection(image):
	# detect circles in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray, cv2.CV_HOUGH_GRADIENT, 1.2, 100)
	output = image.copy()
	
	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

		# show the output image
		cv2.imshow("output", np.hstack([image, output]))
		return output
	else:
		return None


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

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

############################## Main Routine ##############################

# index : [0-5], length = 6
TestPhotos_HIV1_file_names = ['HIV_Negative_Android_3162017.jpg', 'HIV_Positive_0516_Residue.JPG',
					   'HIV_Positive_0516.JPG', 'HIV_Positive_Sample.jpg', 'PSF8_square.jpg',
					   'PSF9_square.jpg']
# index : [0-10], length = 11
OnePlusOne_folder_file_names = ['IMG_20180406_135708.jpg', 'IMG_20180406_135715.jpg',
					   'IMG_20180406_135830.jpg', 'IMG_20180406_140915.jpg', 'IMG_20180406_140916.jpg',
					   'IMG_20180406_140928.jpg', 'IMG_20180406_140933.jpg', 'IMG_20180406_142924.jpg',
					   'IMG_20180406_142932.jpg', 'IMG_20180406_144902.jpg', 'IMG_20180406_144906.jpg']

isDebugging = True # This is set manually

if __name__ == "__main__":
	
	# Determine input image path based on cmd line args
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		if isDebugging:
			isVideo = False
			img_name = 'TestPhotos_HIV1/'+TestPhotos_HIV1_file_names[5] # 3,4,5 contour incorrect detection
			#img_name = 'TestPhotos_OnePlusOne/'+OnePlusOne_folder_file_names[0]
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
				opened_frame = morph_open(threshold_frame)
				cv2.imshow('opened_frame', opened_frame)
				
				if cv2.waitKey(30) >= 0:
					break
			else: 
				break
		
		cap.release()
		cv2.destroyAllWindows()
		
	else:
		
		img = read_n_resize(img_name)
		#cv2.imshow('img', img)
		
		#pst_mask, pst_img = PST(img)
		#cv2.imshow('PSTd_mask', pst_mask*255.0)
		
		#img = preprocess(img)
		#cv2.imshow('img_preprocessed', img)
		
		threshold_mask = adaptive_threshold_mean(img)
		#cv2.imshow('threshold_img', threshold_img)
		opened_mask = morph_open(threshold_mask)
		cv2.imshow('opened_mask', opened_mask)
		cv2.imshow('overlayed', mh.overlay(img.mean(axis=2), opened_mask))
		
		uped = cv2.pyrUp(opened_mask)
		#cv2.imshow('uped', uped)
		blurred = uped
		for _ in range(15):
			blurred = cv2.medianBlur(blurred, 15)
		#cv2.imshow('blurred1', blurred)
		blurred = (blurred > 30)*255.0
		#cv2.imshow('blurred2', blurred)
		opened_sm = cv2.pyrDown(blurred)
		opened_sm = cv2.threshold(opened_sm,127,255,cv2.THRESH_BINARY)[1]
		cv2.imshow('open_sm', opened_sm)
		
		edged = opened_sm.astype("uint8")
		image = img.copy()
		
		_, cnts, hrchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		hrchy = hrchy[0]
		i=0
		while i < len(cnts):
			if hrchy[i][3] < 0:
				#cnts.pop(i)
				np.delete(hrchy, i, 0)
			i += 1
		#print(len(cnts))
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
		
		# find the contours in the edged image, keeping only the
		# largest ones, and initialize the screen contour
		boxCnts = []
		approx = 0
		n=-1
		#print()
		# loop over the contours
		for c in cnts:
			n+=1
			#print(n)
			#if n == 0: continue
			ct_img = img.copy()
			# show the contour (outline) of the piece of paper
			cv2.drawContours(ct_img, [c], -1, (0, 255, 0), 2)
			cv2.imshow("ct_img", ct_img)
			
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			print(len(approx))
			cv2.waitKey(0)
			
			# if our approximated contour has four points, then we
			# can assume that we have found our screen
			if len(approx) == 4:
				#print("matched")
				rect = cv2.minAreaRect(c) # returns (center, (width, height), rotation angle)
				#print(rect[1])
				#print(rect[1][0]/rect[1][1])
				boxCnt = cv2.boxPoints(rect).astype(approx.dtype)
				#print(boxCnt)
				
				# Below returns distance between corresponding pts of
				#	 a = boxCntBiggestSqr, b = boxCntBiggestRect
				# dist = [int(((a[i][0]-b[i][0])**2+(a[i][1]-b[i][1])**2)**0.5) for i in range(len(a))]
				# 	-> [9, 54, 54, 9]
				# 	if we can extract index, then we get orientation.
				
				boxCnt = np.reshape(boxCnt, approx.shape)
				boxCnts.append(boxCnt)
				break
		
		if len(boxCnts) <= 0:
			print("Rectangle not found!")
			sys.exit(1)
		
		# show the contour (outline) of the piece of paper
		cv2.drawContours(image, [boxCnt], -1, (0, 255, 0), 2)
		cv2.imshow("Outline", image)
		'''
		cnt_img2 = img.copy()
		cv2.drawContours(cnt_img2, [approx], -1, (0, 255, 0), 2)
		cv2.imshow("Outline2", cnt_img2)
		'''
		cv2.waitKey(0)
		# apply the four point transform to obtain a top-down
		# view of the original image
		warped = four_point_transform(img, boxCnt.reshape(4, 2))
		
		'''
		# convert the warped image to grayscale, then threshold it
		# to give it that 'black and white' paper effect
		warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		T = adaptive_threshold_mean(warped, False)
		warped = (warped > T).astype("uint8") * 255
		'''
		
		# show the original and scanned images
		#cv2.imshow("Original", img)
		cv2.imshow("Scanned", warped)
		print("Lightness of {}: {}".format(img_name, get_lightness(warped)))
		
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

	