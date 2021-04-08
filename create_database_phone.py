# This creates the database of pictures of right and left eyes without intro video
# The user will look at the mouse cursor and the program will save the captured cropped pics to the path
# Additionally it will save the mouse cursor coordinates corresponding to a specific picture
import glob

import dlib
import pyautogui
from face_utils import get_face_landmarks
from face_utils import get_eyes
from utils import *
import pickle
import os
# parameters to the script
path = "pictures/phone/"
counter_path = path+"/counter.txt"
predictor_path = "shape_predictor_68_face_landmarks.dat" # =args["shape_predictor"]

# These two variables will be used in image processing after capturing pics from the camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# get the counter for the images to append to
i = get_counter(counter_path)

# load the previous the coordinates
try:
   fh = open(path + "/coordinates.txt", "rb")
   coordinates = pickle.load(fh)
except IOError:
   print ("Error: can't find file or read data, thus creating new one")
   coordinates = []
else:
   print ("read the file succesfully")
   fh.close()



def load_video_paths(folder_path):
	videoPaths = []
	print(folder_path)
	for file_path in glob.iglob(folder_path+'/*.mp4', recursive=True):
	    videoPaths.append(file_path)
	return videoPaths

paths = load_video_paths("phonePic")

def readVideo(videopath, x, y):# This captures the live feed from the camera
	global i
	global path
	cap = cv2.VideoCapture(videopath)
	numberofpics = 0
	# continued live feed until terminated
	while cap.isOpened():
		ret, img = cap.read()
		if ret == True:

			img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

			numberofpics+=1

			# This creates 2 images from the img and the calculated landmarks
			landmarks = get_face_landmarks(predictor, detector, img)
			left, right = get_eyes(img, landmarks)

			#show the images
			cv2.imshow("left", left)
			cv2.imshow("right", right)

			#saving the pictures
			cv2.imwrite(path+"/{}_left.jpg".format(i), left)
			cv2.imwrite(path+"/{}_right.jpg".format(i), right)

			#adding the coordinates
			coordinates.append((i, (x, y)))
			i=i+1

			#show the pictures camptured with the camera
			cv2.imshow("camera", img)
			if numberofpics == 50:
				break
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
	cap.release()
	return numberofpics

coor = [(76, 990),(76, 540),(76, 90),(380, 990),(380, 540),(380, 90),(760, 990),(760, 540),(760, 90),(1140, 990),(1140, 540),
		(1140, 90),(1520, 990),(1520, 540),(1520, 90),(1900, 990),(1900, 540), (1900, 90),(2204, 990),(2204, 540), (2204, 90)]
a  =0


for vpath in paths:
	num = readVideo(vpath, coor[a][0], coor[a][1])
	a+=1
	print(num)



#save the coordinates list for late use
with open(path+"/coordinates.txt", "wb") as fp:
	pickle.dump(coordinates, fp)

# update the counter for adding the pictures in later runs
update_counter(counter_path,i)

