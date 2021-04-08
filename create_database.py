
import dlib
import pyautogui
from face_utils import get_face_landmarks
from face_utils import get_eyes
from utils import *
import pickle

# parameters to the script
path = "pictures/stable2"
counter_path = path+"/counter.txt"
predictor_path = "shape_predictor_68_face_landmarks.dat" # =args["shape_predictor"]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

i = get_counter(counter_path)

try:
   fh = open(path + "/coordinates.txt", "rb")
   coordinates = pickle.load(fh)
except IOError:
   print ("Error: can't find file or read data")
   coordinates = []
else:
   print ("read the file succesfully")
   fh.close()

cap = cv2.VideoCapture(0)
while cap.isOpened():
	_, img = cap.read()
	mousex, mousey = pyautogui.position()

	landmarks = get_face_landmarks(predictor, detector, img)
	left, right = get_eyes(img, landmarks)

	cv2.imshow("left", left)
	cv2.imshow("right", right)

	cv2.imwrite(path+"/{}_left.jpg".format(i), left)
	cv2.imwrite(path+"/{}_right.jpg".format(i), right)
	coordinates.append((i, (mousex, mousey)))
	i=i+1

	cv2.imshow("camera", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

with open(path+"/coordinates.txt", "wb") as fp:
	pickle.dump(coordinates, fp)

update_counter(counter_path,i)