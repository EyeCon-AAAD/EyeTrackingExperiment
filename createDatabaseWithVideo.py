# This creates the database of pictures of right and left eyes with intro video
# The user will look at the pictures appearing on the screen and follow the points
# The program will load the pics from given folders and show them to the user sequentially
# and the program will save the captured cropped pics of the eyes to the path
# Additionally it will save the mouse cursor coordinates corresponding to a specific picture

import dlib
import cv2
import pyautogui
from face_utils import get_face_landmarks
from face_utils import get_eyes
import pickle
import time
givenFolder = "video2"
path = "pictures/with_video"
predictor_path = "shape_predictor_68_face_landmarks.dat" # =args["shape_predictor"]

# These two variables will be used in image processing after capturing pics from the camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# This captures the live feed from the camera
# capturedVideo = cv2.VideoCapture(0)
capturedVideo = cv2.VideoCapture(0, cv2.CAP_DSHOW)
i = 1
coordinates = []

file = open("{}\\centers.pkl".format(givenFolder), "rb")
centers = pickle.load(file)
picsCount = centers["picsCount"]
file.close()
print(centers)

cv2.namedWindow("full_window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("full_window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
j =0
while capturedVideo.isOpened():
    j+=1
    if ( not j%5 ==0):
        continue
        i = i+1
    # This loads the images to be shown
    loadedImageName = "img{}.png".format(i)
    loadedImage = cv2.imread(givenFolder+"/"+loadedImageName, 0)
    # cv2.circle(loadedImage, (500, 500), 5, (125, 125, 0), 5)

    cv2.imshow("full_window", loadedImage)

    # wait for the eye adjustments
    #time.sleep(0.001)

    # This takes the current pics
    _, img = capturedVideo.read()

    # time.sleep(60)

    # this loads the coordination of the point on the shown picture
    centerX, centerY = centers[loadedImageName]

    # The landmarks is getting fixed points of important features from the pictures
    landmarks = get_face_landmarks(predictor, detector, img)
    # print(landmarks)

    # This creates 2 images from the img and the calculated landmarks
    leftEye, rightEye = get_eyes(img, landmarks)

    # This shows the cropped pics of the 2 eyes
    cv2.imshow("leftEye", leftEye)
    cv2.imshow("rightEye", rightEye)

    # This saves the pics of the eyes
    cv2.imwrite(path + "/{}_left.jpg".format(i), leftEye)
    cv2.imwrite(path + "/{}_right.jpg".format(i), rightEye)

    # Here we save the corresponding point coordinates
    coordinates.append((i, (centerX, centerY)))
    i = i + 1

    if i == picsCount:
        break
    # cv2.imshow("camera", img)

    # comment here
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capturing
capturedVideo.release()

# Save the coordinates in the file
with open(path+"/coordinates.txt", "wb") as fp:
    pickle.dump(coordinates, fp)

