The program aims to test gaze estimation algorithms with given data.
There are two types of data: one is through the mouse cursor and the other is with given shown black points on the screen

This program is written in python 3.8 on pycharm with conda environment
First to run it you need to install the following packages: numpy, pickle, Pillow, dlib, cv2, pyautogui, random, math,
ctypes, ntpath, imutils, sklearn.

The file createPics.py creates pictures with black points to show to the user. This save it to a folder with the dictionary corresponding to its coordinations.
The file faceUtilities and facialLandMarks is for auxilary functions that takes a picture and return the cropped eye pictures.
The file createDatabase creates a database of cropped eye pictures with the corresponding mouse location.
The file createDatabaseWithVideo creates a database of cropped eye pictures with the corresponding black point location.
The file train.py is responsible for training the models.
The file test.py is responsible for the live gaze estimation from following the mouse location
The file testWithVideo is responsible for the live gaze estimation from following black points on the screen from the video

Requirments:
the data folder should be downloaded to the main directory. The "data" folder can be found in the dropbox under Dataset

