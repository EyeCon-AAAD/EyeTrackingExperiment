from collections import OrderedDict
import numpy as np
import cv2

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])


def get_face_landmarks(predictor, detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    landmarks = []

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        landmarks.append(shape)

    return landmarks

def crop_eye(image, landmarks):
	(x, y, w, h) = cv2.boundingRect(landmarks)
	eye = image[y - 3:y + h + 3, x - 5:x + w + 5]
	eye = cv2.resize(eye, (64, 32), interpolation=cv2.INTER_AREA)
	return eye

def get_eyes(image, landmarks):

	landmarks = landmarks[0]

	i, j = FACIAL_LANDMARKS_IDXS["left_eye"]
	left_landmarks = np.array([landmarks[i:j]])
	left_eye = crop_eye(image, left_landmarks)

	i, j = FACIAL_LANDMARKS_IDXS["right_eye"]
	right_landmarks = np.array([landmarks[i:j]])
	right_eye = crop_eye(image, right_landmarks)

	return left_eye, right_eye
