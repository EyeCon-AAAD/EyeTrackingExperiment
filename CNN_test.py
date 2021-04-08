# import the necessary packages
import dlib
import pyautogui
import ctypes
from face_utils import get_face_landmarks
from face_utils import get_eyes
import torch
from CNN_net import *
pyautogui.FAILSAFE = False
from utils import *

dataset_path = "pictures/retrain"
model_version = "CNN5_STABLE2"
predictor_path = "shape_predictor_68_face_landmarks.dat" # =args["shape_predictor"]

count = 0
sum = 0
def get_precision(x, y, px, py):
    global count, sum
    count +=1
    sum += distance2D(x, y, px, py)
    return sum/count

model = CNN5(1)
model.load_state_dict(torch.load("models/"+model_version+".pt"))
model.eval()

color_object = pointer_color()
color = color_object.get_color()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# creating the canvas for showing the results
user32 = ctypes.windll.user32
width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
blank_image = np.ones((height, width, 3), np.uint8)

cv2.namedWindow("full_window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("full_window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow("full_window", blank_image)
no_face_win = "face is not detected"
no_face = False

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, webcam_img = cap.read()
    img = webcam_img
    mousex, mousey = pyautogui.position()

    landmarks = get_face_landmarks(predictor, detector, img)
    if len(landmarks) < 1:
        print('ERROR:: no face detected')
        continue

    limg, rimg = get_eyes(img, landmarks)

    limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    limg = cv2.equalizeHist(limg)
    cv2.imshow("leftt", limg)
    limg = np.array(limg)

    rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    rimg = cv2.equalizeHist(rimg)
    cv2.imshow("rightt", rimg)
    rimg = np.array(rimg)

    img = np.concatenate((limg, rimg))

    img = img.astype(np.float32)
    img/=255
    img = torch.from_numpy(np.array([img])).to(torch.float)
    img = img.reshape(len(img), 1, 64, 64)

    #-------------------geting the prediction------------------
    pred = model(img)
    print(pred)
    _, predicted = torch.max(pred, 1)

    print('prediction', predicted)


    predx, predy = grid2point(predicted, 4, 4)

    draw_grid(blank_image, width//4, height//4)
    cv2.circle(blank_image, (predx, predy), 5, color_object.get_color(), 5)
    cv2.imshow("full_window", blank_image)
    blank_image = np.ones((height, width, 3), np.uint8)

    #--------------------evaluate------------------------------
    result_totall = get_precision(mousex, mousey, predx, predy)
    result_x = get_precision(mousex, 0, predx, 0)
    result_y = get_precision(0, mousey, 0, predy)

    '''
    print("x & y: ",mousex, mousey, "-", predx, predy, "->", result_totall)
    print("x: ", mousex, "-", predx, "->", result_x)
    print("y: ", mousey, "-", predy, "->", result_y)
    '''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Results:")
print("x & y: ",result_totall)
print("x: ", result_x)
print("y: ", result_y)

cap.release()
