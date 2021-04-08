# import the necessary packages
import dlib
import pyautogui
import ctypes
from face_utils import get_face_landmarks
from face_utils import get_eyes
from train import get_trained_models
pyautogui.FAILSAFE = False
from utils import *
from pynput.mouse import Button, Listener
import threading

# is used for error in pixels
count = 0
sum = 0
def get_precision(x, y, px, py):
    global count, sum
    count +=1
    sum += distance2D(x, y, px, py)
    return sum/count

#-------------------------- mouse listener--------------------------

#adds the new data created by the mouse clikcs
def add_new_data(dataset_path, new_image, x, y):

    # getting the data for appending the new picture to the folder
    counter_path = dataset_path + "/counter.txt"
    i = get_counter(counter_path)
    try:
        fh = open(dataset_path + "/coordinates.txt", "rb")
        coordinates = pickle.load(fh)
    except IOError:
        print("Error: can't find file the coordinates.txt, thus creating one")
        coordinates = []
    else:
        print("read the coordinates.txt file succesfully")
        fh.close()

    # getting the eye pictures
    landmarks = get_face_landmarks(predictor, detector, new_image)
    left, right = get_eyes(new_image, landmarks)

    # saving multiple pictures of the new data because it is more valuable
    for j in range(10):
        cv2.imwrite(dataset_path + "/{}_left.jpg".format(i), left)
        cv2.imwrite(dataset_path + "/{}_right.jpg".format(i), right)
        coordinates.append((i, (x, y)))
        i = i + 1

    #updating the coordinates and the counter
    update_counter(counter_path, i)
    with open(dataset_path + "/coordinates.txt", "wb") as fp:
        pickle.dump(coordinates, fp)

# the handler of the mouse click

def on_click(x, y, button, pressed):
    global dataset_path
    if button == Button.left:
        print("left button of mouse is pressed at:", x, y)
        add_new_data(dataset_path, webcam_img, x, y)
        print("new data is added")
        thread = threading.Thread(name='retrain-Thread', target=retrain_thread, args=())
        thread.start()
        print("retrain thread is created and started")
        thread.join()

def retrain_thread():
    global modelx, modely, dataset_path, color_object, model_version
    print("Thread retrain is starting")
    X, y_x, y_y = load_dataset(dataset_path)
    print("shape of dataset: ", len(X))

    y = []
    for i in range(len(y_x)):
        y.append((y_x[i], y_y[i]))

    modelx, modely = get_trained_models(X, y, type="linear", save_name=model_version+"_retrain")    #make it general
    color_object.change()
    print("model has been updated")
    print("Thread retrain is finishing")


#------------------------------------------------------main------------------------------------
#------------------global variables----------------------
dataset_path = "pictures/stable2_retrain2"
predictor_path = "shape_predictor_68_face_landmarks.dat" # =args["shape_predictor"]
model_version = "linear_stable2"

#-----------------------------------------------------------------
filenamex = 'models/modelx_'+model_version+'.sav'
filenamey = 'models/modely_'+model_version+'.sav'

color_object = pointer_color()
color = color_object.get_color()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

modelx = pickle.load(open(filenamex, 'rb'))
modely = pickle.load(open(filenamey, 'rb'))


# creating the canvas for showing the results
user32 = ctypes.windll.user32
width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
blank_image = np.ones((height, width, 3), np.uint8)

cv2.namedWindow("full_window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("full_window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow("full_window", blank_image)


listener = Listener(on_click=on_click)
listener.start()

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
    limg = limg.flatten()

    rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    rimg = cv2.equalizeHist(rimg)
    cv2.imshow("rightt", rimg)
    rimg = np.array(rimg)
    rimg = rimg.flatten()

    img = np.concatenate((limg, rimg))

    #-------------------geting the prediction------------------
    predx = round(modelx.predict([img])[0])
    predy = round(modely.predict([img])[0])

    cv2.circle(blank_image, (predx, predy), 10, color_object.get_color(), 5)
    cv2.imshow("full_window", blank_image)
    blank_image = np.ones((height, width, 3), np.uint8)

    #--------------------evaluate------------------------------
    result_totall = get_precision(mousex, mousey, predx, predy)
    result_x = get_precision(mousex, 0, predx, 0)
    result_y = get_precision(0, mousey, 0, predy)
    '''
    print("x & y: ", mousex, mousey, "-", predx, predy, "->", result_totall)
    print("x: ", mousex, "-", predx, "->", result_x)
    print("y: ", mousey, "-", predy, "->", result_y)
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print("Results:")
print("x & y: ",result_totall)
print("x: ", result_x)
print("y: ", result_y)
