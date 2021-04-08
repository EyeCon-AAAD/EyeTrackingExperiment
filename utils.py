from imutils import paths
import cv2
import ntpath
import numpy as np
import pickle
import math

def point2grid(x, y, xd, yd,  w= 1920, h = 1080):

    loc_x = x//(w//xd)
    loc_y = y//(h//yd)

    grid = 0
    for i in range(yd):
        for j in range(xd):
            if loc_x == j and loc_y == i:
                return grid
            grid += 1


def grid2point(predicted, xd, yd, width= 1920, heigth = 1080):
    grid = -1
    xstep = width // xd
    ystep = heigth // yd

    initx =  xstep//2
    inity =  ystep//2

    for i in range(yd):
        y = inity + i * ystep
        for j in range(xd):
            grid += 1
            x = initx + j * xstep
            if predicted == grid:
                return [x, y]


def draw_grid(img, pxstep=50, pystep =50, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pystep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pystep

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)


def load_dataset(path, flatten=True):
    print("loading data set at:", path)
    with open(path + "/coordinates.txt", "rb") as fp:  # Unpickling
        coordinates = pickle.load(fp)

    imagePaths = list(paths.list_images(path))

    def customComparator(e):
        return e[0]

    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail

    paired_paths = []
    for left, right in grouped(imagePaths, 2):
        file_name = path_leaf(left)
        i = int(file_name.split("_")[0])
        paired_paths.append((i, (left, right)))

    paired_paths = sorted(paired_paths, key=customComparator)
    print(paired_paths)

    X = []
    y_x = []
    y_y = []
    for (i, (left, right)), (j, (x, y)) in zip(paired_paths, coordinates):
        if (i != j):
            print("Error i={} j={}".format(i, j))
            exit(1)
        limg = cv2.imread(left)
        limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        limg = cv2.equalizeHist(limg)
        if flatten:
            limg = np.array(limg)
            limg = limg.flatten()
            # print(limg.shape)

        rimg = cv2.imread(right)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        rimg = cv2.equalizeHist(rimg)
        if flatten:
            rimg = np.array(rimg)
            rimg = rimg.flatten()
            # print(rimg.shape)

        y_x.append(x)
        y_y.append(y)

        img = np.concatenate((limg, rimg))

        X.append(img)

    print("dataset loaded at:", path)
    return X, y_x, y_y

def update_counter(path, counter):
	with open(path, 'w') as f:
		f.write('%d' % counter)

def get_counter(path):
	with open(path, 'r') as f:
		input = f.readline()
		return int(input)

def distance1D(x1, x2):
    return abs(x1-x2)

def distance2D(x1, y1, x2, y2):
    return math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

class pointer_color():
    def __init__(self):
        self.pointer_color_list = [(255, 0, 0), (0,255,0)]
        self.index = 0
        self.color = self.pointer_color_list[self.index]
    def change(self):
        self.index = 1 - self.index
        self.color = self.pointer_color_list[self.index]
    def get_color(self):
        return self.color