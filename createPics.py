# This file creates the pics for the intro video by randomizing the point postion and save the pics to a given file
# Additionally we save the radius of the points and the count of the pictures so we can work on different folders
# with different radiuses and pics count

import pickle
from PIL import Image, ImageDraw
import random
import math

width = 1920
height = 1080
frame = 20 # The frame where it should be always empty
picsCount = 120
pointRadius = 10
givenFolderName = "pics"


# This functions create pics and save them to the given file
def createPics():
    img = Image.new('RGB', (height, width), color="white")

    # This dictionary saves each pictures with the pic name as a key and its coordinates and a value
    # The first to keys are the points radius and the pics count in the file
    centers = {"pointsRadius": pointRadius, "picsCount": picsCount}

    for x in range(0, picsCount):

        # randomizing the point coordinates
        centerX = random.randint(frame, width - frame)
        centerY = random.randint(frame, height - frame)
        img = Image.new('RGB', (width, height), color="white")

        # This to allow making changes to the image
        circle = ImageDraw.Draw(img)

        # This is to draw the points with giving the coordinates of the upper left and bottom right corners
        circle.ellipse((centerX - pointRadius, centerY - pointRadius, centerX + pointRadius, centerY + pointRadius), fill="black",
                       outline="black")
        # img.save("{}\\img{}.png" + str(x) + ".png")
        img.save("{}\\img{}.png".format(givenFolderName, x))
        name = "img{}.png".format(x)
        centers[name] = (centerX, centerY)

        # save the dictionary to the files
        file = open("pics\\centers.pkl", "wb")
        pickle.dump(centers, file)
        file.close()
        file = open("pics\\centers.json", "wb")
        pickle.dump(centers, file)
        file.close()


# This runs the function
createPics()
