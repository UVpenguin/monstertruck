import cv2
import imutils
import numpy as np
import os

orb = cv2.ORB_create(nfeatures=1000)


# Loading images from folder reading and storing names
def readImages():
    images = []
    names = []
    path = os.getcwd()
    path = os.path.join(path, "templates")
    for root, dirs, files in os.walk(path):
        for image in files:
            img = cv2.imread(os.path.join(path, image), 0)  # Reading in grayscale
            img = imutils.resize(img, width=400)
            images.append(img)
            names.append(image.split(".")[0])
    return images, names


# Find descriptors of all images and store in list
def getDescriptors(images):
    descriptors = []
    for image in images:
        kps, des = orb.detectAndCompute(image, None)
        descriptors.append(des)
    return descriptors


# Get matching descriptors for frame with features > threshold value
def findMatch(gray_frame, descriptors, names, thresh=15):
    kps, des = orb.detectAndCompute(gray_frame, None)
    scores = []
    bf = cv2.BFMatcher()
    try:  # May be no feature detected in frame
        for descriptor in descriptors:
            matches = bf.knnMatch(
                des, descriptor, k=2
            )  # using knn to match descriptors k=2 as we apply ratio test further
            good = []
            for m, n in matches:  # Finding good matches
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            scores.append(len(good))
    except:
        pass
    name = ""
    if len(scores) != 0:  # return name of book ie with most good features matched
        if max(scores) > thresh:
            name = names[scores.index(max(scores))]
    return name
