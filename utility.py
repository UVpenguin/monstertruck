import cv2
import imutils
import numpy as np
import os

orb = cv2.ORB_create(nfeatures=1000)


def readImages():
    images, names = [], []
    base = os.path.join(os.getcwd(), "templates")
    for root, _, files in os.walk(base):
        for imgfile in files:
            img = cv2.imread(os.path.join(root, imgfile), 0)
            img = imutils.resize(img, width=400)
            images.append(img)
            names.append(os.path.splitext(imgfile)[0])
    return images, names


def getDescriptors(images):
    return [orb.detectAndCompute(img, None)[1] for img in images]


def findMatch(gray_frame, descriptors, names, thresh=5):
    # match at the same scale as your templates
    gray = imutils.resize(gray_frame, width=400)

    kps, des = orb.detectAndCompute(gray, None)
    if des is None:
        return ""  # no features

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    scores = []
    for templ_des in descriptors:
        if templ_des is None:
            scores.append(0)
            continue

        # knn + ratio test
        matches = bf.knnMatch(des, templ_des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        scores.append(len(good))

    # pick best
    best = max(scores)
    if best > thresh:
        return names[scores.index(best)]
    return ""


# ---- in your main loop: ----
# frame = picam2.capture_array()
# name  = findMatch(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), descriptors, names)
