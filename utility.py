import cv2
import imutils
import numpy as np
import os

# 1) Tuned ORB detector for low‑light, simple shapes
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.1,
    nlevels=12,
    edgeThreshold=5,
    fastThreshold=7,
    scoreType=cv2.ORB_HARRIS_SCORE,
)

# 2) Hamming‐distance BFMatcher (no crossCheck here, we’ll use ratio test)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# 3) CLAHE (adaptive histogram equalization) for contrast boost
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def readImages():
    images, names = [], []
    base = os.path.join(os.getcwd(), "templates")
    for root, _, files in os.walk(base):
        for file in files:
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = imutils.resize(img, width=400)
            images.append(img)
            names.append(os.path.splitext(file)[0])
    return images, names


def getDescriptors(images):
    descriptors = []
    for img in images:
        _, des = orb.detectAndCompute(img, None)
        descriptors.append(des)
    return descriptors


def _preprocess(gray):
    # resize to match template scale
    gray = imutils.resize(gray, width=400)
    # boost local contrast
    gray = clahe.apply(gray)
    # mild denoising
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def findMatch(gray_frame, descriptors, names, thresh=4, ratio=0.9):
    """
    Returns the template name with the most good ORB matches, or "" if none pass thresh.
      - thresh: minimum # of ratio‐test matches
      - ratio: Lowe’s ratio test threshold (loosened for symmetric shapes)
    """
    gray = _preprocess(gray_frame)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < 4:
        return ""

    best_score = 0
    best_name = ""
    for templ_des, name in zip(descriptors, names):
        if templ_des is None:
            continue

        raw = bf.knnMatch(des, templ_des, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        score = len(good)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name if best_score > thresh else ""
