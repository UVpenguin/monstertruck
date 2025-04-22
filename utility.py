# utility.py
import cv2
import imutils
import numpy as np
import os

# tuned ORB for low‑light & simple shapes
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.1,
    nlevels=12,
    edgeThreshold=5,
    fastThreshold=7,
    scoreType=cv2.ORB_HARRIS_SCORE,
)

# Hamming matcher for ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# CLAHE for contrast boost
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def readImages():
    images, names = [], []
    base = os.path.join(os.getcwd(), "templates")
    for root, _, files in os.walk(base):
        for f in files:
            img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = imutils.resize(img, width=400)
            images.append(img)
            names.append(os.path.splitext(f)[0])
    return images, names


def getDescriptors(images):
    return [orb.detectAndCompute(img, None)[1] for img in images]


def _preprocess(gray):
    # 1) scale to template width
    gray = imutils.resize(gray, width=400)
    # 2) boost local contrast
    gray = clahe.apply(gray)
    # 3) mild blur to suppress noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def findMatch(
    gray_frame,
    descriptors,
    names,
    min_inliers=4,  # require ≥4 RANSAC inliers
    ratio=0.9,  # relaxed ratio test
    ransac_thresh=5.0,
):  # reprojection threshold
    """
    Returns the name of the best‑matching template, or "" if none
    have at least `min_inliers` geometrically consistent matches.
    """
    gray = _preprocess(gray_frame)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < min_inliers:
        return ""  # no chance

    best_name, best_inliers = "", 0

    for templ_des, tname in zip(descriptors, names):
        if templ_des is None:
            continue

        # KNN + ratio test
        raw = bf.knnMatch(des, templ_des, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < min_inliers:
            continue  # not enough matches to bother

        # build point arrays for homography
        src_pts = np.float32(
            [orb.detectAndCompute(gray, None)[0][m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [orb.detectAndCompute(gray, None)[0][m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        # find homography & count inliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if mask is None:
            continue

        inliers = int(mask.sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_name = tname

    return best_name if best_inliers >= min_inliers else ""
