import cv2
import imutils
import numpy as np
import os

# 1) Tuned ORB detector
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.1,
    nlevels=12,
    edgeThreshold=5,
    fastThreshold=7,
    scoreType=cv2.ORB_HARRIS_SCORE,
)

# 2) Hamming-distance BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# 3) CLAHE for contrast boost
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


def loadTemplates():
    """Load template images and return their keypoints, descriptors, and names."""
    images, names = readImages()
    kps_list, des_list = [], []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        kps_list.append(kp)
        des_list.append(des)
    return kps_list, des_list, names


def _preprocess(gray):
    # 1) resize to template scale
    gray = imutils.resize(gray, width=400)
    # 2) boost local contrast
    gray = clahe.apply(gray)
    # 3) mild denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def findMatch(
    gray_frame, kps_list, des_list, names, min_inliers=4, ratio=0.9, ransac_thresh=5.0
):
    """
    Return the best-matching template name, or "" if none.
      - Requires at least `min_inliers` RANSAC inliers
      - Uses relaxed ratio test `ratio`
      - RANSAC reproj threshold `ransac_thresh`
    """
    gray = _preprocess(gray_frame)
    # detect on scene
    scene_kp, scene_des = orb.detectAndCompute(gray, None)
    if scene_des is None or len(scene_kp) < min_inliers:
        return ""

    best_name, best_inliers = "", 0

    # for each template
    for templ_kp, templ_des, name in zip(kps_list, des_list, names):
        if templ_des is None:
            continue

        # k-NN match + ratio test
        raw = bf.knnMatch(scene_des, templ_des, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < min_inliers:
            continue

        # build points for homography
        src_pts = np.float32([templ_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        # RANSAC homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if mask is None:
            continue

        inliers = int(mask.sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_name = name

    return best_name if best_inliers >= min_inliers else ""
