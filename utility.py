import cv2
import imutils
import numpy as np
import os

# Tuned ORB detector
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.1,
    nlevels=12,
    edgeThreshold=5,
    fastThreshold=7,
    scoreType=cv2.ORB_HARRIS_SCORE,
)

# Hamming-distance matcher for ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# CLAHE for contrast boost
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def loadTemplates():
    """
    Reads grayscale template images from ./templates, resizes to width=400,
    and returns:
      - images: list of ndarray
      - keypoints: list of list of cv2.KeyPoint
      - descriptors: list of ndarray
      - sizes: list of (width, height)
      - names: list of str
    """
    images, kps_list, des_list, sizes, names = [], [], [], [], []
    base = os.path.join(os.getcwd(), "templates")
    for root, _, files in os.walk(base):
        for f in files:
            path = os.path.join(root, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # resize for consistency
            img = imutils.resize(img, width=400)
            images.append(img)
            names.append(os.path.splitext(f)[0])
            sizes.append(img.shape[::-1])  # (w, h)

            # detect ORB features on template once
            kp, des = orb.detectAndCompute(img, None)
            kps_list.append(kp)
            des_list.append(des)

    return images, kps_list, des_list, sizes, names


def _preprocess(gray):
    # 1) scale to match template width
    gray = imutils.resize(gray, width=400)
    # 2) enhance local contrast
    gray = clahe.apply(gray)
    # 3) mild blur to suppress noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def findMatch(
    gray_frame,
    kps_list,
    des_list,
    sizes,
    names,
    min_inliers=4,
    ratio=0.9,
    ransac_thresh=5.0,
):
    """
    Returns (best_name, scene_corners) or ("", None).
    scene_corners is a 4×1×2 array of the template corners transformed into the frame.
    Requires at least `min_inliers` RANSAC inliers to accept.
    """
    gray = _preprocess(gray_frame)
    scene_kp, scene_des = orb.detectAndCompute(gray, None)
    if scene_des is None or len(scene_kp) < min_inliers:
        return "", None

    best_name = ""
    best_scene_corners = None
    best_inliers = 0

    # iterate templates
    for templ_kp, templ_des, templ_size, name in zip(kps_list, des_list, sizes, names):
        if templ_des is None:
            continue

        # match descriptors
        raw = bf.knnMatch(scene_des, templ_des, 2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < min_inliers:
            continue

        # prepare points for homography
        src_pts = np.float32([templ_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if mask is None:
            continue

        inliers = int(mask.sum())
        if inliers >= min_inliers and inliers > best_inliers:
            best_inliers = inliers
            best_name = name
            # compute scene corners of template
            w, h = templ_size
            tmpl_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(
                -1, 1, 2
            )
            best_scene_corners = cv2.perspectiveTransform(tmpl_corners, M)

    return (
        (best_name, best_scene_corners)
        if best_scene_corners is not None
        else ("", None)
    )
