import cv2
import time
import numpy as np
from picamera2 import Picamera2  # type: ignore

# --- Your existing helper functions ---
# (detect_shape, get_arrow_direction, classify_color, preprocess)
# [These remain unchanged]


def detect_shape(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    elif vertices == 6:
        shape = "hexagon"
    elif vertices > 6:
        shape = "circle"

    return shape, approx


def get_arrow_direction(contour):
    # same as before
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    pts = np.array([leftmost, rightmost, topmost, bottommost])
    center = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    arrow_tip = pts[np.argmax(dists)]

    dx = arrow_tip[0] - center[0]
    dy = center[1] - arrow_tip[1]
    angle = np.degrees(np.arctan2(dy, dx))

    if -45 <= angle <= 45:
        return "left"
    elif 45 < angle <= 135:
        return "down"
    elif angle > 135 or angle < -135:
        return "right"
    else:
        return "up"


def classify_color(avg_color):
    b, g, r = avg_color
    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    else:
        return "undefined"


def color_prepreprocess(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, (36, 50, 50), (70, 255, 255))
    red_mask = cv2.inRange(hsv, (0, 50, 50), (30, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 150, 0), (140, 255, 255))

    color = cv2.bitwise_and(frame, frame, mask=green_mask | red_mask | blue_mask)
    return color


def preprocess(frame):
    color = color_prepreprocess(frame)
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2
    )
    return thresh


# --- Camera setup ---
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(preview_config)
picam2.start()
time.sleep(2)


while True:
    # 1) Capture full frame
    frame = picam2.capture_array()
    if frame is None:
        continue

    # 3) Preprocess & find contours
    processed_frame = preprocess(frame)
    contours, hierarchy = cv2.findContours(
        processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 1000 and cv2.contourArea(cnt) < 200:
                continue

            # --- Shape detection & annotation ---
            outer_label, outer_poly = detect_shape(cnt)

            found_arrow = False
            arrow_label = None

            # scan all child contours of this outer one
            child_idx = [j for j, h in enumerate(hierarchy) if h[3] == i]
            for j in child_idx:
                inner = contours[j]
                inner_area = cv2.contourArea(inner)
                if inner_area < 50:
                    continue

                shape, approx_inner = detect_shape(inner)
                if shape:
                    direction = get_arrow_direction(inner)

                    arrow_label = f"arrow ({direction})"
                    found_arrow = True
                    break

            # decides what arrow and label to use
            label = arrow_label if found_arrow else outer_label
            poly = outer_poly if not found_arrow else approx_inner

            cv2.drawContours(frame, [poly], -1, (0, 255, 0), 2)
            # compute centroid
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] else 0

            # draw frame
            cv2.drawContours(frame, [poly], -1, (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (cX - 40, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # color
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            avg = cv2.mean(frame, mask=mask)[:3]
            color = classify_color(tuple(map(int, avg)))
            cv2.putText(
                frame,
                color,
                (cX - 40, cY + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            break

    # 5) Show
    cv2.imshow("Thresholded", processed_frame)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
