import cv2
import time
import numpy as np
from picamera2 import Picamera2  # type: ignore


def detect_shape(contour):
    """
    Detects the shape of a contour using its approximated polygon and convexity.
    Returns a shape name such as "triangle", "rectangle", "circle", or "arrow".
    """
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
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull)
            except cv2.error:
                defects = None
            # require exactly 2–3 defects for an arrow
            if defects is not None and 2 <= defects.shape[0] <= 3:
                shape = "arrow"
            else:
                shape = "circle"
        else:
            shape = "circle"

    return shape, approx


def classify_color(avg_color):
    """
    Classifies the average color (B, G, R) as 'red', 'green', or 'blue'
    based on the dominant channel.
    """
    b, g, r = avg_color
    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    else:
        return "undefined"


# Initialize camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(preview_config)
picam2.start()
time.sleep(2)

while True:
    frame = picam2.capture_array()
    if frame is None:
        continue

    # 1) Threshold & clean up
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
    )

    # 2) Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # skip tiny noise
            continue

        shape_name, approx = detect_shape(cnt)
        label = shape_name

        # compute centroid for labeling
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # --- YOUR FOUR‐EXTREME‐POINT ARROW DIRECTION METHOD ---
        if shape_name == "arrow":
            # get the four extreme points
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            pts4 = np.array(
                [leftmost, rightmost, topmost, bottommost], dtype=np.float32
            )
            center = pts4.mean(axis=0)

            # tip = the extreme point farthest from center
            dists = np.linalg.norm(pts4 - center, axis=1)
            tip_pt = pts4[np.argmax(dists)]

            # compute angle (invert Y so up is positive)
            dx = tip_pt[0] - center[0]
            dy = center[1] - tip_pt[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # map angle → direction
            if -45 <= angle <= 45:
                direction = "right"
            elif 45 < angle <= 135:
                direction = "up"
            elif angle > 135 or angle < -135:
                direction = "left"
            else:  # -135 <= angle < -45
                direction = "down"

            label = f"{label} ({direction})"
        # -------------------------------------------------------

        # draw the contour and label
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (cX - 40, cY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # compute and annotate avg color
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        avg_color = cv2.mean(frame, mask=mask)[:3]
        color_name = classify_color(tuple(map(int, avg_color)))
        cv2.putText(
            frame,
            color_name,
            (cX - 40, cY + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

        break  # only show the first detected shape per frame

    # display results
    cv2.imshow("Thresholded", thresh)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
