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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
    )

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        continue
    hierarchy = hierarchy[0]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # 1) detect the outer shape
        outer_label, outer_poly = detect_shape(cnt)

        # 2) now look for arrow *inside* this outer contour
        found_arrow = False
        arrow_label = None

        # scan all child contours of this outer one
        child_idx = [j for j, h in enumerate(hierarchy) if h[3] == i]
        for j in child_idx:
            inner = contours[j]
            inner_area = cv2.contourArea(inner)
            if inner_area < 100:
                continue

            shape, approx_inner = detect_shape(inner)
            if shape == "arrow":
                # we found an arrow inside!
                # compute its direction via your 4‑extreme‑points
                pts = approx_inner.reshape(-1, 2)
                leftmost = tuple(inner[inner[:, :, 0].argmin()][0])
                rightmost = tuple(inner[inner[:, :, 0].argmax()][0])
                topmost = tuple(inner[inner[:, :, 1].argmin()][0])
                bottommost = tuple(inner[inner[:, :, 1].argmax()][0])

                pts4 = np.array(
                    [leftmost, rightmost, topmost, bottommost], dtype=np.float32
                )
                center = pts4.mean(axis=0)
                dists = np.linalg.norm(pts4 - center, axis=1)
                tip = pts4[np.argmax(dists)]

                dx = tip[0] - center[0]
                dy = center[1] - tip[1]
                ang = np.degrees(np.arctan2(dy, dx))

                if -45 <= ang <= 45:
                    direction = "right"
                elif 45 < ang <= 135:
                    direction = "up"
                elif ang > 135 or ang < -135:
                    direction = "left"
                else:
                    direction = "down"

                arrow_label = f"arrow ({direction})"
                found_arrow = True
                break

        # choose final label
        label = arrow_label if found_arrow else outer_label

        # draw contour (outer or inner) and label
        poly = outer_poly if not found_arrow else approx_inner
        cv2.drawContours(frame, [poly], -1, (0, 255, 0), 2)

        M = cv2.moments(cnt if not found_arrow else inner)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv2.putText(
            frame,
            label,
            (cX - 40, cY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # annotate color
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        avg = cv2.mean(frame, mask=mask)[:3]
        color = classify_color(tuple(map(int, avg)))
        cv2.putText(
            frame,
            color,
            (cX - 40, cY + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

        break  # only first shape per frame

    cv2.imshow("Thresholded", thresh)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
