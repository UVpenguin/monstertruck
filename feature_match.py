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


def get_arrow_direction(contour):
    """
    Determines the rough orientation of an arrow using a minimum area rectangle.
    Returns a string indicating the direction ("up", "down", "left", or "right").
    """
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    # Draw circles at the extreme points.
    cv2.circle(contour, leftmost, 3, (0, 0, 255), -1)
    cv2.circle(contour, rightmost, 3, (0, 0, 255), -1)
    cv2.circle(contour, topmost, 3, (0, 0, 255), -1)
    cv2.circle(contour, bottommost, 3, (0, 0, 255), -1)
    # Optionally, draw lines connecting them.
    cv2.line(contour, leftmost, topmost, (255, 0, 0), 1)
    cv2.line(contour, topmost, rightmost, (255, 0, 0), 1)
    cv2.line(contour, rightmost, bottommost, (255, 0, 0), 1)
    cv2.line(contour, bottommost, leftmost, (255, 0, 0), 1)

    # Now, compute the arrow orientation using these four points.
    pts = np.array([leftmost, rightmost, topmost, bottommost])
    center = np.mean(pts, axis=0)  # centroid of extreme points
    # Compute Euclidean distances from the center.
    dists = np.linalg.norm(pts - center, axis=1)
    # The arrow tip is assumed to be the point farthest from the center.
    arrow_tip = pts[np.argmax(dists)]

    # For computing angle, adjust for image coordinate system.
    dx = arrow_tip[0] - center[0]
    dy = center[1] - arrow_tip[1]  # invert y so that upward is positive
    angle = np.degrees(np.arctan2(dy, dx))

    # Determine direction based on angle.
    # For example, assume:
    # - Right: angle between -45 and 45
    # - Up: angle between 45 and 135
    # - Left: angle above 135 or below -135
    # - Down: angle between -135 and -45
    if -45 <= angle <= 45:
        arrow_direction = "left"
    elif 45 < angle <= 135:
        arrow_direction = "down"
    elif angle > 135 or angle < -135:
        arrow_direction = "right"
    elif -135 <= angle < -45:
        arrow_direction = "up"

    return arrow_direction


def preprocess(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    v = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)

    # white mask
    white_mask = cv2.inRange(hsv, (0, 0, 100), (0, 255, 255))
    v_masked = cv2.bitwise_not(v, v, mask=white_mask)
    cv2.imshow("white_mask", v_masked)

    _, thresh = cv2.threshold(v_masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cleanup Edges
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2
    )

    return thresh


# Initialize camera
picam2 = Picamera2()
preview_config = picam2.createconfig = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(preview_config)
picam2.start()
time.sleep(2)

while True:
    frame = picam2.capture_array()
    if frame is None:
        continue

    processed_frame = preprocess(frame)

    contours, hierarchy = cv2.findContours(
        processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        continue

    hierarchy = hierarchy[0]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500 and area < 200:
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
            if inner_area < 50:
                continue

            shape, approx_inner = detect_shape(inner)
            if shape:
                direction = get_arrow_direction(inner)

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
        mask = np.zeros_like(processed_frame)
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

    cv2.imshow("Thresholded", processed_frame)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
