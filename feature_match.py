import cv2
import time
import numpy as np
from picamera2 import Picamera2  # type: ignore


# --- Your existing helper functions ---
def detect_shape(contour):
    # ... (same as your earlier detect_shape)
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
    # ... (same as your earlier get_arrow_direction)
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
        return "right"
    elif 45 < angle <= 135:
        return "up"
    elif angle > 135 or angle < -135:
        return "left"
    else:
        return "down"


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


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

# --- ROI offset trackers ---
x_off, y_off, w_off, h_off = 0, 0, 0, 0

while True:
    # Capture full frame
    full_frame = picam2.capture_array()
    if full_frame is None:
        continue

    # Apply last ROI crop if available
    if w_off > 0 and h_off > 0:
        frame = full_frame[y_off : y_off + h_off, x_off : x_off + w_off]
    else:
        frame = full_frame.copy()

    # Preprocess on the (possibly cropped) frame
    processed = preprocess(frame)
    contours, hierarchy = cv2.findContours(
        processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        cv2.imshow("Detection", full_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    hierarchy = hierarchy[0]

    # Find first significant contour
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 1000:
            continue

        # Detect shape & color
        shape, poly = detect_shape(cnt)
        label = shape
        if shape == "arrow":
            dirn = get_arrow_direction(cnt)
            label += f" ({dirn})"

        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] else 0

        cv2.drawContours(frame, [poly], -1, (0, 255, 0), 2)
        cv2.putText(
            frame, label, (cX - 40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

        # Color
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

        # Compute local bounding box, then update global offsets
        x_rel, y_rel, w_rel, h_rel = cv2.boundingRect(cnt)
        if w_off > 0 and h_off > 0:
            x_off += x_rel
            y_off += y_rel
        else:
            x_off, y_off = x_rel, y_rel
        w_off, h_off = w_rel, h_rel

        # Draw on full_frame (with offset)
        # translate poly points to full_frame coords
        pts = poly + np.array([[x_off - x_rel, y_off - y_rel]])
        cv2.drawContours(full_frame, [pts], -1, (0, 255, 0), 2)
        cv2.putText(
            full_frame,
            label,
            (x_off + cX - 40, y_off + cY),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        break  # only first contour

    # Show results
    cv2.imshow("Thresholded", processed)
    cv2.imshow("Detection", full_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
