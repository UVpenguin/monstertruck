import cv2
import numpy as np
import os
import glob
import time
from picamera2 import Picamera2  # type: ignore
import collections


# ================== UPDATED DETECTORS ==================
def detect_shapes(image):
    """Improved shape detection with ROI isolation"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 1000:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        vertices = len(approx)

        # Get rotated bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        x, y = box[0]

        shape = ""
        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            # Calculate aspect ratio using min area rectangle
            (_, _), (w, h), _ = rect
            ar = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            shape = "square" if ar < 1.1 else "rectangle"
        elif vertices == 5:
            shape = "pentagon"
        elif 6 <= vertices <= 8:
            shape = "circle"
        else:
            shape = "complex"

        if shape:
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            cv2.putText(
                image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    return image


# Global history for smoothing arrow direction over the last few frames
arrow_history = collections.deque(maxlen=5)


def get_smoothed_direction(new_direction):
    """Keep a history of directions and return the majority vote."""
    arrow_history.append(new_direction)
    # Get the most common direction in the history
    return max(set(arrow_history), key=arrow_history.count)


def detect_arrow(image, area_threshold=1000):
    """
    Detects an arrow by finding the largest contour, assuming it is the arrow.
    It then computes the centroid and finds the farthest point on the contour as the tip.
    Returns a tuple (detected: bool, direction: str) and overlays an arrow on the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur and threshold to create a binary image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Using THRESH_BINARY_INV because arrows on a light background (or vice versa)
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours and get the largest one
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None

    arrow_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(arrow_contour) < area_threshold:
        return False, None

    # Calculate centroid of the arrow contour
    M = cv2.moments(arrow_contour)
    if M["m00"] == 0:
        return False, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])

    # Find the contour point farthest from the centroid (arrow tip)
    farthest = None
    max_dist = 0
    for point in arrow_contour:
        pt = point[0]
        dist = np.linalg.norm(np.array(pt) - centroid)
        if dist > max_dist:
            max_dist = dist
            farthest = pt

    if farthest is None:
        return False, None

    # Determine arrow direction based on the vector from centroid to farthest point
    dx = farthest[0] - cx
    dy = farthest[1] - cy
    if abs(dx) > abs(dy):  # Horizontal motion dominates
        direction = "right" if dx > 0 else "left"
    else:  # Vertical motion dominates
        direction = "down" if dy > 0 else "up"

    # Draw the arrow on the image for visualization
    cv2.arrowedLine(image, (cx, cy), tuple(farthest), (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(
        image,
        f"Arrow: {direction}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    return True, direction


def detect_color(image):
    """Color detection with minimum area requirement"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colors = {
        "red": ([0, 100, 100], [10, 255, 255]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "blue": ([100, 50, 50], [140, 255, 255]),
        "yellow": ([20, 100, 100], [30, 255, 255]),
    }

    detected = []
    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Require minimum contiguous area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if any(cv2.contourArea(c) > 500 for c in contours):
            detected.append(color)

    return detected


# ===================================================


# ============== YOUR ORIGINAL CODE ==============
def compute_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return area_inter / (area_box1 + area_box2 - area_inter)


def non_max_suppression(detections, iou_threshold=0.4):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    keep = []

    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [
            det for det in detections if compute_iou(current[1], det[1]) < iou_threshold
        ]
    return keep


def get_template_images(folder):
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))

    templates = []
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.Canny(img, 50, 150)
            templates.append((os.path.basename(file), img.astype(np.uint8)))
    return templates


# Initialize camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(preview_config)
picam2.start()
time.sleep(2)

# Configuration
FRAME_SIZE = (320, 240)
SCALE_RANGE = np.linspace(0.3, 1.5, 15)
THRESHOLD = 0.65
IOU_THRESHOLD = 0.4

templates = get_template_images("templates")
if not templates:
    print("No valid templates found")
    exit()

while True:
    # Capture original frame
    original_frame = picam2.capture_array()
    original_frame = cv2.resize(original_frame, FRAME_SIZE)

    # Create working copies
    template_frame = original_frame.copy()
    display_frame = original_frame.copy()

    # Run template matching on original frame
    gray = cv2.cvtColor(template_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(gray, 50, 150)

    # Run shape/color/arrow detection on display frame
    color_info = detect_color(display_frame)
    display_frame = detect_shapes(display_frame)
    detected, direction = detect_arrow(display_frame)
    if detected and direction is not None:
        smooth_direction = get_smoothed_direction(direction)
    else:
        smooth_direction = "None"

    # Add informational overlay
    cv2.putText(
        display_frame,
        f"Colors: {', '.join(color_info)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    cv2.putText(
        display_frame,
        f"Arrow: {smooth_direction}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    # ... (rest of your template matching code using template_frame)

    # Show final results
    cv2.imshow("Multi Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
