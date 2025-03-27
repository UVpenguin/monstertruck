import cv2
import numpy as np
import os
import glob
import time
from picamera2 import Picamera2  # type: ignore


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


def detect_arrow(image):
    """
    Improved arrow detection with precise direction identification

    Returns:
    - False if no arrow detected
    - Tuple (bool, str) where first element is arrow presence,
      and second element is arrow direction
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    cv2.imshow("lines", lines)
    cv2.imshow("edges", edges)

    if lines is None or len(lines) < 2:
        return (False, "")

    # Collect line information
    line_info = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle in degrees
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Store line with leftmost point first to normalize direction
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        line_info.append((angle_deg, x1, y1, x2, y2))

    # Look for converging lines (arrowhead pattern)
    arrowhead_found = False
    direction = ""

    # Check for arrowhead formation
    for i in range(len(line_info)):
        for j in range(i + 1, len(line_info)):
            angle1, x1_1, y1_1, x2_1, y2_1 = line_info[i]
            angle2, x1_2, y1_2, x2_2, y2_2 = line_info[j]

            # Check if lines converge
            angle_diff = np.abs(angle1 - angle2)
            if angle_diff < 90:
                arrowhead_found = True

                # Determine arrow direction more precisely
                # Vertical detection (Up/Down)
                if abs(angle1) > 60 or abs(angle2) > 60:
                    # Check if points are moving vertically
                    if y2_1 < y1_1 and y2_2 < y1_2:
                        direction = "UP"
                    elif y2_1 > y1_1 and y2_2 > y1_2:
                        direction = "DOWN"
                # Horizontal detection (Left/Right)
                else:
                    # Check horizontal movement
                    if x2_1 > x1_1 and x2_2 > x1_2:
                        direction = "RIGHT"
                    elif x2_1 < x1_1 and x2_2 < x1_2:
                        direction = "LEFT"

                break

        if arrowhead_found:
            break

    return (arrowhead_found, direction)


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
    direction, arrow_detected = detect_arrow(display_frame)

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
        f"Arrow: {arrow_detected}",
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
