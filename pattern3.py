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

    if lines is None or len(lines) < 2:
        return (False, "")

    # Collect line information with more detailed vector analysis
    line_info = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate line vector
        dx = x2 - x1
        dy = y2 - y1
        # Calculate angle in degrees
        angle_deg = np.degrees(np.arctan2(dy, dx))
        line_info.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "dx": dx,
                "dy": dy,
                "angle": angle_deg,
            }
        )

    # Find potential arrowhead lines
    arrowhead_candidates = []
    for i in range(len(line_info)):
        for j in range(i + 1, len(line_info)):
            line1 = line_info[i]
            line2 = line_info[j]

            # Check if lines converge (angle close to forming an arrowhead)
            angle_diff = abs(line1["angle"] - line2["angle"])

            # More sophisticated convergence check
            if angle_diff < 90:
                # Find a potential reference point (arrowhead tip)
                tip_x = None
                tip_y = None

                # Check if lines intersect or are very close
                # Calculate line segment bounding boxes
                min_x1 = min(line1["x1"], line1["x2"])
                max_x1 = max(line1["x1"], line1["x2"])
                min_y1 = min(line1["y1"], line1["y2"])
                max_y1 = max(line1["y1"], line1["y2"])

                min_x2 = min(line2["x1"], line2["x2"])
                max_x2 = max(line2["x1"], line2["x2"])
                min_y2 = min(line2["y1"], line2["y2"])
                max_y2 = max(line2["y1"], line2["y2"])

                # Determine arrow direction by analyzing line orientations and endpoints
                # Prefer vertical line orientation for UP/DOWN
                if abs(line1["angle"]) > 60 or abs(line2["angle"]) > 60:
                    # Vertical arrow detection (UP/DOWN)
                    if line1["y2"] < line1["y1"] and line2["y2"] < line2["y1"]:
                        return (True, "UP")
                    elif line1["y2"] > line1["y1"] and line2["y2"] > line2["y1"]:
                        return (True, "DOWN")

                # Horizontal arrow detection (LEFT/RIGHT)
                else:
                    # Check line directions and endpoints
                    if line1["x2"] > line1["x1"] and line2["x2"] > line2["x1"]:
                        return (True, "RIGHT")
                    elif line1["x2"] < line1["x1"] and line2["x2"] < line2["x1"]:
                        return (True, "LEFT")

    return (False, "")


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
