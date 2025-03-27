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


def detect_arrow(image, arrow_templates):
    """
    Arrow detection using template matching

    Args:
    - image: Input image
    - arrow_templates: List of template images for different arrow directions

    Returns:
    - Tuple (bool, str) indicating arrow presence and direction
    """
    # Prepare the image for template matching
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Store detections
    detections = []

    # Try to match each arrow template
    for template_name, template in arrow_templates:
        # Perform template matching
        for scale in np.linspace(0.3, 1.5, 15):
            # Resize template
            resized_template = cv2.resize(template, None, fx=scale, fy=scale)

            # Perform template matching
            result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)

            # Find locations above threshold
            locations = np.where(result >= 0.7)

            # Add detections
            for pt in zip(*locations[::-1]):
                # Get bounding box
                h, w = resized_template.shape
                detections.append(
                    (
                        result[pt[1], pt[0]],
                        (pt[0], pt[1], pt[0] + w, pt[1] + h),
                        template_name,
                    )
                )

    # Perform non-maximum suppression to remove overlapping detections
    if detections:
        detections = non_max_suppression(detections, iou_threshold=0.4)

    # Determine arrow presence and direction
    if detections:
        # Prioritize first detection
        _, box, direction = detections[0]

        # Optionally draw rectangle around detected arrow
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            image,
            direction,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        return (True, direction)

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
    """Get template images, with special handling for arrow templates"""
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))

    templates = []
    for file in files:
        # Distinguish arrow templates
        if "arrow" in file.lower():
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Extract direction from filename
                if "up" in file.lower():
                    direction = "UP"
                elif "down" in file.lower():
                    direction = "DOWN"
                elif "left" in file.lower():
                    direction = "LEFT"
                elif "right" in file.lower():
                    direction = "RIGHT"
                else:
                    direction = os.path.splitext(os.path.basename(file))[0]

                templates.append((direction, img.astype(np.uint8)))
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

# Load templates (including arrow templates)
templates = get_template_images("templates")
arrow_templates = [
    t for t in templates if len(t[0]) <= 4
]  # Assume direction-based template names
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

    # Run shape/color/arrow detection on display frame
    color_info = detect_color(display_frame)
    display_frame = detect_shapes(display_frame)

    # Detect arrow using template matching
    arrow_detected, direction = detect_arrow(display_frame, arrow_templates)

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
        f"Arrow: {direction if arrow_detected else 'None'}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    # Show final results
    cv2.imshow("Multi Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
