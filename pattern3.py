import cv2
import numpy as np
import os
import glob
import time
from picamera2 import Picamera2  # type: ignore


# ================== NEW ADDITIONS ==================
def detect_shapes(image):
    """Recognize basic shapes using contour analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        vertices = len(approx)

        # Get bounding box coordinates for all shapes
        x, y, w, h = cv2.boundingRect(approx)

        shape = ""
        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            ar = w / float(h)
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif vertices == 5:
            shape = "pentagon"
        elif vertices >= 6:
            shape = "circle"

        if shape:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            # Use the bounding box coordinates for text placement
            cv2.putText(
                image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
    return image


def detect_color(image):
    """Identify dominant colors in HSV space"""
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
        if cv2.countNonZero(mask) > 1000:  # Minimum pixel threshold
            detected.append(color)
    return detected


def detect_arrow(image):
    """Simple arrow detection using line geometry"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    # Properly handle numpy array output
    if lines is not None and len(lines) >= 2:
        return True
    return False


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
    # Capture and basic processing
    frame = picam2.capture_array()
    frame = cv2.resize(frame, FRAME_SIZE)

    # ============== NEW PROCESSING ==============
    # Run parallel detection systems
    color_info = detect_color(frame)
    frame = detect_shapes(frame)
    arrow_detected = detect_arrow(frame)

    # Add informational overlay
    cv2.putText(
        frame,
        f"Colors: {', '.join(color_info)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Arrow: {arrow_detected}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    # ============================================

    # Your original template matching pipeline
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(gray, 50, 150)
    detections = []

    for name, template in templates:
        template_h, template_w = template.shape
        best_score = 0
        best_bbox = None

        for scale in SCALE_RANGE:
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)
            if (
                scaled_w < 20
                or scaled_h < 20
                or scaled_w > FRAME_SIZE[0]
                or scaled_h > FRAME_SIZE[1]
            ):
                continue

            resized = cv2.resize(
                template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
            result = cv2.matchTemplate(processed_frame, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_bbox = (
                    max_loc[0],
                    max_loc[1],
                    max_loc[0] + scaled_w,
                    max_loc[1] + scaled_h,
                )

        if best_score >= THRESHOLD and best_bbox:
            detections.append((best_score, best_bbox, name))

    filtered = non_max_suppression(detections, IOU_THRESHOLD)

    # Draw template matches
    for score, (x1, y1, x2, y2), name in filtered:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name}: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imshow("Multi Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
