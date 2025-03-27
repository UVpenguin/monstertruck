import cv2
import numpy as np
import os
import glob
import time
from picamera2 import Picamera2


def compute_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
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
    """Apply non-maximum suppression to detection results"""
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
    """Load template images with edge detection preprocessing"""
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))

    templates = []
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.Canny(img, 50, 150)  # Edge detection
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
SCALE_RANGE = np.linspace(0.1, 1.5, 15)  # Optimized scale range
THRESHOLD = 0.65  # Increased threshold for better precision
IOU_THRESHOLD = 0.4  # NMS overlap threshold

# Load templates with edge preprocessing
templates = get_template_images("templates")
if not templates:
    print("No valid templates found")
    exit()

while True:
    # Capture and preprocess frame
    frame = picam2.capture_array()
    # frame = cv2.resize(frame, FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(gray, 50, 150)  # Edge detection

    detections = []

    for name, template in templates:
        template_h, template_w = template.shape
        best_score = 0
        best_bbox = None

        for scale in SCALE_RANGE:
            # Skip templates that become too small or too large
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)
            if (
                scaled_w < 20
                or scaled_h < 20
                or scaled_w > FRAME_SIZE[0]
                or scaled_h > FRAME_SIZE[1]
            ):
                continue

            # Resize with edge-preserving interpolation
            resized = cv2.resize(
                template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )

            # Perform template matching
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

        if best_score >= THRESHOLD and best_bbox is not None:
            detections.append((best_score, best_bbox, name))

    # Apply non-maximum suppression
    filtered = non_max_suppression(detections, IOU_THRESHOLD)

    # Draw remaining detections
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

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
