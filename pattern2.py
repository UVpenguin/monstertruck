import cv2
import numpy as np
import os
import glob
import time
from picamera2 import Picamera2


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (
        (box1[2] - box1[0]) * (box1[3] - box1[1])
        + (box2[2] - box2[0]) * (box2[3] - box2[1])
        - inter_area
    )
    return inter_area / union_area if union_area else 0


def non_max_suppression(detections, iou_thresh=0.4):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: -x[0])
    keep = []
    suppressed = set()

    for i in range(len(detections)):
        if i in suppressed:
            continue
        keep.append(detections[i])
        for j in range(i + 1, len(detections)):
            if compute_iou(detections[i][1], detections[j][1]) > iou_thresh:
                suppressed.add(j)
    return keep


def get_template_pyramid(template, scales):
    pyramids = []
    h, w = template.shape
    for scale in scales:
        if scale <= 0:
            continue
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w < 20 or new_h < 20:
            continue
        resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramids.append((new_w, new_h, resized))
    return pyramids


def load_templates(folder):
    templates = []
    for path in glob.glob(os.path.join(folder, "**/*.jpg"), recursive=True):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        edged = cv2.Canny(blurred, 50, 150)
        templates.append(
            (
                os.path.basename(path),
                edged.astype(np.uint8),
                get_template_pyramid(edged, SCALE_RANGE),
            )
        )
    return templates


# Initialize camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (320, 240)}
)
picam2.configure(preview_config)
picam2.start()
time.sleep(2)

# Configuration
SCALE_RANGE = np.logspace(np.log10(0.3), np.log10(1.5), num=10, base=10.0)
THRESHOLD = 0.68
IOU_THRESHOLD = 0.4
GAUSSIAN_KERNEL = (5, 5)
MIN_EDGE_THRESHOLD = 50
MAX_EDGE_THRESHOLD = 150

# Load templates with precomputed pyramids
templates = load_templates("templates")
if not templates:
    print("No templates found")
    exit()


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edged = cv2.Canny(blurred, MIN_EDGE_THRESHOLD, MAX_EDGE_THRESHOLD)

    detections = []

    for name, _, pyramids in templates:
        best_score = 0
        best_bbox = None

        for w, h, tpl in pyramids:
            if w > frame.shape[1] or h > frame.shape[0]:
                continue

            res = cv2.matchTemplate(edged, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score and max_val > THRESHOLD:
                best_score = max_val
                best_bbox = (max_loc[0], max_loc[1], max_loc[0] + w, max_loc[1] + h)

        if best_bbox:
            detections.append((best_score, best_bbox, name))

    return non_max_suppression(detections, IOU_THRESHOLD)


while True:
    frame = picam2.capture_array()
    filtered = process_frame(frame)

    for score, (x1, y1, x2, y2), name in filtered:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name}: {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
