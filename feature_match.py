import cv2
import imutils
import numpy as np
import os
from picamera2 import Picamera2  # type: ignore
import utility

# load templates once (correct unpacking!)
_, tmpl_kps, tmpl_des, tmpl_sizes, tmpl_names = utility.loadTemplates()

# configure PiCamera2
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(cfg)
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        label, scene_pts = utility.findMatch(
            gray, tmpl_kps, tmpl_des, tmpl_sizes, tmpl_names
        )
        if not label:
            label = "No Match"

        # only refine arrows using your extreme?point method
        elif "arrow" in label and scene_pts is not None:
            h_f, w_f = frame.shape[:2]
            pts = scene_pts.reshape(-1, 2)
            x, y, w_box, h_box = cv2.boundingRect(pts)

            # clamp ROI inside frame
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(x + w_box, w_f)
            y2 = min(y + h_box, h_f)

            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]

                # isolate arrow with Otsu
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(
                    gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                cnts, _ = cv2.findContours(
                    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)

                    # your extreme?point method:
                    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

                    pts4 = np.array(
                        [leftmost, rightmost, topmost, bottommost], dtype=np.float32
                    )
                    center = pts4.mean(axis=0)

                    # find the tip as the extreme?point farthest from center
                    dists = np.linalg.norm(pts4 - center, axis=1)
                    tip_pt = pts4[np.argmax(dists)]

                    # compute angle (invert y so that up is positive)
                    dx = tip_pt[0] - center[0]
                    dy = center[1] - tip_pt[1]
                    angle = np.degrees(np.arctan2(dy, dx))

                    # map to directions per your rules
                    if -45 <= angle <= 45:
                        arrow_dir = "left"
                    elif 45 < angle <= 135:
                        arrow_dir = "down"
                    elif angle > 135 or angle < -135:
                        arrow_dir = "right"
                    else:  # -135 <= angle < -45
                        arrow_dir = "up"

                    label = f"{label} ({arrow_dir})"

        # finally draw
        cv2.putText(
            frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) in (27, ord("q")):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
