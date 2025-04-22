import cv2
from picamera2 import Picamera2  # type: ignore
import numpy as np
import utility

# load templates once
tmpl_kps, tmpl_des, tmpl_sizes, names = utility.loadTemplates()

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
        label, scene_pts = utility.findMatch(gray, templ_kps, templ_des, names)
        if not label:
            label = "No Match"
        if not label:
            label = "No Match"
        else:
            # if it's an arrow template, refine the direction
            if "arrow" in label:
                # compute bounding rect of the matched polygon (scene_pts)
                pts = scene_pts.reshape(-1, 2)
                x, y, w_box, h_box = cv2.boundingRect(pts)
                roi = frame[y : y + h_box, x : x + w_box]
                # simple threshold to isolate white arrow on colored background
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(
                    gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                # find the largest contour (the arrow)
                cnts, _ = cv2.findContours(
                    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    # approximate polygon to find the tip
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                    # helper to compute interior angle at vertex i
                    def interior_angle(a, b, c):
                        ba = a - b
                        bc = c - b
                        cosang = (ba.dot(bc)) / (
                            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
                        )
                        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

                    # find the vertex with the sharpest angle → arrow tip
                    tip_pt = None
                    min_ang = 180
                    for i in range(len(approx)):
                        p_prev = approx[(i - 1) % len(approx)][0]
                        p = approx[i][0]
                        p_next = approx[(i + 1) % len(approx)][0]
                        ang = interior_angle(p_prev, p, p_next)
                        if ang < min_ang:
                            min_ang = ang
                            tip_pt = p

                    if tip_pt is not None:
                        # contour centroid
                        M_cnt = cv2.moments(cnt)
                        cx = int(M_cnt["m10"] / M_cnt["m00"])
                        cy = int(M_cnt["m01"] / M_cnt["m00"])
                        dx = tip_pt[0] - cx
                        dy = tip_pt[1] - cy
                        # decide direction
                        if abs(dx) > abs(dy):
                            label = "right arrow" if dx > 0 else "left arrow"
                        else:
                            label = "down arrow" if dy > 0 else "up arrow"

        cv2.putText(
            frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) in (27, ord("q")):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
