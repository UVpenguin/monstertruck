import cv2
import numpy as np
import os
import glob
import time
from picamera2 import Picamera2


def get_template_images(folder):
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return files


templates_folder = "templates"
template_files = get_template_images(templates_folder)
if not template_files:
    print("No template images found in folder:", templates_folder)
    exit()

templates = []
for file in template_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img.astype(np.uint8)
        templates.append((file, img))
    else:
        print(f"Warning: Could not load {file}")

picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(preview_config)
picam2.start()

time.sleep(2)
threshold = 0.7
frame_width, frame_height = 320, 240  # Match resize dimensions

while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (frame_width, frame_height))

    if len(frame.shape) == 3:
        if frame.shape[2] == 4:
            gray_frame = cv2.cvtColor(frame[:, :, 1:4], cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    gray_frame = cv2.convertScaleAbs(gray_frame).astype(np.uint8)

    for file, template in templates:
        best_val = -1
        best_loc = None
        best_scale = 1.0

        for scale in np.linspace(0.2, 2.0, 20):  # Wider scale range
            try:
                resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            except cv2.error:
                continue

            h, w = resized_template.shape
            # Skip templates that are too large relative to frame
            if h >= frame_height or w >= frame_width:
                continue

            result = cv2.matchTemplate(
                gray_frame, resized_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale

        if best_val >= threshold and best_loc is not None:
            resized_template = cv2.resize(
                template, (0, 0), fx=best_scale, fy=best_scale
            )
            h, w = resized_template.shape
            top_left = best_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Draw on color frame
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            label = f"{os.path.basename(file)}: {best_val:.2f}"
            cv2.putText(
                frame,
                label,
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
