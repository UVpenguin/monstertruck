import cv2
import numpy as np
import os
import glob


def get_template_images(folder):
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return files


templates_folder = "templates"
template_files = get_template_images(templates_folder)

# load the templates into a list as (filename, image) tuples
templates = []
for file in template_files:
    img = cv2.imread(file)
    if img is not None:
        templates.append((file, img))
    else:
        print(f"Warning: Could not load {file}")


camera = cv2.VideoCapture(0)

threshold = 0.7

while True:
    ret, frame = camera.read()
    if not ret:
        break

    for file, template in templates:
        best_val = -1
        best_loc = None
        best_scale = 1.0

        for scale in np.linspace(0.1, 1.5, 20):
            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            h, w, _ = resized_template.shape

            # if the resized template is larger than the frame, skip it
            if frame.shape[0] < h or frame.shape[1] < w:
                continue

            # template matching on blue channel
            result = cv2.matchTemplate(
                frame[:, :, 0], resized_template[:, :, 0], cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # recordsthe best match
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale

        # if the best match for the current template exceeds the threshold, draw the rectangle and label it
        if best_val >= threshold:
            resized_template = cv2.resize(
                template, (0, 0), fx=best_scale, fy=best_scale
            )
            h, w, _ = resized_template.shape
            top_left = best_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
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

camera.release()
cv2.destroyAllWindows()
