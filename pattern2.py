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


# Get templates from folder
templates_folder = "templates"
template_files = get_template_images(templates_folder)
if not template_files:
    print("No template images found in folder:", templates_folder)

# Load templates into a list of (filename, image)
templates = []
for file in template_files:
    img = cv2.imread(file)
    if img is not None:
        templates.append((file, img))
    else:
        print(f"Warning: Could not load {file}")

# Open the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

threshold = 0.7

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Frame capture failed.")
        break

    frame = cv2.resize(frame, (320, 240))

    # For each template, do multi-scale matching
    for file, template in templates:
        best_val = -1
        best_loc = None
        best_scale = 1.0

        for scale in np.linspace(0.1, 1.5, 10):
            try:
                resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            except cv2.error as e:
                print("Resize error:", e)
                continue

            h, w, _ = resized_template.shape

            # Skip if resized template is larger than the frame
            if frame.shape[0] < h or frame.shape[1] < w:
                continue

            # Template matching on the blue channel
            result = cv2.matchTemplate(
                frame[:, :, 0], resized_template[:, :, 0], cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale

        # Draw rectangle if match exceeds the threshold and best_loc is valid
        if best_val >= threshold and best_loc is not None:
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

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
