import cv2
import numpy as np
import os
import glob
import time


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

# Load templates in grayscale for faster processing
templates = []
for file in template_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        templates.append((file, img))
    else:
        print(f"Warning: Could not load {file}")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Allow the camera to warm up
time.sleep(2)

threshold = 0.7
frame_count = 0

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Frame capture failed.")
        break

    # Resize frame to reduce processing load
    frame = cv2.resize(frame, (320, 240))

    # Check if the frame is already grayscale (2D) or in color (3D)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame  # Frame is already grayscale

    frame_count += 1
    if frame_count % 3 == 0:  # Process every third frame to reduce load
        for file, template in templates:
            best_val = -1
            best_loc = None
            best_scale = 1.0

            # Loop through different scales
            for scale in np.linspace(0.5, 1.2, 10):
                try:
                    resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                except cv2.error as e:
                    print("Resize error:", e)
                    continue

                h, w = resized_template.shape

                # Skip if the resized template is larger than the frame
                if gray_frame.shape[0] < h or gray_frame.shape[1] < w:
                    continue

                # Perform template matching on the grayscale image
                result = cv2.matchTemplate(
                    gray_frame, resized_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale

            # Draw rectangle if match exceeds threshold and a valid match is found
            if best_val >= threshold and best_loc is not None:
                resized_template = cv2.resize(
                    template, (0, 0), fx=best_scale, fy=best_scale
                )
                h, w = resized_template.shape
                top_left = best_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                label = f"{os.path.basename(file)}: {best_val:.2f}"

                # Draw on the original frame if it's in color, or on gray_frame if not
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.rectangle(gray_frame, top_left, bottom_right, 255, 2)
                    cv2.putText(
                        gray_frame,
                        label,
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        255,
                        2,
                    )

    # Display the appropriate frame
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        cv2.imshow("Live Feed", frame)
    else:
        cv2.imshow("Live Feed", gray_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
