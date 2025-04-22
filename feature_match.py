import cv2
import numpy as np
import os

# Initialize the ORB detector and BFMatcher (Hamming distance for ORB's binary descriptors)
orb = cv2.ORB_create(nfeatures=500)  # You can adjust nfeatures for performance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load template images and compute their ORB descriptors
template_dir = "templates"
template_files = [
    "blue circle down arrow.jpg",
    "blue circle up arrow.jpg",
    "blue rectangle.jpg",
    "blue rectangle left arrow.jpg",
    "blue rectangle right arrow.jpg",
    "blue triangle.jpg",
    "green hexagon.jpg",
    "green semi-circle.jpg",
    "red circle.jpg",
    "red pentagon.jpg",
]
templates = []
for filename in template_files:
    path = os.path.join(template_dir, filename)
    label = os.path.splitext(filename)[0]  # use filename (without extension) as label
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load template image {path}")
        continue
    # Compute ORB keypoints and descriptors for the template
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        print(f"No features found in template {label}, skipping.")
        continue
    # Store the template's data for matching
    templates.append(
        {
            "label": label,
            "image": img,
            "keypoints": kp,
            "descriptors": des,
            "size": img.shape[::-1],  # (width, height) for drawing bounding box
        }
    )

# Set up video capture (assuming a camera is connected)
cap = cv2.VideoCapture(0)  # Use the appropriate camera index or video file
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

MIN_MATCH_COUNT = 4  # minimum number of matches to consider a template detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for feature detection
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    if gray_frame is None or gray_frame.size == 0:
        print("Warning: Skipping empty frame.")
        continue

    # Detect ORB keypoints and descriptors in the current frame
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    if des_frame is None:
        # No features found in frame (likely a blank or very uniform frame); skip processing
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:  # press 'ESC' to exit
            break
        continue

    # Iterate over each template and attempt to find it in the frame
    for template in templates:
        des_template = template["descriptors"]
        kp_template = template["keypoints"]
        label = template["label"]
        template_size = template["size"]  # (width, height)

        # Match template descriptors to frame descriptors using BFMatcher
        matches = bf.match(des_template, des_frame)
        if not matches:
            continue
        # Sort matches by distance (lower distance = better match)
        matches = sorted(matches, key=lambda x: x.distance)

        # Use the top matches (e.g., top 20) for homography to improve robustness
        good_matches = matches[:20]
        # Proceed only if we have enough good matches
        if len(good_matches) >= MIN_MATCH_COUNT:
            # Extract matched keypoint coordinates in both template and frame
            src_pts = np.float32(
                [kp_template[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # Compute homography matrix to find the transformation from template to scene
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                # Get template corners (top-left, top-right, bottom-right, bottom-left)
                w, h = template_size  # width, height of template image
                template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(
                    -1, 1, 2
                )
                # Transform template corners to frame coordinates using homography
                scene_corners = cv2.perspectiveTransform(template_corners, M)
                scene_corners = np.int32(
                    scene_corners
                )  # convert to integer pixel coordinates

                # Draw a polygon (bounding box) around the detected template in the frame
                cv2.polylines(
                    frame,
                    [scene_corners],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                # Put label text near the top-left corner of the detected polygon
                x, y, w_box, h_box = cv2.boundingRect(scene_corners)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2,
                )

    # Display the frame with drawn matches (bounding boxes and labels)
    cv2.imshow("Frame", frame)
    # Exit on pressing 'q' or 'ESC'
    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
