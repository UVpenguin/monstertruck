import cv2
import time
import numpy as np
from picamera2 import Picamera2  # type: ignore


def detect_shape(contour):
    """
    Detects the shape of a contour using its approximated polygon and convexity.
    Returns a shape name such as "triangle", "rectangle", "circle", or "arrow".
    """
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    elif vertices == 6:
        shape = "hexagon"
    elif vertices > 5:
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull)
            except cv2.error:
                defects = None
            if defects is not None:
                if 2 <= defects.shape[0] <= 3:
                    shape = "arrow"
                else:
                    shape = "circle"
            else:
                shape = "circle"
        else:
            shape = "circle"
    return shape, approx


def get_arrow_direction(contour):
    """
    Determines the rough orientation of an arrow using a minimum area rectangle.
    Returns a string indicating the direction ("up", "down", "left", or "right").
    """
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    (w, h) = rect[1]
    if w < h:
        direction = "up" if angle < 45 else "down"
    else:
        direction = "right" if angle < 45 else "left"
    return direction


# Initialize camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(preview_config)
picam2.start()
time.sleep(2)

cropped_contours = []
thresholded_images = []

while True:
    # Capture original frame
    frame = picam2.capture_array()

    # Preprocessing: convert to grayscale, threshold and erode.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_val, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cropped_contours.clear()

    # Find contours.
    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        shape_name, approx = detect_shape(contour)
        label = shape_name

        if shape_name == "arrow":
            direction = get_arrow_direction(contour)
            label += " (" + direction + ")"

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (cX - 50, cY),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # Create a mask for the contour.
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Get bounding rectangle of the contour.
        x, y, w, h = cv2.boundingRect(contour)

        # Crop both the frame and mask to the bounding rectangle.
        cropped_frame = frame[y : y + h, x : x + w]
        cropped_mask = mask[y : y + h, x : x + w]

        # Apply the mask to the cropped frame.
        cropped_exact = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)
        cropped_contours.append(cropped_exact)

        # Further process: convert to grayscale and apply threshold.
        gray_crop = cv2.cvtColor(cropped_exact, cv2.COLOR_BGR2GRAY)
        # Using Otsu's method to determine threshold value automatically.
        ret, thresh_crop = cv2.threshold(
            gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresholded_images.append(thresh_crop)

        # Detect contours in the thresholded cropped image.
        crop_contours, crop_hierarchy = cv2.findContours(
            thresh_crop.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_to_use = None
        if shape_name == "circle" and crop_hierarchy is not None:
            # Compute outer centroid relative to the cropped ROI.
            outer_cx, outer_cy = (w // 2, h // 2)
            candidates = []
            for idx, hier in enumerate(crop_hierarchy[0]):
                if hier[3] != -1:  # inner contour
                    M_inner = cv2.moments(crop_contours[idx])
                    if M_inner["m00"] != 0:
                        inner_cx = int(M_inner["m10"] / M_inner["m00"])
                        inner_cy = int(M_inner["m01"] / M_inner["m00"])
                    else:
                        inner_cx, inner_cy = 0, 0
                    dist = np.sqrt(
                        (inner_cx - outer_cx) ** 2 + (inner_cy - outer_cy) ** 2
                    )
                    candidates.append((dist, crop_contours[idx]))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                contour_to_use = candidates[0][1]

        if contour_to_use is None and crop_contours:
            contour_to_use = max(crop_contours, key=cv2.contourArea)

        # Convert threshold crop to BGR to draw colored markers.
        thresh_crop_color = cv2.cvtColor(thresh_crop, cv2.COLOR_GRAY2BGR)

        arrow_direction = "N/A"  # Default if not computed
        if contour_to_use is not None:
            # Extract the four extreme points.
            leftmost = tuple(contour_to_use[contour_to_use[:, :, 0].argmin()][0])
            rightmost = tuple(contour_to_use[contour_to_use[:, :, 0].argmax()][0])
            topmost = tuple(contour_to_use[contour_to_use[:, :, 1].argmin()][0])
            bottommost = tuple(contour_to_use[contour_to_use[:, :, 1].argmax()][0])

            # Draw circles at the extreme points.
            cv2.circle(thresh_crop_color, leftmost, 3, (0, 0, 255), -1)
            cv2.circle(thresh_crop_color, rightmost, 3, (0, 0, 255), -1)
            cv2.circle(thresh_crop_color, topmost, 3, (0, 0, 255), -1)
            cv2.circle(thresh_crop_color, bottommost, 3, (0, 0, 255), -1)
            # Optionally, draw lines connecting them.
            cv2.line(thresh_crop_color, leftmost, topmost, (255, 0, 0), 1)
            cv2.line(thresh_crop_color, topmost, rightmost, (255, 0, 0), 1)
            cv2.line(thresh_crop_color, rightmost, bottommost, (255, 0, 0), 1)
            cv2.line(thresh_crop_color, bottommost, leftmost, (255, 0, 0), 1)

            # Now, compute the arrow orientation using these four points.
            pts = np.array([leftmost, rightmost, topmost, bottommost])
            center = np.mean(pts, axis=0)  # centroid of extreme points
            # Compute Euclidean distances from the center.
            dists = np.linalg.norm(pts - center, axis=1)
            # The arrow tip is assumed to be the point farthest from the center.
            arrow_tip = pts[np.argmax(dists)]

            # For computing angle, adjust for image coordinate system.
            dx = arrow_tip[0] - center[0]
            dy = center[1] - arrow_tip[1]  # invert y so that upward is positive
            angle = np.degrees(np.arctan2(dy, dx))

            # Determine direction based on angle.
            # For example, assume:
            # - Right: angle between -45 and 45
            # - Up: angle between 45 and 135
            # - Left: angle above 135 or below -135
            # - Down: angle between -135 and -45
            if -45 <= angle <= 45:
                arrow_direction = "left"
            elif 45 < angle <= 135:
                arrow_direction = "down"
            elif angle > 135 or angle < -135:
                arrow_direction = "right"
            elif -135 <= angle < -45:
                arrow_direction = "up"

            # Annotate the processed crop with the computed direction.
            cv2.putText(
                thresh_crop_color,
                f"Dir: {arrow_direction}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # Use the processed crop for display.
        display_crop = thresh_crop_color
        break

    if thresh_crop is not None:
        cv2.imshow("Thresholded Crops", display_crop)
    else:
        # Show a blank image if no crop is available.
        cv2.imshow("Thresholded Crops", np.zeros((100, 100), dtype=np.uint8))

    # Display
    cv2.imshow("Detected Shapes", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
