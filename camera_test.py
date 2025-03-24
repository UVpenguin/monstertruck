from picamera2 import Picamera2
import RPi.GPIO as GPIO
import cv2
import numpy as np
import movement as motor
from movement import left, right, forward, stop  # Motor control functions
import time

# ---------- Configuration Parameters ----------
# Initial threshold and limits for dynamic adjustment
THRESHOLD_INIT = 120
THRESHOLD_MAX = 180
THRESHOLD_MIN = 40
TH_ITERATIONS = 10

# Desired “white” pixel percentage in the ROI (but this “white” is actually the black line in the inverted image)
LINE_MIN = 3   # in percent
LINE_MAX = 12  # in percent

# Region of interest parameters (process lower part of the image)
ROI_Y_START_RATIO = 0.6  # Start cropping from 60% of frame height

# Control decision margin (in pixels)
CENTER_MARGIN = 20

# ---------- Setup Camera and GPIO ----------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

enA = motor.enA
enB = motor.enB
in1 = motor.in1
in2 = motor.in2
in3 = motor.in3
in4 = motor.in4

GPIO.setup(enA, GPIO.OUT)
GPIO.setup(enB, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

pwmA = GPIO.PWM(enA, 1000)
pwmB = GPIO.PWM(enB, 1000)
pwmA.start(100)
pwmB.start(100)

# ---------- Helper Functions ----------

def crop_roi(image):
    """
    Crop the region of interest (ROI) from the grayscale image.
    We take the bottom part of the image where the line is expected.
    Returns the cropped image and the offset (y_start).
    """
    height = image.shape[0]
    y_start = int(height * ROI_Y_START_RATIO)
    roi = image[y_start:height, :]
    return roi, y_start

def compute_line_percentage(roi):
    """
    Computes the percentage of non-zero (white) pixels in the ROI.
    In our inverted image, these “white” pixels represent the black line.
    """
    line_pixels = cv2.countNonZero(roi)
    total_pixels = roi.shape[0] * roi.shape[1]
    perc = (line_pixels / total_pixels) * 100
    return perc

def balance_threshold_for_black_line(gray_image, threshold_value):
    """
    Dynamically adjust the threshold value until the percentage of the black line
    (which appears white in the inverted thresholded image) in the ROI
    falls between LINE_MIN and LINE_MAX.

    Returns:
      - roi_final: the thresholded (inverted) ROI
      - adjusted_threshold: the updated threshold value
      - y_start: offset for the cropped ROI
    """
    direction = 0  # 1 means increasing threshold; -1 means decreasing
    adjusted_threshold = threshold_value

    for _ in range(TH_ITERATIONS):
        # Apply inverted threshold to detect black as white
        _, binary_full = cv2.threshold(gray_image, adjusted_threshold, 255, cv2.THRESH_BINARY_INV)
        roi, _ = crop_roi(binary_full)
        perc = compute_line_percentage(roi)

        # If line percentage is too high, we increase threshold
        if perc > LINE_MAX:
            if adjusted_threshold >= THRESHOLD_MAX:
                break
            if direction == -1:
                break
            adjusted_threshold += 10
            direction = 1

        # If line percentage is too low, we decrease threshold
        elif perc < LINE_MIN:
            if adjusted_threshold <= THRESHOLD_MIN:
                break
            if direction == 1:
                break
            adjusted_threshold -= 10
            direction = -1

        else:
            # Percentage is within desired range, no need to adjust more
            break

    # Apply the final adjusted threshold
    _, final_binary_full = cv2.threshold(gray_image, adjusted_threshold, 255, cv2.THRESH_BINARY_INV)
    roi_final, y_start = crop_roi(final_binary_full)
    return roi_final, adjusted_threshold, y_start

# ---------- Main Loop ----------
current_threshold = THRESHOLD_INIT

try:
    while True:
        # Capture frame from PiCamera2
        frame = picam2.capture_array()

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dynamically adjust threshold for the bottom ROI to detect the black line
        binary_roi, current_threshold, y_start = balance_threshold_for_black_line(
            gray_frame, current_threshold
        )

        # Find contours in the inverted-threshold ROI
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Use the largest contour (assumed to be the black line)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] > 0:
                # Calculate center of contour in ROI coordinates
                cX_roi = int(M["m10"] / M["m00"])
                cY_roi = int(M["m01"] / M["m00"])

                # Convert ROI coordinates to full image coordinates
                cX = cX_roi
                cY = cY_roi + y_start

                # For debug: draw the contour on the original frame in color
                shifted_contour = []
                for point in largest_contour:
                    x, y = point[0]
                    shifted_contour.append([[x, y + y_start]])
                shifted_contour = np.array(shifted_contour)

                cv2.drawContours(frame, [shifted_contour], -1, (0, 255, 0), 3)
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)  # Mark center

                # Determine turning action based on center position
                roi_width = binary_roi.shape[1]
                roi_center = roi_width // 2

                if cX_roi < roi_center - CENTER_MARGIN:
                    left()
                elif cX_roi > roi_center + CENTER_MARGIN:
                    right()
                else:
                    forward()
            else:
                stop()
        else:
            # Stop if no contour is detected
            stop()

        # Show both the binary ROI (for debugging) and the main frame with color contour
        cv2.imshow("ROI Binary (Inverted)", binary_roi)
        cv2.imshow("Line Detection (Original Frame)", frame)

        # Allow a small delay and exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Cleanup resources
    cv2.destroyAllWindows()
    GPIO.cleanup()
