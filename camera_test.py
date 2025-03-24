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

# Desired white pixel percentage (in ROI)
WHITE_MIN = 3   # in percent
WHITE_MAX = 12  # in percent

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
    """
    height = image.shape[0]
    y_start = int(height * ROI_Y_START_RATIO)
    roi = image[y_start:height, :]
    return roi

def compute_white_percentage(roi):
    """
    Computes the percentage of white (non-zero) pixels in the ROI.
    """
    white_pixels = cv2.countNonZero(roi)
    total_pixels = roi.shape[0] * roi.shape[1]
    perc = (white_pixels / total_pixels) * 100
    return perc

def balance_pic(gray_image, threshold_value):
    """
    Dynamically adjust the threshold value until the white percentage in the ROI
    falls between WHITE_MIN and WHITE_MAX.
    Returns the binary ROI image and the updated threshold value.
    """
    direction = 0  # 1 means increasing threshold; -1 means decreasing
    adjusted_threshold = threshold_value

    for i in range(TH_ITERATIONS):
        # Apply thresholding on the entire grayscale image first
        _, binary = cv2.threshold(gray_image, adjusted_threshold, 255, cv2.THRESH_BINARY)
        roi = crop_roi(binary)
        perc = compute_white_percentage(roi)
        # If white percentage is too high, increase threshold
        if perc > WHITE_MAX:
            if adjusted_threshold >= THRESHOLD_MAX:
                break
            # If we already tried lowering, then stop
            if direction == -1:
                break
            adjusted_threshold += 10
            direction = 1
        # If white percentage is too low, decrease threshold
        elif perc < WHITE_MIN:
            if adjusted_threshold <= THRESHOLD_MIN:
                break
            if direction == 1:
                break
            adjusted_threshold -= 10
            direction = -1
        else:
            # Percentage is within desired range
            break
    # Final binary ROI using the adjusted threshold
    _, final_binary = cv2.threshold(gray_image, adjusted_threshold, 255, cv2.THRESH_BINARY)
    roi_final = crop_roi(final_binary)
    return roi_final, adjusted_threshold

# ---------- Main Loop ----------
current_threshold = THRESHOLD_INIT

while True:
    # Capture frame from PiCamera2
    frame = picam2.capture_array()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adjust threshold dynamically on ROI
    binary_roi, current_threshold = balance_pic(gray_frame, current_threshold)

    # Find contours within the ROI
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Use the largest contour (assumed to be the line)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            # ROI width and center
            roi_width = binary_roi.shape[1]
            roi_center = roi_width // 2

            # Draw a circle at the contour's center for debug purposes
            cv2.circle(binary_roi, (cX, binary_roi.shape[0]//2), 5, (128, 128, 128), -1)

            # Determine turning action based on center position of the contour
            if cX < roi_center - CENTER_MARGIN:
                left()
            elif cX > roi_center + CENTER_MARGIN:
                right()
            else:
                forward()
        else:
            stop()
    else:
        # Stop if no contour is detected
        stop()

    # Debug: show the binary ROI image
    cv2.imshow("ROI Binary", binary_roi)

    # Allow a small delay and exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup resources
cv2.destroyAllWindows()
GPIO.cleanup()
