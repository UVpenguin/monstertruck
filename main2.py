import RPi.GPIO as GPIO
from movement import forward, left, right, stop
import movement as motor
import cv2 as cv
from picamera2 import Picamera2  # type: ignore
import numpy as np
import time


# GPIO CLEANUP
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# GPIO SETUP
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

pwmA.start(60)
pwmB.start(60)


def crop_frame(frame, x_start=150, x_end=430, y_start=80, y_end=475):
    return frame[y_start:y_end, x_start:x_end]


def preprocess(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    return binary


def detect_line_direction(binary_img, sample_offset=50, pixel_threshold=128):
    """
    Detect the line direction by sampling a fixed row.
    - binary_img: a binary (thresholded) image of the ROI.
    - sample_offset: vertical offset (in pixels) from the bottom of the cropped image where we sample.
    - pixel_threshold: threshold for deciding if a pixel is considered 'black' (part of the line).

    Returns the angle (in degrees) between the vertical and the vector from the fixed bottom-center
    to the midpoint of the line edges found at the sample row.
    If no line is detected, returns None.
    """
    height, width = binary_img.shape
    # The sample row is some pixels above the bottom.
    sample_row = height - sample_offset
    if sample_row < 0:
        sample_row = height - 1

    # Get the row of pixels.
    row_pixels = binary_img[sample_row, :]

    # Find indices where pixel value is below the pixel_threshold (i.e. part of the black line).
    # (Since thresholding set dark areas to 0.)
    line_indices = np.where(row_pixels < pixel_threshold)[0]

    if line_indices.size == 0:
        return None

    # Assume the left and right edges of the line are the first and last detected indices.
    left_edge = line_indices[0]
    right_edge = line_indices[-1]
    # Calculate the midpoint of the line at the sample row.
    line_mid_x = (left_edge + right_edge) // 2

    # Define a fixed bottom-center point of the cropped image.
    fixed_x = width // 2
    fixed_y = height  # bottom of the image

    # The vector from the fixed point to the detected line midpoint.
    dx = line_mid_x - fixed_x
    dy = sample_offset  # vertical difference (from fixed_y to sample_row)

    # Calculate the angle in radians relative to vertical.
    # Since dy is the known vertical distance, we use arctan2(dx, dy).
    angle_rad = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)

    # debugging
    cv.circle(binary_img, (line_mid_x, sample_row), 3, (127,), -1)
    cv.line(binary_img, (fixed_x, height), (line_mid_x, sample_row), (127,), 2)

    return angle_deg


def adjust_motors(avg_angle, tolerance=30):
    """
    - If the angle is near 0 (within tolerance), go forward.
    - If the angle is negative, go left.
    - If the angle is positive, go right.
    """
    print(f"Average angle: {avg_angle:.2f}")
    if abs(avg_angle) < tolerance:
        forward()
    elif avg_angle < 0:
        left()
    else:
        right()


def main():
    # Initialize the Picamera2 instance
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    time.sleep(2)

    try:
        while True:
            frame = picam2.capture_array()
            cropped_frame = crop_frame(frame)
            binary_img = preprocess(cropped_frame)

            angle = detect_line_direction(binary_img, sample_offset=50)

            if angle is not None:
                adjust_motors(angle)
            else:
                stop()
                print("No line detected, stopping.")

            cv.imshow("Binary Image", binary_img)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        motor.stop()
        pwmA.stop()
        pwmB.stop()
        cv.destroyAllWindows()
        GPIO.cleanup()

    finally:
        cv.destroyAllWindows()
        stop()


if __name__ == "__main__":
    main()
