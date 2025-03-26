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

pwmA.start(70)
pwmB.start(70)

# CAMERA SETUP


def process_frame(frame):
    """
    Convert the captured frame to grayscale, blur it to reduce noise,
    and run Canny edge detection.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Apply Gaussian blur to smooth the image
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Use Canny to detect edges; adjust thresholds as needed.
    edges = cv.Canny(blur, 50, 150)
    return edges


def detect_line(edges):
    """
    Use the probabilistic Hough transform to detect line segments.
    Returns the average angle of detected lines and the lines array.
    """
    # Parameters for HoughLinesP: distance resolution = 1 pixel, angle resolution = 1 degree (in radians),
    # threshold for minimum intersections, minLineLength, and maxLineGap.
    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
    )

    if lines is None:
        return None, None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the angle (in degrees) of the line segment relative to the horizontal
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        angles.append(angle)

    if angles:
        avg_angle = np.mean(angles)
        return avg_angle, lines
    return None, lines


def adjust_motors(avg_angle, tolerance=5):
    """
    Based on the average angle of the detected line, steer the robot.
      - If the angle is near 0 (within tolerance), drive forward.
      - If the angle is negative, steer left.
      - If the angle is positive, steer right.
    """
    print(f"Average angle: {avg_angle:.2f}")
    if abs(avg_angle) < tolerance:
        forward()
    elif avg_angle < 0:
        left()
    else:
        right()


def main():
    # Initialize the Picamera2 instance and start the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    time.sleep(2)  # allow the camera to warm up

    try:
        while True:
            # Capture a frame as a numpy array
            frame = picam2.capture_array()

            # Process frame for edge detection
            edges = process_frame(frame)

            # Detect the line using Hough transform
            avg_angle, lines = detect_line(edges)

            # If a line is detected, adjust motors based on the average angle;
            # otherwise, stop the robot.
            if avg_angle is not None:
                adjust_motors(avg_angle)
            else:
                stop()
                print("No line detected, stopping.")

            # For debugging: show the edge-detected image (optional)
            cv.imshow("Edges", edges)
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
