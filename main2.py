import RPi.GPIO as GPIO
import movement as motor
import cv2 as cv
import numpy as np
from time import sleep
from movement import forward, left, right, stop
from picamera2 import Picamera2  # type: ignore
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
import threading


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

# Servo Setup
factory = PiGPIOFactory()
servo = AngularServo(
    "BOARD35", pin_factory=factory, min_pulse_width=0.0006, max_pulse_width=0.0023
)

line_detected_event = threading.Event()


def preprocess(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    return binary


def detect_line_direction(binary_img, sample_offset=50, pixel_threshold=128):

    height, width = binary_img.shape
    sample_row = height - sample_offset
    if sample_row < 0:
        sample_row = height - 1

    row_pixels = binary_img[sample_row, :]

    # Find indices where pixel value is below the pixel_threshold
    line_indices = np.where(row_pixels < pixel_threshold)[0]

    if line_indices.size == 0:
        return None

    # Assume the left and right edges of the line are the first and last detected indices.
    left_edge = line_indices[0]
    right_edge = line_indices[-1]
    # Calculate the midpoint
    line_mid_x = (left_edge + right_edge) // 2

    # Define a fixed bottom-center point
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


def servo_sweep(angle_range=(-45, 45), step=5, delay=0.2):
    angle = angle_range[0]
    direction = step
    while not line_detected_event.is_set():
        servo.angle = angle
        sleep(delay)
        angle += direction
        if angle >= angle_range[1] or angle <= angle_range[0]:
            direction *= -1  # Reverse direction when reaching bounds
    # When the line is detected, reset servo to center (0 degrees)
    sleep(2)
    servo.angle = 0


def main():
    sweep_thread = threading.Thread(target=servo_sweep, args=(servo,), daemon=True)
    sweep_thread.start()
    # Initialize the Picamera2 instance
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    sleep(2)

    try:
        while True:
            frame = picam2.capture_array()
            binary_img = preprocess(frame)

            angle = detect_line_direction(binary_img, sample_offset=50)

            if angle is not None:
                if (
                    not line_detected_event.is_set()
                ):  # once line is found servo sweep is switched off and reset
                    print("Line detected, stopping servo sweep.")
                    line_detected_event.set()
                adjust_motors(angle)
            else:
                stop()
                print("No line detected, stopping.")

                if line_detected_event.is_set():  # sets the servo sweep function off
                    line_detected_event.clear()
                    sweep_thread = threading.Thread(
                        target=servo_sweep, args=(servo,), daemon=True
                    )
                    sweep_thread.start()

            cv.imshow("Binary Image", binary_img)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        motor.stop()
        pwmA.stop()
        pwmB.stop()
        cv.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
