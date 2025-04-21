import threading
import RPi.GPIO as GPIO
import cv2 as cv
import numpy as np
from time import sleep
from movement import forward, left, right, stop
import movement as motor
from picamera2 import Picamera2  # type: ignore
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo

# Setup GPIO
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

pwmA.start(60)
pwmB.start(60)

# Servo Setup
factory = PiGPIOFactory()
servo = AngularServo(
    "BOARD35", pin_factory=factory, min_pulse_width=0.0006, max_pulse_width=0.0023
)


sweeping_enabled = threading.Event()
sweeping_enabled.clear()


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
    line_indices = np.where(row_pixels < pixel_threshold)[0]

    if line_indices.size == 0:
        return None

    left_edge = line_indices[0]
    right_edge = line_indices[-1]

    line_mid_x = (left_edge + right_edge) // 2
    fixed_x = width // 2

    dx = line_mid_x - fixed_x
    dy = sample_offset

    angle_rad = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)
    # Debug
    # cv.circle(binary_img, (line_mid_x, sample_row), 3, (127,), -1)
    # cv.line(binary_img, (fixed_x, height), (line_mid_x, sample_row), (127,), 2)
    return angle_deg


def adjust_motors(avg_angle, tolerance=45):

    print(f"Average angle: {avg_angle:.2f}")
    if abs(avg_angle) < tolerance:
        forward()
    elif avg_angle < 0:
        left()
    else:
        right()


def servo_control():
    min_angle, max_angle = -65, 65
    step = 10
    delay = 0.2  # delay between steps while sweeping
    current_angle = min_angle
    direction = step

    while True:
        if sweeping_enabled.is_set():
            servo.angle = current_angle
            sleep(delay)
            current_angle += direction
            if current_angle >= max_angle or current_angle <= min_angle:
                direction *= -1  # reverse sweep direction
        else:
            # If not sweeping, ensure servo is centered.
            if servo.angle != 0:
                servo.angle = 0
            sleep(0.1)


def color_masking(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # mask of green (36,25,25) ~ (86, 255,255)
    mask = cv.inRange(hsv, (36, 25, 25), (70, 255, 255))
    green = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Green", green)


def main():
    servo_thread = threading.Thread(target=servo_control, daemon=True)
    servo_thread.start()

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
                # Line detected: disable sweeping so servo stays centered.
                if sweeping_enabled.is_set():
                    print("Line detected, stopping servo sweep.")
                    sweeping_enabled.clear()
                adjust_motors(angle)
            else:
                stop()
                print("No line detected, enabling servo sweep.")
                if not sweeping_enabled.is_set():
                    sweeping_enabled.set()

            cv.imshow("Binary Image", binary_img)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        stop()
        pwmA.stop()
        pwmB.stop()
        cv.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
