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
import shape_detection as shape_detect

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

FRAME_OVERRIDE = False


def preprocess(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    return binary


#
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


def color_percentage(hsv, color_mask):
    height, width, _ = hsv.shape
    total_pixels = height * width

    num_color_pixels = np.count_nonzero(color_mask)

    fraction = num_color_pixels / total_pixels
    return fraction


def color_mask_override(frame):
    global FRAME_OVERRIDE

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # mask of green (36,25,25) ~ (86, 255,255)
    # green_mask = cv.inRange(hsv, (30, 50, 50), (70, 255, 255))
    # green = cv.bitwise_and(frame, frame, mask=green_mask)
    # green_color_percentage = color_percentage(green, green_mask)

    # mask of red
    # camera detects blue as red
    red_mask = cv.inRange(hsv, (0, 50, 50), (30, 255, 255))
    red = cv.bitwise_and(frame, frame, mask=red_mask)
    cv.imshow("Blue", red)
    red_color_percentage = color_percentage(red, red_mask)

    # mask of blue
    # blue is actually red
    # blue_mask = cv.inRange(hsv, (100, 150, 0), (179, 255, 255))
    # blue = cv.bitwise_and(frame, frame, mask=blue_mask)
    # cv.imshow("Blue", blue)
    # blue_color_percentage = color_percentage(blue, blue_mask)

    # mask of yellow
    # yellow is actually blue
    yellow_mask = cv.inRange(hsv, (90, 50, 0), (100, 255, 255))
    yellow = cv.bitwise_and(frame, frame, mask=yellow_mask)
    cv.imshow("Yellow", yellow)
    yellow_color_percentage = color_percentage(yellow, yellow_mask)

    # print(
    #     f"Red: {red_color_percentage:.2f}, Green: {green_color_percentage:.2f}, Blue: {blue_color_percentage:.2f}, Yellow: {yellow_color_percentage:.2f}"
    # )

    if red_color_percentage > 0.15:
        FRAME_OVERRIDE = True
        return red
    # if green_color_percentage > 0.15:
    #     FRAME_OVERRIDE = True
    #     return green
    # if blue_color_percentage > 0.15:
    #     FRAME_OVERRIDE = True
    #     frame = blue
    #     return blue
    if yellow_color_percentage > 0.15:
        FRAME_OVERRIDE = True
        frame = yellow
        return yellow
    else:
        FRAME_OVERRIDE = False


def main():
    servo_thread = threading.Thread(target=servo_control, daemon=True)
    servo_thread.start()

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888"}))
    picam2.start()
    sleep(2)

    try:
        while True:

            frame = picam2.capture_array()
            shapes = shape_detect.main(frame)
            override = color_mask_override(frame)
            if not FRAME_OVERRIDE:
                binary_img = preprocess(frame)
            else:
                binary_img = preprocess(override)
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
            cv.imshow("Processed Frame", shapes)
            # cv.imshow("Binary Image", frame)
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
