import RPi.GPIO as GPIO
from movement import forward, left, right, stop
import movement as motor
import cv2 as cv
from picamera2 import Picamera2  # type: ignore
import numpy as np
from time import sleep

bool lineFound

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
servo = 33

GPIO.setup(enA, GPIO.OUT)
GPIO.setup(enB, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(servo, GPIO.OUT)

pwmA = GPIO.PWM(enA, 1000)
pwmB = GPIO.PWM(enB, 1000)
pwmServo = GPIO.PWM(servo, 50)

pwmA.start(60)
pwmB.start(60)


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

def move_servo()
    pwmServo.start(0)
    pwmServo.ChangeDutyCycle(5)
    sleep(1)
    pwmServo.ChangeDutyCycle(2.5)
    sleep(1)
    

def main():
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
                adjust_motors(angle)
            else:
                stop()
                print("No line detected, stopping.")
                while angle is None: # while no angle is found, move servo and search for one
                    move_servo()
                pwmServo.stop() 
                

            cv.imshow("Binary Image", binary_img)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        motor.stop()
        pwmA.stop()
        pwmB.stop()
        pwmServo.stop()
        cv.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
