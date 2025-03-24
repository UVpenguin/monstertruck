from picamera2 import Picamera2
import RPi.GPIO as GPIO
import cv2
from movement import left, right, forward, stop  # Import motor control functions

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

camera = 19

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


while True:
    frame = picam2.capture_array()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(
        gray_frame, 120, 255, cv2.THRESH_BINARY
    )  # Thresholding to create binary image

    contours, _ = cv2.findContours(
        binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:  # Check if any contours are found

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            frame_center = frame.shape[1] // 2

            if cX < frame_center - 20:
                # Turn left
                left()
            elif cX > frame_center + 20:
                # Turn right
                right()
            else:
                # Move forward
                forward()
        else:
            # Stop if no contour is found

            stop()
    else:
        # Stop if no contours detected
        stop()

    cv2.imshow("Camera Feed", frame)

    # Press 'q' to exit the loop

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
