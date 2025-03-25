import RPi.GPIO as GPIO
import movement as motor
import cv2 as cv
import numpy as np
from picamera2 import Picamera2

MARGIN = 100

## GPIO CLEANUP
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

## GPIO SETUP
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

## CAMERA SETUP
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()


try:
    while True:
        # captures frame data from camera
        frame = picam2.capture_array()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        contours, _ = cv.findContours(
            gray_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            M = cv.moments(largest_contour)

            if M["m00"] > 0:
                # Find centroid of contour
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # draw largest contour
                cv.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)  # mark center

                # find center of region of interest (contour)
                roi_width = gray_frame.shape[1]
                roi_center = roi_width // 2

                # move logic
                if cX < roi_center - MARGIN:
                    motor.left()
                elif cX > roi_center + MARGIN:
                    motor.right()
                else:
                    motor.forward()
        else:
            motor.stop()

        cv.imshow("Gray Frame", gray_frame)
        cv.imshow("Line Detection (Original Frame)", frame)

        # Allow a small delay and exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        cv.waitKey(10)

except KeyboardInterrupt:  # Cleanup
    motor.stop()
    pwmA.stop()
    pwmB.stop()
    cv.destroyAllWindows()
    GPIO.cleanup()
