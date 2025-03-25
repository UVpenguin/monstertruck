import RPi.GPIO as GPIO
import movement as motor
import cv2 as cv
from picamera2 import Picamera2  # type: ignore

MARGIN = 80

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
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration()
preview_config["size"] = (150, 150)
preview_config["framerate"] = 60
picam2.configure(preview_config)
picam2.start()


try:
    while True:
        # captures frame data from camera
        frame = picam2.capture_array()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        lsd = cv.createLineSegmentDetector(0)

        lines = lsd.detect(gray_frame)[0]

        drawn_img = lsd.drawSegments(frame, lines)

        cv.imshow("LSD Detection", drawn_img)
        cv.imshow("Line Detection (Original Frame)", frame)

        # Allow a small delay and exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:  # Cleanup
    motor.stop()
    pwmA.stop()
    pwmB.stop()
    cv.destroyAllWindows()
    GPIO.cleanup()
