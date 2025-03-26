import RPi.GPIO as GPIO
import time

servo_pin = 19

# GPIO setup
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)


pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)


def set_angle(angle):
    duty_cycle = (angle / 18) + 2
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)


try:
    while True:
        set_angle(180)
        time.sleep(2)
        set_angle(90)
        time.sleep(2)
        set_angle(180)
        time.sleep(2)

except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
