import RPi.GPIO as GPIO
import time


HIGH = 1
LOW = 0


enA = 12
enB = 32
in1 = 7
in2 = 11
in3 = 13
in4 = 15

pwmA = GPIO.PWM(enA, 1000)
pwmB = GPIO.PWM(enB, 1000)


def setup():
    """sets up motor output pins and pwm frequency and duty cycle"""
    GPIO.setup(enA, GPIO.OUT)
    GPIO.setup(enB, GPIO.OUT)
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(in3, GPIO.OUT)
    GPIO.setup(in4, GPIO.OUT)

    pwmA.start(100)
    pwmB.start(100)


def forward():
    """Move the motor forward"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, HIGH)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)
    time.sleep(1)
    stop()


def right():
    """Turn the motor right"""
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)
    time.sleep(1)
    stop()


def backward():
    """Move the motor backward"""
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, LOW)
    GPIO.output(in4, HIGH)
    time.sleep(1)
    stop()


def left():
    """Turn the motor left"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, HIGH)
    GPIO.output(in3, LOW)
    GPIO.output(in4, HIGH)
    time.sleep(1)
    stop()


def stop():
    """Stop the motor"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, LOW)
    GPIO.output(in3, LOW)
    GPIO.output(in4, LOW)


def left90():
    """Turn the motor leftF by 90 degrees. uses measured time delays"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, HIGH)
    GPIO.output(in3, LOW)
    GPIO.output(in4, HIGH)
    wait(0.5)

    stop()


def right90():
    """Turn the motor right by 90 degrees. uses measured time delays"""
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)
    wait(0.5)
    stop()


def wait(x: float):
    finish = time.time() + x
    while time.time() < finish:
        pass
