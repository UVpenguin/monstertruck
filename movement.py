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


def forward():
    """Move the motor forward"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, HIGH)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)
    time.sleep(1)


def right():
    """Turn the motor right"""
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)
    time.sleep(1)


def backward():
    """Move the motor backward"""
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, LOW)
    GPIO.output(in4, HIGH)
    time.sleep(1)


def left():
    """Turn the motor left"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, HIGH)
    GPIO.output(in3, LOW)
    GPIO.output(in4, HIGH)
    time.sleep(1)


def stop():
    """Stop the motor"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, LOW)
    GPIO.output(in3, LOW)
    GPIO.output(in4, LOW)
    time.sleep(1)


def left90():
    """Turn the motor leftF by 90 degrees. uses measured time delays"""
    GPIO.output(in1, LOW)
    GPIO.output(in2, HIGH)
    GPIO.output(in3, LOW)
    GPIO.output(in4, HIGH)
    time.sleep(0.75)


def right90():
    """Turn the motor right by 90 degrees. uses measured time delays"""
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)
    time.sleep(0.8)
