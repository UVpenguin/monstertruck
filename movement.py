import RPi.GPIO as GPIO

HIGH = 1
LOW = 0

def forward(): 
    GPIO.output(in1, LOW)
    GPIO.output(in2, HIGH)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)

def right():
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, HIGH)
    GPIO.output(in4, LOW)

def left():
    GPIO.output(in1, HIGH)
    GPIO.output(in2, LOW)
    GPIO.output(in3, LOW)
    GPIO.output(in4, HIGH)

def stop():
    GPIO.output(in1, LOW)
    GPIO.output(in2, LOW)
    GPIO.output(in3, LOW)
    GPIO.output(in4, LOW)