import RPi.GPIO as GPIO

HIGH = 1
LOW = 0

enA = 12
enB = 32
in1 = 7
in2 = 11
in3 = 13
in4 = 15

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