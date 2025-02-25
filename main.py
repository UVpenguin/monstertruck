import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD) 
GPIO.setwarnings(False)

enA = 12
enB = 32
in1 = 7
in2 = 11
in3 = 13
in4 = 15

GPIO.cleanup

GPIO.setup(enA, GPIO.OUT) 
GPIO.setup(enB, GPIO.OUT) 
GPIO.setup(in1, GPIO.OUT) 
GPIO.setup(in2, GPIO.OUT) 
GPIO.setup(in3, GPIO.OUT) 
GPIO.setup(in4, GPIO.OUT) 

pwmA = GPIO.PWM(enA, 255)
pwmB = GPIO.PWM(enB, 255) 

while True:
    GPIO.output(in1, 1)
    GPIO.output(in2, 1)

    pwmA.start(100)
    pwmB.start(100)

    time.sleep(2) #Delay

    GPIO.output(in1, 0)
    GPIO.output(in2, 0)

    time.sleep(2) #Delay