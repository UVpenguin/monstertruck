import RPi.GPIO as GPIO
import time
from movement import *

GPIO.cleanup()

GPIO.setmode(GPIO.BOARD) 
GPIO.setwarnings(False)

enA = 12
enB = 32
in1 = 7
in2 = 11
in3 = 13
in4 = 15

GPIO.setup(enA, GPIO.OUT) 
GPIO.setup(enB, GPIO.OUT) 
GPIO.setup(in1, GPIO.OUT) 
GPIO.setup(in2, GPIO.OUT) 
GPIO.setup(in3, GPIO.OUT) 
GPIO.setup(in4, GPIO.OUT) 

pwmA = GPIO.PWM(enA, 255)
pwmB = GPIO.PWM(enB, 255)

pwmA.start(100)
pwmB.start(100)

forward()
time.sleep(2) 

right()
time.sleep(2) 

left()
time.sleep(2)

stop()
pwmA.stop()
pwmB.stop()