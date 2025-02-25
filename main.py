import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM) 
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

while True:
    GPIO.output(in1, 1)
    GPIO.output(in2, 1)

    time.sleep(2) #Delay

    GPIO.output(in1, 0)
    GPIO.output(in2, 0)

    time.sleep(2) #Delay
    
    GPIO.cleanup