import RPi.GPIO as GPIO
import time

GPIO.cleanup()

GPIO.setmode(GPIO.BOARD) 
GPIO.setwarnings(False)

HIGH = 1
LOW = 0

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

pwmA = GPIO.PWM(enA, 1000)
pwmB = GPIO.PWM(enB, 1000) 

# move forward
GPIO.output(in1, HIGH)
GPIO.output(in2, LOW)
GPIO.output(in3, HIGH)
GPIO.output(in4, LOW)

pwmA.start(100)
pwmB.start(100)

time.sleep(2) #Delay

# Stop
GPIO.output(in1, LOW)
GPIO.output(in2, LOW)
GPIO.output(in3, LOW)
GPIO.output(in4, LOW)

time.sleep(2) #Delay
pwmA.stop()
pwmB.stop()
