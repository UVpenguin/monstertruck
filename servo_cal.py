from gpiozero import Servo
import RPi.GPIO as GPIO

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

s = Servo(33)
print(s.min())
print(s.max())
