from gpiozero import Servo
import RPi.GPIO as GPIO

GPIO.cleanup()

s = Servo(13)
print(s.min())
print(s.max())
