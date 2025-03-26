from gpiozero import Servo

s = Servo(33)
print(s.min())
print(s.max())
