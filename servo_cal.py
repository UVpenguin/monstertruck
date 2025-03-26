from gpiozero import Servo


s = Servo(13)
print(s.min())
print(s.max())
