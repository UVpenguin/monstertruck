from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from time import sleep

factory = PiGPIOFactory()
s = AngularServo("BOARD35", pin_factory=factory, min_angle=-180, max_angle=180)

while True:
    s.angle = -180
    sleep(2)
    s.angle = -90
    sleep(2)
    s.angle = -45
    sleep(2)
    s.angle = 0
    sleep(2)
    s.angle = 90
    sleep(2)
    s.angle = 180
    sleep(2)
