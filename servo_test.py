from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from time import sleep

factory = PiGPIOFactory()
s = AngularServo("BOARD35", pin_factory=factory, min_pulse_width=0.0006, max_pulse_width=0.0023)

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
