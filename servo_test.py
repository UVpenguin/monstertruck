from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo

factory = PiGPIOFactory()
s = Servo("BOARD35", pin_factory=factory)

s.max()
