from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo
from gpiozero.tools import sin_values
from signal import pause

factory = PiGPIOFactory()
s = Servo("BOARD35", pin_factory=factory)

s.source = sin_values()
s.source_delay = 0.1

pause()
