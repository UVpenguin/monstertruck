from gpiozero import Servo


# GPIO SETUP
s = Servo("BOARD35")


try:
    s.mid()

finally:
    pass
