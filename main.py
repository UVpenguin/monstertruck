import RPi.GPIO as GPIO
import movement as motor

GPIO.cleanup()

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

enA = motor.enA
enB = motor.enB
in1 = motor.in1
in2 = motor.in2
in3 = motor.in3
in4 = motor.in4

motor.setup()

try:
    while True:
        userInput = input()

        if userInput == "w":
            motor.forward()
        if userInput == "s":
            motor.backward()
        if userInput == "d":
            motor.right()
        if userInput == "a":
            motor.left()
        if userInput == "g":
            motor.right()
        if userInput == "f":
            motor.left()
        motor.stop()
except KeyboardInterrupt:
    motor.stop()
    motor.pwmA.stop()
    motor.pwmB.stop()
