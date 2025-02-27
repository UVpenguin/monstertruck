import RPi.GPIO as GPIO
import movement as motor
import time

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
            time.sleep(1)
        if userInput == "s":
            motor.backward()
            time.sleep(1)
        if userInput == "d":
            motor.right()
            time.sleep(1)
        if userInput == "a":
            motor.left()
            time.sleep(1)
        if userInput == "g":
            motor.right()
            time.sleep(0.75)
        if userInput == "f":
            motor.left()
            time.sleep(0.75)
        motor.stop()
except KeyboardInterrupt:
    motor.stop()
    motor.pwmA.stop()
    motor.pwmB.stop()
