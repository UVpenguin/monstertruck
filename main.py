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

GPIO.setup(enA, GPIO.OUT)
GPIO.setup(enB, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

pwmA = GPIO.PWM(enA, 1000)
pwmB = GPIO.PWM(enB, 1000)

pwmA.start(100)
pwmB.start(100)


try:
    while True:
        userInput = input()

        if userInput == "w":
            motor.forward()
            motor.precise_sleep(1)  # Replaced time.sleep
        elif userInput == "s":
            motor.backward()
            motor.precise_sleep(1)
        elif userInput == "d":
            motor.right()
            motor.precise_sleep(1)
        elif userInput == "a":
            motor.left()
            motor.precise_sleep(1)
        elif userInput == "g":
            motor.right()
            motor.precise_sleep(0.75)  # Adjusted for 90-degree turn
        elif userInput == "f":
            motor.left()
            motor.precise_sleep(0.75)
        motor.stop()
except KeyboardInterrupt:
    motor.stop()
    pwmA.stop()
    pwmB.stop()
