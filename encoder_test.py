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

encoder = 16

# sets motor pins as write
GPIO.setup(enA, GPIO.OUT)
GPIO.setup(enB, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

GPIO.setup(encoder, GPIO.IN)  # read encoder pin

# encoder variables
lastState = GPIO.input(
    encoder
)  # sets the intial state of the variable to the inital state of encoder
rotationCount = 0
stateCount = 0
stateCountTotal = 0

circ = 20.43  # cm
statesPerRotation = 40
distancePerStep = circ / statesPerRotation

# pwm pins set to 1000Hz
pwmA = GPIO.PWM(enA, 1000)
pwmB = GPIO.PWM(enB, 1000)
# duty cycle set to 100%
pwmA.start(100)
pwmB.start(100)


def encoder_callback(channel):
    global stateCount, stateCountTotal, rotationCount, statesPerRotation, currentState, lastState  # noqa: E501

    currentState = GPIO.input(encoder)

    if currentState != lastState:
        lastState = currentState
        stateCount += 1
        stateCountTotal += 1

    if stateCount == statesPerRotation:
        rotationCount += 1
        stateCount = 0


GPIO.add_event_detect(
    encoder, GPIO.RISING, callback=encoder_callback, bouncetime=10
)  # noqa: E501

try:
    while True:
        # currentState = GPIO.input(encoder)
        # motor.forward()

        # if currentState != lastState:
        #     lastState = currentState
        #     stateCount += 1
        #     stateCountTotal += 1
        # if stateCount == statesPerRotation:
        #     rotationCount += 1
        #     stateCount = 0
        distance = distancePerStep * stateCountTotal
        print("Distance", distance)

except KeyboardInterrupt:  # ctrl+C
    motor.stop()
    pwmA.stop()
    pwmB.stop()
