import RPi.GPIO as GPIO
import time

# Use physical pin numbering
GPIO.setmode(GPIO.BOARD)

# Define the servo control pin (physical pin 35)
servo_pin = 35
GPIO.setup(servo_pin, GPIO.OUT)

# Set PWM frequency to 50Hz (period = 20ms)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)


def set_servo_angle(angle):
    """
    Sets the servo to the specified angle between -90 and 90 degrees.
    The servo mapping is based on:
      -90 degrees -> effective angle 0 (DutyCycle ~2%)
       0 degrees -> effective angle 90 (DutyCycle ~7.5%)
      90 degrees -> effective angle 180 (DutyCycle ~12%)
    """
    # Convert from -90 to 90 into 0 to 180 range
    effective_angle = angle + 90
    # Calculate the duty cycle (using the formula: DutyCycle = 1/18 * angle + 2)
    duty = (1.0 / 18.0) * effective_angle + 2
    pwm.ChangeDutyCycle(duty)
    # Short delay to allow servo to move
    time.sleep(0.02)
    # Optional: setting duty cycle to 0 can reduce jitter after moving
    pwm.ChangeDutyCycle(0)


try:
    while True:
        # Sweep from -90 to 90 degrees
        for angle in range(-90, 91, 1):  # increment by 1 degree
            set_servo_angle(angle)
            time.sleep(0.01)
        time.sleep(0.5)  # pause at the end of the sweep

        # Sweep back from 90 to -90 degrees
        for angle in range(90, -91, -1):  # decrement by 1 degree
            set_servo_angle(angle)
            time.sleep(0.01)
        time.sleep(0.5)
except KeyboardInterrupt:
    # Exit the loop when Ctrl+C is pressed
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
