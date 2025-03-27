import cv2 as cv
from picamera2 import Picamera2  # type: ignore

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()


while True:
    frame = picam2.capture_array()
    cv.imshow("Live Feed", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        # Save the captured image
        cv.imwrite("captured_image.jpg", frame)
        print("Photo taken and saved as captured_image.jpg")

# Release the camera
cv.destroyAllWindows()
