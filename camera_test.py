from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

frame = picam2.capture_array()

cv2.imshow("Camera Feed", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
