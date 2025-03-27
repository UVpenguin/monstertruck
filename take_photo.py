import cv2 as cv

# Open the webcam (0 is the default camera)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


while True:
    ret, frame = cap.read()
    cv.imshow("Live Feed", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        # Save the captured image
        cv.imwrite("captured_image.jpg", frame)
        print("Photo taken and saved as captured_image.jpg")

# Release the camera
cap.release()
cv.destroyAllWindows()
