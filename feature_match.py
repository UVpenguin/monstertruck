import cv2
import imutils
import utility
from picamera2 import Picamera2  # type: ignore

# --- Load all your templates and descriptors once at startup ---
images, names = utility.readImages()  # returns list of template images and their labels
descriptors = utility.getDescriptors(
    images
)  # computes ORB descriptors for each template

# --- Configure and start the PiCamera2 preview ---
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (640, 480)}
)
picam2.configure(preview_config)
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        name = utility.findMatch(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), descriptors, names)

        # Optional: resize to speed up matching / fit window
        frame = imutils.resize(frame, width=400)

        # Convert to gray for matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the best template match for this frame
        name = utility.findMatch(gray, descriptors, names)
        if name is None:
            name = "No Match"

        # Draw the result on the frame
        cv2.putText(
            frame,
            name,
            (20, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

        # Show the live feed
        cv2.imshow("Detection", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):  # ESC or 'q' to quit
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
