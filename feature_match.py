import cv2
from picamera2 import Picamera2  # type: ignore
import utility

# load templates once
templ_kps, templ_des, names = utility.loadTemplates()

# configure PiCamera2
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(cfg)
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        label = utility.findMatch(gray, templ_kps, templ_des, names)
        if not label:
            label = "No Match"

        cv2.putText(
            frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) in (27, ord("q")):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
