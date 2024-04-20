from ultralytics import YOLO
import cv2 as cv2

model= YOLO("doors.pt")
with open('classes.txt') as f:
    classes = f.read().splitlines()


# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        # Perform object detection
        results = model(frame)

        # Display results
        results.show()

    if cv2.waitKey(1) == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()