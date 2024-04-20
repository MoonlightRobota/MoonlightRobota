from ultralytics import YOLO
import cv2

model = YOLO("doors.pt")  # Ensure the model file path is correct

# Load class names
with open('classes.txt') as f:
    classes = f.read().splitlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # Perform object detection
        results = model(frame)

        # Clear the frame for displaying confidence scores

        # Manually process and display results
        for i, result in enumerate(results):
            if result and len(result) >= 6:  # Check if result is non-empty and has at least 6 elements
                conf = result[4]  # Extract the confidence score
                cls = int(result[5])  # Extract the class index

                # Display the confidence score and class
                text = f"Class: {classes[cls]}, Confidence: {conf:.2f}"
                cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame with confidence scores
        cv2.imshow('Confidence Scores', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
