from roboflow import Roboflow
import supervision as sv
import cv2
import time

# Initialize Roboflow
rf = Roboflow(api_key="rRp7y8gRWMWkvv7fSnRN")
project = rf.workspace().project("door-window-detection-pipvh")
model = project.version(1).model

# Dictionary to map class indices to names
index_to_class = {
    '0': "Window",
    '1': "Handle",
    '2': "DoorFrame",
    '3': "Brotha Uhhhh"
}

# Initialize annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoundingBoxAnnotator()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, image = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Perform prediction on the current frame
    frame_result = model.predict(image, confidence=40, overlap=30).json()
    labels = [index_to_class[str(item["class"])] for item in frame_result["predictions"]]
    detections = sv.Detections.from_inference(frame_result)

    # Annotate the image with labels and bounding boxes
    annotated_image = label_annotator.annotate(scene=image, detections=detections, labels=labels)
    annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)

    # Display the annotated image
    cv2.imshow('Annotated Image', annotated_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
