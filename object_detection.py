import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
capture = cv2.VideoCapture(0)
count = 0

while True:
    count += 1
    ret, picture = capture.read()
    if count >= 50:
        capture.release()
        break


# Load classes and colors
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



 # Get frame dimensions
height, width, channels = picture.shape

# Create a 4D blob from frame
blob = cv2.dnn.blobFromImage(picture, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

# Run forward pass of YOLO
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# Parse detections
boxes = []
confidences = []
class_ids = []
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Perform non-maximum suppression 
# Draw detections

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(picture, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
        cv2.putText(picture, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)



# Show frame
cv2.imshow("Object Detection", picture)
cv2.waitKey(0)
cv2.destroyAllWindows()