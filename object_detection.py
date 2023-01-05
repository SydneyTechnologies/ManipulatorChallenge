import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
capture = cv2.VideoCapture(0)
count = 0


# Load classes and colors
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)




def start_detection(image):



     # Get frame dimensions
    height, width, channels = image.shape
    # Create a 4D blob from frame
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
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


    # corners, _, _, = detector.detectMarkers(image)
    # aruco_perimeter = cv2.arcLength(corners[0], True)
    # pixel_cm_ratio = aruco_perimeter / 20

    # object_width = w / pixel_cm_ratio
    # object_height = h / pixel_cm_ratio

    # cv2.putText(picture, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    # cv2.putText(picture, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    # Perform non-maximum suppression 
    # Draw detections

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            # where x, y is the location of the center point
            (x, y) = (boxes[i][0], boxes[i][1])
            # w, h is the width and height of the object
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[class_ids[i]]]
            # draw a dot at the center of the shape
            cv2.circle(image, (int(x + (w/2)), int(y + (h/2))), 5, (0, 0, 255), -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# Take a picture whenever the X button is pressed on the keyboard
while True:
    count += 1
    ret, picture = capture.read()
    if ret:
        cv2.imshow("frame", picture)
        # let's find the aruco marker and then detect the corners
        corners, _, _, = detector.detectMarkers(picture)
        # lets draw a bounding box around the aruco marker
        int_corners = np.intp(corners)
        cv2.polylines(picture, int_corners, True, (255, 0, 0), 5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(f"saved/result{count}.jpg", start_detection(picture))
 

    # if count >= 20:
    #     if cv2.waitKey(0):
    #         cv2.imwrite(f"saved/imageAtFrame{count}.jpg", start_detection(picture))


cv2.waitKey(0)
cv2.destroyAllWindows()