
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

output_file = open("object_center_coordinates.txt", "a")

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            output_file.write(f"Object: {classNames[int(box.cls[0])]}, Center: ({center_x}, {center_y})\n")

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[int(box.cls[0])], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

output_file.close()

cap.release()
cv2.destroyAllWindows()





















'''
from ultralytics import YOLO
import cv2
import torch
from midas.midas_net import MidasNet


# Load the depth estimation model
depth_model = MidasNet()
depth_model.eval()

# Load the YOLO object detection model
yolo_model = YOLO("yolo-Weights/yolov8n.pt")

# Define class names for YOLO
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Open output file for writing object center coordinates
output_file = open("object_center_coordinates.txt", "a")

# Initialize video capture
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Main loop for object detection and depth estimation
while True:
    success, img = cap.read()
    
    # Perform object detection with YOLO
    results = yolo_model(img, stream=True)
    
    # Perform depth estimation with MiDaS
    with torch.no_grad():
        input_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        depth_prediction = depth_model.forward(input_tensor)
    depth_map = depth_prediction.squeeze().cpu().numpy()

    # Process object detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Extract depth at the center of the object
            depth = depth_map[center_y, center_x]  # Assuming depth_map is in the same resolution as the image
            
            # Write object information to output file
            object_class = class_names[int(box.cls[0])]
            output_file.write(f"Object: {object_class}, Center: ({center_x}, {center_y}), Depth: {depth}\n")

            # Draw bounding box and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, object_class, org, font, fontScale, color, thickness)

    # Display the image with detected objects
    cv2.imshow('Webcam', img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Close the output file and release resources
output_file.close()
cap.release()
cv2.destroyAllWindows()
'''


from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

output_file = open("object_center_coordinates.txt", "a")

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            output_file.write(f"Object: {classNames[int(box.cls[0])]}, Center: ({center_x}, {center_y})\n")

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[int(box.cls[0])], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

output_file.close()

cap.release()
cv2.destroyAllWindows()
