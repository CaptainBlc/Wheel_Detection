import os
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

YOLO_WEIGHTS_URL = 'https://pjreddie.com/media/files/yolov3.weights'
YOLO_CFG_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
COCO_NAMES_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

YOLO_WEIGHTS = 'yolov3.weights'
YOLO_CFG = 'yolov3.cfg'
COCO_NAMES = 'coco.names'

def download_file(url, local_filename):
    if not os.path.exists(local_filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS)
download_file(YOLO_CFG_URL, YOLO_CFG)
download_file(COCO_NAMES_URL, COCO_NAMES)

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(COCO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_wheels(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and (class_id == classes.index('car') or class_id == classes.index('truck')):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def show_image(image):
   
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def detect_wheels_in_image_file(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at {image_path} could not be loaded.")
        return
    result_image = detect_wheels(image)
    show_image(result_image)

def detect_wheels_in_camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = detect_wheels(frame)
        show_image(result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1. Canlı kamera beslemesi")
    print("2. Fotoğraf yükle ve işleme")
    choice = input("Bir seçenek girin (1 veya 2): ")

    if choice == '1':
        detect_wheels_in_camera()
    elif choice == '2':
        Tk().withdraw()  
        image_path = filedialog.askopenfilename()
        detect_wheels_in_image_file(image_path)
    else:
        print("Geçersiz seçenek. Lütfen 1 veya 2 girin.")
