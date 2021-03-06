#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import socket
import sys
import pickle
import struct
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def create_classes(input):
    classes = None
    with open(input, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
    
#image = cv2.imread(args.image)
#print(type(image))
#print(image.shape)
#Width = image.shape[1]
#Height = image.shape[0]
Width = 640
Height = 480
scale = 0.00392
classes = create_classes(args.classes)
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
cam.set(3,Width)
cam.set(4,Height)
#print(classes)
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('10.138.19.36', 8089))
net = cv2.dnn.readNet(args.weights, args.config)
# classes = None

# with open(args.classes, 'r') as f:
#     classes = [line.strip() for line in f.readlines()]
 
while True:
    ret, image = cam.read()
    if not ret:
        break
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    print(class_ids)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print(indices)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print(class_ids[i])
        print(confidences[i])
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # cv2.imshow("test", image)
    data = pickle.dumps(image)
    clientsocket.sendall(struct.pack("L", len(data))+data)
    print("Image sent")
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
# cv2.waitKey()
    
# cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
