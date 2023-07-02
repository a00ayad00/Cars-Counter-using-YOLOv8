from ultralytics import YOLO
import cv2
import cvzone
from sort import Sort
import numpy as np

model = YOLO('E:\Object Detection\Chapter 5 - Running Yolo\yolov8n.pt')

cap = cv2.VideoCapture("../Videos/cars.mp4")

mask = cv2.imread('E:\Object Detection\python\mask.png')

tracker = Sort(22)

line = (402, 297), (650, 297)

cars = []

while 2023:
    done, frm = cap.read()
    # frm = cv2.flip(frm, 1)
    roi = cv2.bitwise_and(cv2.resize(mask, (frm.shape[1], frm.shape[0])), frm)
    res = model(roi, stream=True)
    res = next(iter(res))
    names = res[0].names
    detects = np.empty((0, 5))
    cv2.line(frm, line[0], line[1], [0, 0, 255], 4)
    for box in res.boxes:
        conf, name = round(float(box.conf[0]) * 100) / 100, res[0].names[int(box.cls[0])]
        if name in ['car', 'truck', 'bus', 'motorcycle'] and conf>0.42:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(frm, (x1, y1), (x2, y2), [0, 255, 0], 2)
            # cvzone.putTextRect(frm, f"{name} {conf}", (max(0, x1), max(35, y1-15)), 1, 2)

            arr = np.array([x1, y1, x2, y2, conf])
            detects = np.vstack([detects, arr])
    trackResults = tracker.update(detects)
    for track_res in trackResults:
        x1, y1, x2, y2, Id = track_res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cv2.rectangle(frm, (x1, y1), (x2, y2), [0, 0, 255], 2)
        cvzone.putTextRect(frm, f"ID {int(Id)}", (max(0, x1), max(35, y1 - 15)), 1, 2)

        cp = x1+w//2, y1+h//2
        cv2.circle(frm, cp, 6, [255, 0, 0], -1)

        if cp[0]>line[0][0] and cp[0]<line[1][0] and cp[1] and\
        cp[1]>line[0][1]-25 and cp[1]<line[0][1]+25:
            if Id not in cars:
                cars.append(Id)
                cv2.line(frm, line[0], line[1], [0, 255, 0], 4)
    cvzone.putTextRect(frm, f"{len(cars)} cars", (12, 30), 2, 2)

    cv2.imshow('cars', frm)
    # cv2.imshow('roi', roi)
    cv2.waitKey(0) #==ord('q'): break