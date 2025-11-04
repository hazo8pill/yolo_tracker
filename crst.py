import cv2
from ultralytics import YOLO
import time

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("./car_driving.mp4")

tracker = None 

frame_count = 0

found = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 20 == 0 and not found:
        results = model.predict(frame, imgsz=640, classes=[2], conf=0.1, iou=0.1, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            found = True
            box = boxes[0].xyxy[0]
            x, y = int(box[0]), int(box[1])
            w = int(box[2] - box[0])
            h = int(box[3] - box[1])
            bbox_xywh = (x, y, w, h)

            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, bbox_xywh)
            x, y, w, h = bbox_xywh
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(frame, "DETECTION", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            print("No car detected, resetting tracker.")
            found = False

    frame_count += 1

    if found:
        success, box = tracker.update(frame)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)
        cv2.putText(frame, "TRACKING", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(success, box)
        found = success

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()