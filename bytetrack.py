import cv2
from ultralytics import YOLO

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("./car_driving.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, imgsz=640, classes=[2], persist=True, conf=0.1, iou=0, verbose=False, tracker="bytetrack.yaml")
    annotated_frame = results[0].plot()
    cv2.imshow("Frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()