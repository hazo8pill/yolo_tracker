import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolo12n.pt") 

cap = cv2.VideoCapture("./car_driving.mp4")

class Tracker:
    def __init__(self, tracker_type="CSRT"):
        if tracker_type == "CSRT":
            self.tracker = cv2.legacy.TrackerCSRT_create()
        elif tracker_type == "MOSSE":
            self.tracker = cv2.legacy.TrackerMOSSE_create()
        else:
            raise ValueError(f"Invalid tracker type: {tracker_type}")
        self.initialized = False

    def init(self, frame, box):
        """(Re)-initializes the tracker."""
        self.tracker.init(frame, box)
        self.initialized = True

    def update(self, frame):
        """Returns (success, box)"""
        return self.tracker.update(frame)

tracker = Tracker(tracker_type="CSRT")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 20 == 0:
        results = model.predict(frame, imgsz=640, classes=[2], conf=0.1, iou=0, verbose=False)
        boxes = results[0].boxes

        largest_box_data = None
        max_area = 0

        for box in boxes:
            _, _, w_n, h_n = box.xywhn[0].tolist()  
            area = w_n * h_n
            if area > max_area:
                max_area = area
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                largest_box_data = (x1, y1, x2 - x1, y2 - y1) # (x, y, w, h) format

        if largest_box_data is not None:
            tracker.init(frame, largest_box_data)
            x, y, w, h = largest_box_data
            
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(frame, "DETECTION", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # No car detected, so stop tracking
            print(f"Frame {frame_count}: No car detected, resetting tracker.")
            tracker.initialized = False
    
    else:
        if tracker.initialized:
            success, box = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "TRACKING", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print(f"Frame {frame_count}: Tracker lost object.")
                tracker.initialized = False

    frame_count += 1
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()