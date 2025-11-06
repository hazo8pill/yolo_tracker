import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolo12n.pt") 

cap = cv2.VideoCapture("./car_driving.mp4")

class MultiTracker:
    def __init__(self):
        # We will create new instances when we init
        self.tracker_csrt = None
        self.tracker_mosse = None
        self.initialized = False

    def init(self, frame, box):
        """(Re)-initializes both trackers."""
        print("Initializing trackers...")
        self.tracker_csrt = cv2.legacy.TrackerCSRT_create()
        self.tracker_mosse = cv2.legacy.TrackerMOSSE_create()
        
        self.tracker_csrt.init(frame, box)
        self.tracker_mosse.init(frame, box)
        self.initialized = True

    def update(self, frame):
        """
        Returns (success, box, label)
        Tries CSRT first, then falls back to MOSSE.
        """
        if not self.initialized:
            return False, None, None

        # 1. Try the high-accuracy tracker first
        success_csrt, box_csrt = self.tracker_csrt.update(frame)
        if success_csrt:
            # Main tracker succeeded
            return True, box_csrt, "CSRT (Precise)"

        # 2. If CSRT failed, try the fallback tracker
        print("CSRT lost object, trying MOSSE fallback...")
        success_mosse, box_mosse = self.tracker_mosse.update(frame)
        if success_mosse:
            # Fallback tracker succeeded
            return True, box_mosse, "MOSSE (Fallback)"

        # 3. Both trackers failed
        print("All trackers lost object.")
        self.initialized = False # Need re-detection
        return False, None, None

# --- Main Loop ---

tracker = MultiTracker() # Use the new class
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- DETECTION (Every 20 frames) ---
    if frame_count % 20 == 0:
        results = model.predict(frame, imgsz=640, classes=[2], conf=0.1, verbose=False)
        boxes = results[0].boxes
        largest_box_data = None
        max_area = 0

        for box in boxes:
            _, _, w_n, h_n = box.xywhn[0].tolist()  
            area = w_n * h_n
            if area > max_area:
                max_area = area
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                largest_box_data = (x1, y1, x2 - x1, y2 - y1) # (x, y, w, h)

        if largest_box_data is not None:
            # (Re)-initialize both trackers
            tracker.init(frame, largest_box_data)
            x, y, w, h = largest_box_data
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(frame, "YOLO DETECT", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            tracker.initialized = False # No detection, force reset
    
    # --- TRACKING (Frames in between) ---
    else:
        if tracker.initialized:
            success, box, label = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                
                # Draw with different colors
                color = (0, 255, 0) if label == "CSRT (Precise)" else (0, 165, 255) # Green for CSRT, Orange for MOSSE
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Both trackers have failed, wait for next YOLO detection
                pass

    frame_count += 1
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()