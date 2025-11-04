import cv2
from ultralytics import YOLO
import sys
import numpy as np

# --- Configuration ---
VIDEO_PATH = "./car_driving.mp4"  
YOLO_MODEL = "yolo11n.pt"  
CONFIDENCE_THRESHOLD = 0.2

class SingleObjectTracker:
    """
    Manages CSRT and MOSSE for tracking a single object.
    Uses CSRT as primary and MOSSE as a fallback to re-initialize CSRT.
    """
    def __init__(self, initial_frame, initial_bbox):
        self.csrt = None
        self.mosse = None
        self.current_bbox = initial_bbox
        
        print(f"Initializing trackers with bbox: {initial_bbox}")
        self._init_trackers(initial_frame, initial_bbox)

    def _init_trackers(self, frame, bbox):
        """Helper to initialize or re-initialize both trackers."""
        try:
            self.csrt = cv2.TrackerCSRT_create()
            self.csrt.init(frame, bbox)

            self.mosse = cv2.legacy.TrackerMOSSE_create()
            self.mosse.init(frame, bbox)
            
            self.current_bbox = bbox
        except Exception as e:
            print(f"[Error] Failed to initialize trackers: {e}")
            self.csrt = None
            self.mosse = None

    def update(self, frame):
        """
        Update the tracker using CSRT-first, MOSSE-fallback logic.
        Returns (success, bounding_box, status_message)
        """
        if not self.csrt or not self.mosse:
            return False, None, "Uninitialized"

        # 1. Try Primary Tracker (CSRT)
        success_csrt, bbox_csrt = self.csrt.update(frame)
        
        if success_csrt:
            self.current_bbox = bbox_csrt
            # Also update MOSSE to keep it synced
            self.mosse.init(frame, bbox_csrt)
            return True, bbox_csrt, "Tracking (CSRT)"
            
        # 2. Try Fallback Tracker (MOSSE)
        # print("CSRT failed, trying MOSSE...")
        success_mosse, bbox_mosse = self.mosse.update(frame)
        
        if success_mosse:
            self.current_bbox = bbox_mosse
            # CRITICAL: Re-initialize CSRT with MOSSE's successful position
            # This helps CSRT recover from temporary occlusions or failures
            print("CSRT lost. Re-initializing CSRT with MOSSE position.")
            self._init_trackers(frame, bbox_mosse) 
            return True, bbox_mosse, "Fallback (MOSSE)"
            
        # 3. Both trackers failed
        print("All trackers lost!")
        return False, None, "LOST"

# --- Main Application Logic ---

def main():
    # 1. Load YOLO Model
    print(f"Loading YOLO model: {YOLO_MODEL}...")
    try:
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure 'ultralytics' is installed and the model file is correct.")
        return

    # 2. Open Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # 3. Initialization
    tracker = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # 4. Tracker Update / Re-Detection
        if tracker:
            # --- We are in "TRACKING" mode ---
            success, bbox, status = tracker.update(frame)
            
            if success:
                # Draw bounding box
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw status text
                cv2.putText(frame, status, (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                # Tracker is lost
                print("Tracker lost. Re-detecting...")
                tracker = None # Switch to "DETECTION" mode
                
        else:
            # --- We are in "DETECTION" mode ---
            cv2.putText(frame, "Detecting (YOLO)...", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Run YOLO
            results = model(frame, verbose=False)
            
            # Find the best detection
            best_box = None
            best_conf = 0.0

            for box in results[0].boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD and box.conf[0] > best_conf:
                    best_conf = box.conf[0]
                    best_box = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            
            if best_box is not None:
                # Convert [x1, y1, x2, y2] to [x, y, w, h]
                x, y = int(best_box[0]), int(best_box[1])
                w = int(best_box[2] - best_box[0])
                h = int(best_box[3] - best_box[1])
                bbox_xywh = (x, y, w, h)
                
                # Initialize the tracker
                print(f"Object detected. Initializing tracker at {bbox_xywh}")
                tracker = SingleObjectTracker(frame, bbox_xywh)
                
        # 5. Display the frame
        cv2.imshow("Multi-Tracker Fusion (CSRT + MOSSE) with YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if VIDEO_PATH == "your_video.mp4":
        print("ERROR: Please change the 'VIDEO_PATH' variable in the script.")
    else:
        main()