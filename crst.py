import cv2
import time
from ultralytics import YOLO

# --- Settings ---
# You can change this to a video file path
VIDEO_SOURCE = './car_driving.mp4'
# YOLO model (n=nano is fast, m=medium is more accurate)
YOLO_MODEL = 'yolo12n.pt' 
# Target class to track (0 is 'person' in the default COCO dataset)
TARGET_CLASS = 2
# Confidence threshold for YOLO detection
CONF_THRESHOLD = 0.2 

def main():
    # Load YOLO model
    try:
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure 'ultralytics' is installed: pip install ultralytics")
        return

    # Open video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    tracker = None
    
    # FPS calculation variables
    fps_list = []
    frame_count = 0
    
    print("--- Controls ---")
    print("Press 'q' to quit.")
    print("Press 'r' to reset tracker and re-detect.")
    print("----------------")

    while True:
        # Start timing
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame or video ended.")
            break

        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker = None  # Reset tracker
            print("Tracker reset. Re-detecting...")

        # --- STATE 1: DETECTION (No active tracker) ---
        if tracker is None:
            cv2.putText(frame, "Detecting...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Run YOLO detection
            # verbose=False silences the Ultralytics log output for each frame
            results = model(frame, stream=True, verbose=False)

            # Find the first valid target
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Check class and confidence
                    if box.cls[0] == TARGET_CLASS and box.conf[0] > CONF_THRESHOLD:
                        # Get bounding box in (x1, y1, x2, y2) format
                        bbox_xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = bbox_xyxy
                        
                        # --- CRITICAL STEP ---
                        # Convert (x1, y1, x2, y2) to (x, y, w, h) for OpenCV tracker
                        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)

                        # Initialize the CSRT tracker
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox_xywh)

                        print(f"Tracker initialized on class {TARGET_CLASS} at {bbox_xywh}")

                        # Draw the initial detection box (Green)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Target Acquired", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Stop searching once a target is found
                        break
                if tracker is not None:
                    break

        # --- STATE 2: TRACKING (Tracker is active) ---
        else:
            # Update the tracker
            success, bbox = tracker.update(frame)

            if success:
                # bbox is (x, y, w, h)
                x, y, w, h = [int(v) for v in bbox]
                x2 = x + w
                y2 = y + h

                # Draw the tracking box (Blue)
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Tracking failed
                cv2.putText(frame, "Tracking FAILED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # Reset tracker to re-detect in the next frame
                tracker = None

        # Calculate FPS
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        fps_list.append(fps)
        frame_count += 1
        
        # Calculate average FPS (over last 30 frames for smoother display)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:]) if fps_list else fps
        
        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display the resulting frame
        # cv2.imshow("YOLO + CSRT Tracker", frame)
        
        # Print FPS to console every 30 frames
        if frame_count % 30 == 0:
            print(f"Average FPS (last 30 frames): {avg_fps:.2f}")

    # Print final statistics
    if fps_list:
        print(f"\nFinal Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {sum(fps_list) / len(fps_list):.2f}")
        print(f"Min FPS: {min(fps_list):.2f}")
        print(f"Max FPS: {max(fps_list):.2f}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()