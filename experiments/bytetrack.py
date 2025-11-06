import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("./driving.mp4")

# FPS calculation variables
fps_list = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Measure time for tracking
    start_time = time.time()
    results = model.track(frame, imgsz=640, classes=[2], persist=True, conf=0.1, iou=0.3, verbose=False, tracker="bytetrack.yaml")
    end_time = time.time()
    
    # Calculate FPS
    processing_time = end_time - start_time
    fps = 1.0 / processing_time if processing_time > 0 else 0
    fps_list.append(fps)
    frame_count += 1
    
    # Calculate average FPS (over last 30 frames for smoother display)
    avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:]) if fps_list else fps
    
    annotated_frame = results[0].plot()
    
    # Display FPS on frame
    cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Frame", annotated_frame)
    
    # Print FPS to console every 30 frames
    if frame_count % 30 == 0:
        print(f"Average FPS (last 30 frames): {avg_fps:.2f}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print final statistics
if fps_list:
    print(f"\nFinal Statistics:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {sum(fps_list) / len(fps_list):.2f}")
    print(f"Min FPS: {min(fps_list):.2f}")
    print(f"Max FPS: {max(fps_list):.2f}")

cap.release()
cv2.destroyAllWindows()