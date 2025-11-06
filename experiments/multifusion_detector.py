import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Settings ---
VIDEO_SOURCE = './driving.mp4'
YOLO_MODEL = 'yolo12n.pt'
TARGET_CLASS = 2
CONF_THRESHOLD = 0.25
AUTO_REDETECT_INTERVAL = 60

class MultiTrackerFusion:
    def __init__(self):
        self.trackers = []
        self.names = ['CSRT', 'NANO']
        self.weights = [0.3, 0.7]
        self.success_count = np.ones(len(self.names))
        self.kalman = self.__init_kalman()

    def __init_kalman(self):
        """Initialize a constant-velocity Kalman filter for bbox center and size."""
        kf = cv2.KalmanFilter(
            8, 4)

        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (we measure x, y, w, h)
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        # Process and measurement noise
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        return kf

    def __init_nano(self):
        params = cv2.TrackerNano_Params()
        params.backbone = "./nano/nanotrack_backbone_sim.onnx"
        params.neckhead = "./nano/nanotrack_head_sim.onnx"
        return cv2.TrackerNano_create(params)

    def init(self, frame, bbox):
        """Initialize multiple trackers using the same starting box."""
        self.trackers = []
        for name in self.names:
            if name == 'CSRT':
                tracker = cv2.TrackerCSRT_create()
            elif name == 'KCF':
                tracker = cv2.TrackerKCF_create()
            elif name == 'MOSSE':
                tracker = cv2.legacy.TrackerMOSSE_create()
            elif name == 'NANO':
                tracker = self.__init_nano()
            tracker.init(frame, bbox)
            self.trackers.append((name, tracker))

            self.kalman.statePre = np.array(
                [bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], dtype=np.float32
            )
            self.kalman.statePost = self.kalman.statePre.copy()

    def update(self, frame):
        boxes = []
        active_weights = []

        for idx, (name, tracker) in enumerate(self.trackers):
            success, box = tracker.update(frame)
            self.success_count[idx] = 0.9 * \
                self.success_count[idx] + 0.1 * (1 if success else 0)
            if success:
                boxes.append(box)
                active_weights.append(self.success_count[idx])

        if len(boxes) == 0:
            # If no trackers succeeded, predict using Kalman only
            predicted = self.kalman.predict()
            x, y, w, h = predicted[:4]
            return False, (x, y, w, h)

        # Weighted average fusion
        active_weights = np.array(active_weights)
        active_weights /= active_weights.sum()
        fused_box = np.average(np.array(boxes), axis=0, weights=active_weights)

        # Kalman correction step
        measurement = np.array(fused_box, dtype=np.float32)
        self.kalman.correct(measurement)

        # Kalman prediction for next frame (smoothed box)
        predicted = self.kalman.predict()
        x, y, w, h = predicted[:4]

        return True, (float(x), float(y), float(w), float(h))


def main():
    try:
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Install with: pip install ultralytics")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    tracker = None
    frame_count = 0
    fps_list = []

    print("--- Controls ---")
    print("Press 'q' to quit.")
    print("Press 'r' to reset tracker and force YOLO re-detect.")
    print("----------------")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video ended or failed to grab frame.")
            break

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker = None
            print("Manual reset: YOLO re-detecting...")

        # --- Automatic YOLO re-detection ---
        auto_redetect = (frame_count % AUTO_REDETECT_INTERVAL == 0)

        # --- Detection Mode ---
        if tracker is None or auto_redetect:
            if auto_redetect:
                cv2.putText(frame, "Auto Re-Detecting...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                cv2.putText(frame, "Detecting...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            results = model(frame, stream=True, classes=[2], conf=CONF_THRESHOLD, verbose=False)

            best_box = None
            best_conf = 0.0
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box.xyxy[0].cpu().numpy().astype(int)

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
                tracker = MultiTrackerFusion()
                tracker.init(frame, bbox_xywh)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"YOLO init (conf={best_conf:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                print(
                    f"Tracker initialized at {bbox_xywh} (conf={best_conf:.2f})")

        # --- Tracking Mode ---
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                x2, y2 = x + w, y + h
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Tracking (fused)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Tracking FAILED - YOLO Re-detect next", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tracker = None  # Force re-detection

        # --- FPS Calculation ---
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-6)
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO + Multi-Tracker Fusion", frame)
        if frame_count % 30 == 0:
            print(f"Average FPS (last 30 frames): {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal Statistics:")
    print(f"Frames processed: {frame_count}")
    if fps_list:
        print(f"Average FPS: {np.mean(fps_list):.2f}")


if __name__ == "__main__":
    main()
