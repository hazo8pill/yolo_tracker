import numpy as np
import cv2

class KalmanBoxTracker:
    """
    Simple Kalman filter with state:
      [x, y, w, h, vx, vy]^T
    Measurement:
      [x, y, w, h]^T
    Constant-velocity for (x, y); width & height modeled without velocity
    """

    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(6, 4)

        # State transition
        # [x, y, w, h, vx, vy] -> dt assumed 1 frame
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 0, 0, 1],  # y' = y + vy
            [0, 0, 1, 0, 0, 0],  # w' = w
            [0, 0, 0, 1, 0, 0],  # h' = h
            [0, 0, 0, 0, 1, 0],  # vx' = vx
            [0, 0, 0, 0, 0, 1],  # vy' = vy
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0],  # h
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def init(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        self.kf.statePost = np.array([x, y, w, h, 0, 0], dtype=np.float32)
        self.kf.statePre = self.kf.statePost.copy()

    def predict(self):
        pred = self.kf.predict()
        x, y, w, h, *_ = pred.flatten()
        return (float(x), float(y), float(w), float(h))

    def correct(self, meas_xywh):
        z = np.array(meas_xywh, dtype=np.float32)
        self.kf.correct(z)