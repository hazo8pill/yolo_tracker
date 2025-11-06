from typing import List, Optional, Tuple
import numpy as np

from collections import deque
from src.fusion.kalman_filter import KalmanBoxTracker
from src.fusion.utils import try_create_tracker

class TrackerFusion:
    """
    Manages multiple OpenCV trackers and fuses their outputs via reliability-weighted average.
    Falls back to Kalman predictions and exposes consecutive failure counts.
    """
    def __init__(
        self,
        tracker_names: List[str],
    ):
        self.tracker_names = [n.upper() for n in tracker_names]

        self.trackers = []  
        self.success_history = []  
        self.kalman = KalmanBoxTracker()
        self.consecutive_failures = 0

    def _make_tracker(self, name: str):
        return try_create_tracker(name)

    def init(self, frame, bbox_xywh):
        self.trackers.clear()
        self.success_history.clear()
        for name in self.tracker_names:
            try:
                tr = self._make_tracker(name)
            except Exception as e:
                print(f"[WARN] Could not init tracker {name}: {e}")
                continue
            ok = tr.init(frame, tuple(map(int, bbox_xywh)))
            if ok is False:
                print(f"[WARN] Tracker {name} failed to initialize; skipping.")
                continue
            self.trackers.append((name, tr))
            self.success_history.append(deque(maxlen=20))

        if not self.trackers:
            raise RuntimeError("No trackers could be initialized.")

        self.kalman.init(bbox_xywh)
        self.consecutive_failures = 0

    def update(self, frame) -> Tuple[bool, Tuple[float, float, float, float]]:
        if not self.trackers:
            pred = self.kalman.predict()
            self.consecutive_failures += 1
            return False, pred

        boxes = []
        weights = []
        for i, (name, tr) in enumerate(self.trackers):
            ok, box = tr.update(frame)
            self.success_history[i].append(1 if ok else 0)
            if ok:
                boxes.append(box)
                hist = self.success_history[i]
                rel = (sum(hist) / max(1, len(hist)))
                weights.append(rel)

        if not boxes:
            pred = self.kalman.predict()
            self.consecutive_failures += 1
            return False, pred

        weights = np.array(weights, dtype=np.float32)
        weights /= max(1e-6, weights.sum())
        fused = np.average(np.array(boxes, dtype=np.float32), axis=0, weights=weights)

        self.kalman.correct(fused)
        pred = self.kalman.predict()
        self.consecutive_failures = 0
        return True, pred