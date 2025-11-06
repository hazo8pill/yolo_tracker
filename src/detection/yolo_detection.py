from typing import Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO

def detect_best_box(
    model: YOLO,
    frame_bgr: np.ndarray,
    target_cls: int,
    conf_thres: float,
    infer_size: Optional[Tuple[int, int]] = None
) -> Optional[Tuple[int, int, int, int]]:
    """
    Runs YOLO detection and returns best xyxy bbox for target class.
    """
    H, W = frame_bgr.shape[:2]

    if infer_size:
        iw, ih = infer_size
        resized = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        result_iter = model(resized, stream=True, verbose=False)
        scale_x = W / iw
        scale_y = H / ih
    else:
        result_iter = model(frame_bgr, stream=True, verbose=False)
        scale_x = scale_y = 1.0

    best_conf = -1.0
    best_xyxy = None

    for r in result_iter:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            if cls == target_cls and conf >= conf_thres:
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(float, xyxy)
                x1 *= scale_x; y1 *= scale_y; x2 *= scale_x; y2 *= scale_y
                if conf > best_conf:
                    best_conf = conf
                    best_xyxy = (int(x1), int(y1), int(x2), int(y2))

    return best_xyxy