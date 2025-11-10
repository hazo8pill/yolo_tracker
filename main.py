#!/usr/bin/env python3
"""
YOLO + Multi-Tracker Fusion (Refactored)
- Modular classes
- Safer Kalman model (position + velocity)
- Robust tracker creation with fallbacks
- Adaptive reliability weighting (moving average)
- On-demand + periodic re-detection
- Optional GPU, input resizing for speed
- CLI configuration
"""

import argparse
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.fusion.tracker_fusion import TrackerFusion
from src.fusion.utils import xyxy_to_xywh, clip_bbox_xywh
from src.detection.yolo_detection import detect_best_box


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO + Multi-Tracker Fusion (Refactored)")
    p.add_argument("--video", type=str, default="./driving.mp4",
                   help="Path to video source or camera index (int).")
    p.add_argument("--model", type=str, default="yolo11s.pt",
                   help="Ultralytics model path or name.")
    p.add_argument("--target", type=int, default=2,
                   help="Target class id (e.g., 2 for 'car' in COCO).")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO confidence threshold.")
    p.add_argument("--detect-every", type=int, default=60,
                   help="Periodic re-detection interval in frames.")
    p.add_argument("--fail-redetect", type=int, default=10,
                   help="Consecutive fusion failures to trigger immediate re-detect.")
    p.add_argument("--gpu", action="store_true", help="Use CUDA if available.")
    p.add_argument("--infer-width", type=int, default=0,
                   help="Optional inference resize width (0=disabled).")
    p.add_argument("--infer-height", type=int, default=0,
                   help="Optional inference resize height (0=disabled).")
    p.add_argument("--trackers", type=str, default="CSRT,NANO",
                   help="Comma list from [CSRT,KCF,MOSSE,NANO].")
    p.add_argument("--nano-backbone", type=str,
                   default="./nano/nanotrack_backbone_sim.onnx")
    p.add_argument("--nano-head", type=str,
                   default="./nano/nanotrack_head_sim.onnx")
    return p.parse_args()

def preprocess_frame(frame):
    """Applies necessary single-frame enhancements."""

    # --- 1. Low-Light Enhancement (CLAHE) ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    frame_clahe = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    # --- 2. Noise Reduction (Optional) ---
    # frame_denoised = cv2.GaussianBlur(frame_clahe, (3, 3), 0)

    return frame_clahe

def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, alpha=1.5, threshold=0):
    """
    Applies the Unsharp Masking technique for image sharpening.

    Args:
        img (np.ndarray): The source image.
        kernel_size (tuple): The size of the Gaussian kernel (e.g., (5, 5)).
        sigma (float): Gaussian standard deviation.
        alpha (float): Strength of the sharpening effect (typically 1.0 to 3.0).
        threshold (int): Minimum difference threshold (optional, often set to 0).
    """
    # 1. Create the blurred version (the 'unsharp' mask basis)
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)

    # 2. Calculate the "mask" or "detail map" (Original - Blurred)
    # This result contains the high-frequency content (edges)
    mask = cv2.subtract(img, blurred)

    # Optional: Threshold the mask to only sharpen significant edges (reduce noise amplification)
    if threshold > 0:
        mask[np.abs(mask) < threshold] = 0

    # 3. Add the scaled mask back to the original image
    # cv2.addWeighted is used for efficient scaling and blending:
    # Output = img * (1.0 + alpha) + blurred * (-alpha) + gamma (gamma=0)
    sharpened = cv2.addWeighted(img, 1.0 + alpha, blurred, -alpha, 0)

    # Note: A simpler approach for the final step is:
    # sharpened = cv2.add(img, cv2.convertScaleAbs(mask, alpha=alpha))

    return sharpened

def main():
    args = parse_args()

    try:
        model = YOLO(args.model)
        if args.gpu:
            try:
                model.to("cuda")
                print("[INFO] Using CUDA")
            except Exception:
                print("[WARN] CUDA move failed; using CPU.")
    except Exception as e:
        print(f"[ERROR] Loading YOLO model '{args.model}': {e}")
        print("Hint: pip install ultralytics")
        return

    cap = None
    try:
        src = int(args.video)
        cap = cv2.VideoCapture(src)
    except ValueError:
        cap = cv2.VideoCapture(args.video)

    if not cap or not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.video}")
        return

    tracker_fusion: Optional[TrackerFusion] = None
    frame_count = 0
    fps_hist = deque(maxlen=30)
    infer_size = None
    if args.infer_width > 0 and args.infer_height > 0:
        infer_size = (args.infer_width, args.infer_height)

    print("--- Controls ---")
    print("Press 'q' to quit.")
    print("Press 'r' to reset tracker and force YOLO re-detect.")
    print("----------------")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video ended or failed to grab frame.")
                break

            frame = cv2.resize(frame, (640, 640))
            frame = cv2.medianBlur(frame, 5)
            H, W = frame.shape[:2]
            frame_count += 1

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                tracker_fusion = None
                cv2.putText(frame, "Manual Re-Detect...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Auto triggers
            periodic_redetect = (frame_count % args.detect_every == 0)
            need_redetect = (tracker_fusion is None) or periodic_redetect
            immediate_fail_redetect = False

            if tracker_fusion and tracker_fusion.consecutive_failures >= args.fail_redetect:
                need_redetect = True
                immediate_fail_redetect = True

            # Detection mode
            if need_redetect:
                msg = "Auto Re-Detecting..." if periodic_redetect or immediate_fail_redetect else "Detecting..."
                color = (0, 200, 255) if periodic_redetect or immediate_fail_redetect else (
                    0, 0, 255)
                cv2.putText(frame, msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                xyxy = detect_best_box(
                    model=model,
                    frame_bgr=frame,
                    target_cls=args.target,
                    conf_thres=args.conf,
                    infer_size=infer_size
                )

                if xyxy is not None:
                    x1, y1, x2, y2 = xyxy
                    bbox_xywh = xyxy_to_xywh(x1, y1, x2, y2)
                    bbox_xywh = clip_bbox_xywh(bbox_xywh, W, H)

                    # Build fusion tracker
                    names = [s.strip()
                             for s in args.trackers.split(",") if s.strip()]
                    tracker_fusion = TrackerFusion(
                        tracker_names=names
                    )
                    try:
                        tracker_fusion.init(frame, bbox_xywh)
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "YOLO init", (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print(
                            f"[INFO] Fusion tracker initialized @ {bbox_xywh}")
                    except Exception as e:
                        print(f"[WARN] Fusion init failed: {e}")
                        tracker_fusion = None
                else:
                    # No detection found; leave tracker as-is (None) and continue
                    pass

            if tracker_fusion:
                ok, bbox_f = tracker_fusion.update(frame)
                x, y, w, h = bbox_f
                x, y, w, h = clip_bbox_xywh((x, y, w, h), W, H)
                if ok:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 255), 2)
                    cv2.putText(frame, "Tracking (fused)", (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Tracking FAILED - will re-detect", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - t0))
            fps_hist.append(fps)
            avg_fps = sum(fps_hist) / len(fps_hist)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLO + Multi-Tracker Fusion (Refactored)", frame)
            if frame_count % 30 == 0:
                print(f"[INFO] Avg FPS (last 30): {avg_fps:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nFinal Statistics:")
        print(f"Frames processed: {frame_count}")
        if len(fps_hist) > 0:
            print(f"Recent Average FPS: {sum(fps_hist)/len(fps_hist):.2f}")


if __name__ == "__main__":
    main()
