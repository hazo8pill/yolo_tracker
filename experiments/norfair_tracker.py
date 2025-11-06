import argparse
import os
from typing import List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO
import norfair
from norfair import Detection, Tracker, Video

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000


class YOLOn:
    def __init__(self, model_path: str, device: Optional[str] = None):
        try:
            self.model = YOLO(model_path)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ):
        """Run YOLO inference and return detections."""
        results = self.model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            classes=classes,
            device=self.device,
            verbose=False,
        )
        return results[0]  # YOLO returns list of Results objects


def yolo_detections_to_norfair_detections(
    yolo_result, track_points: str = "centroid"
) -> List[Detection]:
    """Convert YOLO results into Norfair detections."""
    norfair_detections: List[Detection] = []

    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return norfair_detections

    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    scores = yolo_result.boxes.conf.cpu().numpy()
    classes = yolo_result.boxes.cls.cpu().numpy().astype(int)

    if track_points == "centroid":
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            centroid = np.array([[ (x1 + x2) / 2, (y1 + y2) / 2 ]])
            norfair_detections.append(
                Detection(points=centroid, scores=np.array([score]), label=cls)
            )
    elif track_points == "bbox":
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            bbox = np.array([[x1, y1], [x2, y2]])
            norfair_detections.append(
                Detection(points=bbox, scores=np.array([score, score]), label=cls)
            )

    return norfair_detections


def main():
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("--files", type=str, nargs="+", required=True, help="Video files to process")
    parser.add_argument("--detector-path", type=str, default="yolo12n.pt", help="YOLO model path")
    parser.add_argument("--img-size", type=int, default=720, help="YOLO inference size (pixels)")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="YOLO object confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="YOLO IOU threshold for NMS")
    parser.add_argument("--classes", nargs="+", type=int, help="Filter by class IDs, e.g. --classes 0 2 3")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
    parser.add_argument("--track-points", type=str, default="bbox", choices=["bbox", "centroid"], help="Track points mode")

    args = parser.parse_args()

    model = YOLOn(args.detector_path, device=args.device)

    for input_path in args.files:
        print(f"Processing {input_path}...")
        video = Video(input_path=input_path)
        distance_function = "iou" if args.track_points == "bbox" else "euclidean"
        distance_threshold = (
            DISTANCE_THRESHOLD_BBOX
            if args.track_points == "bbox"
            else DISTANCE_THRESHOLD_CENTROID
        )

        tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
        )

        for frame in video:
            yolo_result = model(
                frame,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                image_size=args.img_size,
                classes=args.classes,
            )

            detections = yolo_detections_to_norfair_detections(yolo_result, track_points=args.track_points)
            tracked_objects = tracker.update(detections=detections)

            if args.track_points == "centroid":
                norfair.draw_points(frame, detections)
                norfair.draw_tracked_objects(frame, tracked_objects)
            else:
                norfair.draw_boxes(frame, detections)
                norfair.draw_tracked_boxes(frame, tracked_objects)

            video.write(frame)

        video.release()
        print(f"✅ Finished processing {input_path}")


if __name__ == "__main__":
    main()
