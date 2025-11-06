import cv2
import os

def try_create_tracker(name: str):
    """
    Create an OpenCV tracker by name with compatibility across OpenCV versions.
    Supported: 'CSRT', 'KCF', 'MOSSE', 'NANO'
    """
    name = name.upper()

    def _create(mod, attr):
        return getattr(mod, attr)()

    if name == "CSRT":
        for mod_name in ("cv2", "cv2.legacy"):
            mod = getattr(cv2, "legacy") if mod_name == "cv2.legacy" else cv2
            if hasattr(mod, "TrackerCSRT_create"):
                return _create(mod, "TrackerCSRT_create")
        raise RuntimeError("CSRT tracker not available in your OpenCV build.")

    if name == "KCF":
        for mod_name in ("cv2", "cv2.legacy"):
            mod = getattr(cv2, "legacy") if mod_name == "cv2.legacy" else cv2
            if hasattr(mod, "TrackerKCF_create"):
                return _create(mod, "TrackerKCF_create")
        raise RuntimeError("KCF tracker not available in your OpenCV build.")

    if name == "MOSSE":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        raise RuntimeError("MOSSE tracker not available in your OpenCV build.")

    if name == "NANO":
        if not hasattr(cv2, "TrackerNano_Params") or not hasattr(cv2, "TrackerNano_create"):
            raise RuntimeError("TrackerNano not available in your OpenCV build.")
        return init_nano()

    raise ValueError(f"Unknown tracker name: {name}")


def init_nano():
    params = cv2.TrackerNano_Params()
    params.backbone = os.path.abspath("nano/nanotrack_backbone_sim.onnx")
    params.neckhead = os.path.abspath("nano/nanotrack_head_sim.onnx")
    return cv2.TrackerNano_create(params)


def xyxy_to_xywh(x1, y1, x2, y2):
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def clip_bbox_xywh(bbox, w, h):
    x, y, bw, bh = bbox
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return (int(x), int(y), int(bw), int(bh))


def scale_bbox(bbox, scale_x, scale_y):
    x, y, w, h = bbox
    return (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))