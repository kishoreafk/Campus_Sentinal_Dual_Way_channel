"""NMS and keypoint post-processing for YOLOv8-Pose output."""

from __future__ import annotations

from typing import List

import numpy as np


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.65) -> List[int]:
    """Greedy NMS. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_threshold]
    return keep


def parse_yolov8_pose_output(
    output: np.ndarray,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.65,
    orig_shape: tuple[int, int] = (720, 1280),
    input_size: int = 640,
):
    """Parse YOLOv8-Pose raw TensorRT output into detection list.

    output shape: [B, 56, 8400]   (56 = 4 bbox + 1 conf + 17*3 kps)
    Returns list (per batch) of detection dicts.
    """
    import numpy as np

    if output.ndim == 3:
        outputs = output
    else:
        outputs = output[None, ...]

    H, W = orig_shape
    results = []
    for b in range(outputs.shape[0]):
        pred = outputs[b].T  # [8400, 56]
        # Filter by confidence
        scores = pred[:, 4]
        mask = scores > conf_threshold
        pred = pred[mask]
        if len(pred) == 0:
            results.append([])
            continue

        # Boxes (cx, cy, w, h) -> xyxy (scaled back to original)
        boxes = xywh2xyxy(pred[:, :4])
        # Scale boxes from input_size space to original image space
        scale = max(H, W) / input_size
        boxes = boxes * scale

        # NMS
        keep = nms(boxes, pred[:, 4], iou_threshold)
        kept = pred[keep]
        kept_boxes = boxes[keep]

        dets = []
        for box, row in zip(kept_boxes, kept):
            kp_flat = row[5:]  # 51 = 17 * 3
            kps = kp_flat.reshape(17, 3).copy()
            kps[:, 0] *= scale  # x
            kps[:, 1] *= scale  # y
            kp_scores = kps[:, 2].tolist()
            kp_list = kps.tolist()
            dets.append({
                "bbox": tuple(float(v) for v in box.tolist()),
                "confidence": float(row[4]),
                "keypoints": kp_list,
                "keypoint_scores": [float(s) for s in kp_scores],
            })
        results.append(dets)
    return results
