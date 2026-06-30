"""Debug why pair analyzer detects 0 pairs on fight videos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from config.settings import get_settings
from src.common.logger import setup_logger
from src.pipeline.orchestrator import SyncPipeline

setup_logger()

settings = get_settings()
settings.mock_mode = True
settings.layer1.use_mock = True
settings.layer2.use_mock = True
settings.layer4a.use_mock = True
settings.layer4b.use_mock = True

video_path = "data/fight_samples/Fight Detection Video Dataset/sample_1.mp4"
cap = cv2.VideoCapture(video_path)
fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {fc}")

cameras = [{"camera_id": "cam_debug", "name": "Debug", "rtsp_url": "", "location": ""}]
pipeline = SyncPipeline(settings, cameras=cameras)

filter_counts = {
    "low_kp_conf": 0,
    "dist_too_far": 0,
    "iou_too_low": 0,
    "dot_none": 0,
    "dot_not_negative": 0,
    "sustained_too_short": 0,
    "passed_all": 0,
    "total_pairs_evaluated": 0,
}

frame_idx = 0
while frame_idx < 600:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    frame_resized = cv2.resize(frame, (1280, 720))

    result = pipeline.process_frame(
        camera_id="cam_debug",
        frame=frame_resized,
        frame_id=frame_idx,
        timestamp=frame_idx / 30.0,
    )

    # Manually trace pair filtering
    if frame_idx % 10 == 0:
        pipe = pipeline._pipelines["cam_debug"]
        tracklets = pipe.track_manager.tracklets
        tracklet_list = list(tracklets.values())
        tl = len(tracklet_list)
        pair_count = tl * (tl - 1) // 2 if tl >= 2 else 0

        if pair_count > 0:
            from src.layer3_pair_analysis.distance_matrix import (
                bbox_center, bbox_iou, euclidean, face_to_face_dot,
            )
            for i in range(tl):
                for j in range(i + 1, tl):
                    a, b = tracklet_list[i], tracklet_list[j]
                    filter_counts["total_pairs_evaluated"] += 1

                    # 1. KP confidence
                    if a.avg_kp_confidence < 0.5 or b.avg_kp_confidence < 0.5:
                        filter_counts["low_kp_conf"] += 1
                        if frame_idx == 10:
                            print(f"  FAIL kp_conf: a={a.avg_kp_confidence:.3f} b={b.avg_kp_confidence:.3f}")
                        continue

                    # 2. Distance
                    c_a = bbox_center(a.bbox)
                    c_b = bbox_center(b.bbox)
                    dist = euclidean(c_a, c_b)
                    avg_h = (a.height + b.height) / 2.0
                    if avg_h < 1e-3 or dist >= 0.8 * avg_h:
                        filter_counts["dist_too_far"] += 1
                        if frame_idx == 10:
                            print(f"  FAIL dist: dist={dist:.1f} avg_h={avg_h:.1f} ratio={dist/max(avg_h,1e-6):.3f} need < 0.8")
                        continue

                    # 3. IoU
                    iou = bbox_iou(a.bbox, b.bbox)
                    if iou < 0.15:
                        filter_counts["iou_too_low"] += 1
                        if frame_idx == 10:
                            print(f"  FAIL iou: iou={iou:.4f} need >= 0.15")
                            print(f"    bbox_a={a.bbox} bbox_b={b.bbox}")
                        continue

                    # 4. Face-to-face
                    dot = face_to_face_dot(a.keypoints, b.keypoints)
                    if dot is None:
                        filter_counts["dot_none"] += 1
                        if frame_idx == 10:
                            print(f"  FAIL dot: None")
                        continue
                    if dot >= -0.3:
                        filter_counts["dot_not_negative"] += 1
                        if frame_idx == 10:
                            print(f"  FAIL dot: dot={dot:.4f} need < -0.3")
                        continue

                    # 5. Sustained proximity
                    key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                    sustained = pipe.pair_analyzer._proximity_counts.get(key, 0) + 1
                    if sustained < 15:
                        filter_counts["sustained_too_short"] += 1
                        continue

                    filter_counts["passed_all"] += 1
                    print(f"  Frame {frame_idx}: PASSED ALL tracks {a.track_id},{b.track_id}")

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}: tracklets={tl} pairs_evaluated={pair_count}")

cap.release()

print("\n=== FILTER BREAKDOWN ===")
for k, v in filter_counts.items():
    print(f"  {k}: {v}")
print(f"Interactions in results: {sum(1 for r in result.get('interactions', []) if r.label != 'none')}")
