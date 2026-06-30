"""Run the pipeline end-to-end on the demo video in WSL.

Usage:
    python3 scripts/run_demo_video.py /mnt/e/cctv-har-monitor-merged/data/sample.mp4
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from src.common.logger import get_logger, setup_logger
from src.pipeline.orchestrator import SyncPipeline

setup_logger()
logger = get_logger()

# Force mock mode
settings = get_settings()
settings.mock_mode = True
settings.layer1.use_mock = True
settings.layer2.use_mock = True
settings.layer4a.use_mock = True
settings.layer4b.use_mock = True


def main(video_path: str):
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

    # Create a single-camera pipeline
    cameras = [{"camera_id": "cam_demo", "name": "Demo Video",
                "rtsp_url": "", "location": "demo"}]
    pipeline = SyncPipeline(cameras=cameras)

    results_log = []
    alert_log = []
    interaction_log = []
    individual_log = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Resize to match pipeline expectations (1280x720)
        frame_resized = cv2.resize(frame, (1280, 720))

        timestamp = time.time()
        result = pipeline.process_frame(
            camera_id="cam_demo",
            frame=frame_resized,
            frame_id=frame_idx,
            timestamp=timestamp,
        )

        interactions = result.get("interactions", [])
        individuals = result.get("individuals", [])
        alerts = result.get("alerts", [])
        tracklets = result.get("tracklets", [])
        pairs = result.get("pairs", [])

        # Log non-trivial results
        for ip in interactions:
            if ip.label != "none":
                interaction_log.append({
                    "frame": frame_idx,
                    "label": ip.label,
                    "confidence": round(ip.confidence, 4),
                    "track_a": ip.track_id_a,
                    "track_b": ip.track_id_b,
                })

        for ind in individuals:
            if ind.label != "none":
                individual_log.append({
                    "frame": frame_idx,
                    "label": ind.label,
                    "confidence": round(ind.confidence, 4),
                    "track_id": ind.track_id,
                })

        for alert in alerts:
            is_interaction = len(alert.track_ids) > 1
            alert_log.append({
                "frame": frame_idx,
                "alert_id": alert.alert_id,
                "action_type": alert.action_type,
                "confidence": round(alert.confidence, 4),
                "track_ids": alert.track_ids,
                "is_interaction": is_interaction,
            })
            logger.info(f"ALERT frame={frame_idx}: {alert.action_type} "
                        f"tracks={alert.track_ids} conf={alert.confidence:.3f}")

        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames  "
                        f"(tracklets={len(tracklets)} pairs={len(pairs)} "
                        f"interactions={len(interaction_log)} "
                        f"individuals={len(individual_log)} alerts={len(alert_log)})")

    cap.release()

    summary = {
        "video": video_path,
        "total_frames": frame_idx,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "pipeline_mode": "mock",
        "tracklets_seen": len(set(
            t.track_id for t in pipeline._pipelines["cam_demo"].track_manager.tracklets.values()
        )),
        "total_detections": frame_idx * 3,  # approximate
        "interactions_detected": interaction_log,
        "individuals_detected": individual_log,
        "alerts_generated": alert_log,
        "interaction_labels_seen": list(set(i["label"] for i in interaction_log)),
        "individual_labels_seen": list(set(i["label"] for i in individual_log)),
    }

    output_path = Path(video_path).with_suffix(".pipeline_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RUN SUMMARY")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Interactions detected: {len(interaction_log)}")
    for il in interaction_log[:10]:
        print(f"  Frame {il['frame']}: {il['label']} ({il['confidence']}) "
              f"tracks={il['track_a']},{il['track_b']}")
    if len(interaction_log) > 10:
        print(f"  ... and {len(interaction_log) - 10} more")
    print(f"Individuals detected: {len(individual_log)}")
    for ind in individual_log[:10]:
        print(f"  Frame {ind['frame']}: {ind['label']} ({ind['confidence']}) "
              f"track={ind['track_id']}")
    if len(individual_log) > 10:
        print(f"  ... and {len(individual_log) - 10} more")
    print(f"Alerts generated: {len(alert_log)}")
    for a in alert_log:
        print(f"  Frame {a['frame']}: ALERT {a['action_type']} "
              f"tracks={a['track_ids']} conf={a['confidence']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/run_demo_video.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
