"""Process a video file through SyncPipeline, printing per-frame results."""
import argparse
import time

import cv2
import numpy as np

from src.pipeline.orchestrator import SyncPipeline
from src.common.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to video file")
    parser.add_argument("--camera-id", default="cam_001")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("video: {}  {}x{}  {} fps  {} frames",
                args.video, int(cap.get(3)), int(cap.get(4)), round(fps, 1), total)

    pipe = SyncPipeline(cameras=[{"camera_id": args.camera_id}])
    t0 = time.time()
    frame_id = 0
    total_pairs = 0
    total_inds = 0
    total_alerts = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        ts = frame_id / fps

        result = pipe.process_frame(args.camera_id, frame, frame_id, timestamp=ts)
        interactions = result.get("interactions", [])
        individuals = result.get("individuals", [])
        alerts = result.get("alerts", [])
        total_pairs += len(interactions)
        total_inds += len(individuals)
        total_alerts += len(alerts)

        if alerts:
            for a in alerts:
                logger.info("frame {} ALERT: {} ({:.2f}) tracks={}",
                            frame_id, a.action_type, a.confidence, a.track_ids)

        if args.max_frames and frame_id >= args.max_frames:
            break

    elapsed = time.time() - t0
    cap.release()

    logger.info("processed {} frames in {:.1f}s ({:.1f} fps)", frame_id, elapsed, frame_id / elapsed)
    logger.info("predictions: {} interactions, {} individuals, {} alerts",
                total_pairs, total_inds, total_alerts)


if __name__ == "__main__":
    main()
