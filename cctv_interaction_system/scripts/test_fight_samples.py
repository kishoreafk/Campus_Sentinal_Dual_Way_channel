"""Test pipeline on fight detection sample videos."""
import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from src.common.logger import get_logger, setup_logger
from src.pipeline.orchestrator import SyncPipeline

setup_logger()
logger = get_logger()

settings = get_settings()
settings.mock_mode = True
settings.layer1.use_mock = True
settings.layer2.use_mock = True
settings.layer4a.use_mock = True
settings.layer4b.use_mock = True

SAMPLES = [
    "data/fight_samples/Fight Detection Video Dataset/sample_1.mp4",
    "data/fight_samples/Fight Detection Video Dataset/sample_2.mp4",
]


def main():
    for video_path in SAMPLES:
        if not os.path.exists(video_path):
            print(f"SKIP: {video_path} not found")
            continue

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = fc / fps if fps > 0 else 0
        print(f"\n{'='*60}")
        print(f"Video: {video_path}")
        print(f"  {w}x{h}, {fps:.1f}fps, {fc} frames, {dur:.1f}s")
        print(f"{'='*60}")

        cameras = [{"camera_id": "cam_fight", "name": "Fight Test",
                    "rtsp_url": "", "location": "test"}]
        pipeline = SyncPipeline(cameras=cameras)

        frame_idx = 0
        interaction_log = []
        individual_log = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame_resized = cv2.resize(frame, (1280, 720))

            result = pipeline.process_frame(
                camera_id="cam_fight",
                frame=frame_resized,
                frame_id=frame_idx,
                timestamp=time.time(),
            )

            for ip in result.get("interactions", []):
                if ip.label != "none":
                    interaction_log.append({
                        "frame": frame_idx,
                        "label": ip.label,
                        "confidence": round(ip.confidence, 4),
                        "track_a": ip.track_id_a,
                        "track_b": ip.track_id_b,
                    })

            for ind in result.get("individuals", []):
                if ind.label != "none":
                    individual_log.append({
                        "frame": frame_idx,
                        "label": ind.label,
                        "confidence": round(ind.confidence, 4),
                        "track_id": ind.track_id,
                    })

            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx}/{fc} frames...")

        cap.release()

        print(f"\nResults for {video_path}:")
        print(f"  Frames processed: {frame_idx}")
        print(f"  Interactions detected: {len(interaction_log)}")
        for il in interaction_log[:15]:
            print(f"    Frame {il['frame']}: {il['label']} ({il['confidence']}) "
                  f"tracks={il['track_a']},{il['track_b']}")
        if len(interaction_log) > 15:
            print(f"    ... and {len(interaction_log) - 15} more")

        label_counts = {}
        for ind in individual_log:
            lbl = ind["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"  Individual label distribution: {label_counts}")
        print(f"  Individuals detected: {len(individual_log)}")

        out_path = video_path.replace(".mp4", "_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "video": video_path,
                "resolution": f"{w}x{h}",
                "fps": fps,
                "total_frames": frame_idx,
                "interactions": interaction_log,
                "individuals": individual_log,
            }, f, indent=2)
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
