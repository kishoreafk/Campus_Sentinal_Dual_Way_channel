"""Create a synthetic interaction video with two people close together.

Then run the pipeline with a custom detector that generates close
detections, to verify the interaction detection flow works end-to-end.
"""
import sys
import time
import math
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from src.common.logger import get_logger, setup_logger
from src.common.schemas import Detection, FrameDetections
from src.layer1_detection.detector import MockDetector
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


class CloseProximityDetector(MockDetector):
    """Mock detector that places two people walking very close together.

    Overrides detect_batch to keep people close so pair analysis can find them.
    """

    def detect_batch(self, frames, metas):
        results = []
        for frame, meta in zip(frames, metas):
            H, W = frame.shape[:2]
            camera_id = meta["camera_id"]
            frame_id = meta["frame_id"]
            ts = meta["timestamp"]
            rng = np.random.default_rng(hash((camera_id, frame_id)) & 0xFFFFFFFF)

            detections = []
            # Two people facing each other (opposite directions)
            facings = [-6.0, 6.0]
            for i in range(2):  # Always 2 people
                phase = frame_id * 0.05
                cx_base = W * 0.5 + 100 * math.sin(phase)
                cy_base = H * 0.55
                cx = cx_base + i * 50  # 50px apart
                cy = cy_base + int(10 * math.cos(phase * 0.5 + i))
                w = 80
                h = 160
                x1 = max(0, cx - w / 2)
                y1 = max(0, cy - h / 2)
                x2 = min(W, cx + w / 2)
                y2 = min(H, cy + h / 2)
                conf = float(rng.uniform(0.7, 0.97))

                kps = self._synthetic_keypoints(
                    x1, y1, x2, y2, rng, facing=facings[i],
                )
                kp_scores = [float(rng.uniform(0.5, 0.95)) for _ in range(17)]
                detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                    keypoints=kps,
                    keypoint_scores=kp_scores,
                ))

            results.append(FrameDetections(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=ts,
                detections=detections,
            ))
        return results


def main():
    output_path = "/mnt/e/cctv-har-monitor-merged/data/interaction_test.mp4"
    logger.info(f"Creating synthetic interaction video: {output_path}")

    fps = 12
    width, height = 768, 432
    total_frames = 300  # 25 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fid in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background
        frame[:] = (40, 40, 60)

        phase = fid * 0.05
        cx_base = width * 0.5 + 100 * math.sin(phase)
        cy_base = height * 0.55

        for i in range(2):
            cx = cx_base + i * 50
            cy = cy_base + int(10 * math.cos(phase * 0.5 + i))
            # Draw a person-like rectangle
            x1, y1 = int(cx - 40), int(cy - 80)
            x2, y2 = int(cx + 40), int(cy + 80)
            color = (0, 180 + i * 40, 100) if i == 0 else (200, 120, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            # Head
            cv2.circle(frame, (int(cx), int(cy - 90)), 20, color, -1)

        out.write(frame)

    out.release()
    logger.info(f"Video created: {output_path} ({total_frames} frames)")

    # Now run pipeline with CloseProximityDetector
    logger.info("Running pipeline with close-proximity detections...")
    cameras = [{"camera_id": "cam_int", "name": "Interaction Test",
                "rtsp_url": "", "location": "test"}]
    pipeline = SyncPipeline(cameras=cameras)
    pipeline.detector = CloseProximityDetector()

    alert_log = []
    interaction_log = []
    individual_log = []

    for fid in range(1, total_frames + 1):
        phase = fid * 0.05
        cx_base = width * 0.5 + 100 * math.sin(phase)
        cy_base = height * 0.55
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(2):
            cx = cx_base + i * 50
            cy = cy_base + int(10 * math.cos(phase * 0.5 + i))
            x1, y1 = int(cx - 40), int(cy - 80)
            x2, y2 = int(cx + 40), int(cy + 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 70), -1)

        result = pipeline.process_frame(
            camera_id="cam_int",
            frame=frame,
            frame_id=fid,
            timestamp=float(fid),
        )

        pairs = result.get("pairs", [])
        interactions = result.get("interactions", [])
        individuals = result.get("individuals", [])
        tracklets = result.get("tracklets", [])

        for ip in interactions:
            if ip.label != "none":
                interaction_log.append({
                    "frame": fid, "label": ip.label,
                    "confidence": round(ip.confidence, 4),
                })
                logger.info(f"INTERACTION frame={fid}: {ip.label} conf={ip.confidence:.3f}")

        for ind in individuals:
            if ind.label != "none":
                individual_log.append({
                    "frame": fid, "label": ind.label,
                    "confidence": round(ind.confidence, 4),
                })

        if fid % 50 == 0:
            logger.info(f"Frame {fid}/{total_frames}: pairs={len(pairs)} "
                        f"tracklets={len(tracklets)} interactions={len(interaction_log)}")

    # Summary
    print("\n" + "=" * 60)
    print("INTERACTION FLOW TEST RESULTS")
    print("=" * 60)
    print(f"Frames processed: {total_frames}")
    print(f"Total interactions detected: {len(interaction_log)}")
    if interaction_log:
        labels = set(i["label"] for i in interaction_log)
        print(f"Interaction labels: {labels}")
        for il in interaction_log[:15]:
            print(f"  Frame {il['frame']}: {il['label']} ({il['confidence']})")
        if len(interaction_log) > 15:
            print(f"  ... and {len(interaction_log) - 15} more")
    else:
        print("NO interactions detected!")

    print(f"\nTotal individuals detected: {len(individual_log)}")
    if individual_log:
        labels = set(i["label"] for i in individual_log)
        print(f"Individual labels: {labels}")

    # Save results
    import json
    summary = {
        "frames": total_frames,
        "interactions": interaction_log,
        "individuals": individual_log,
    }
    with open("/mnt/e/cctv-har-monitor-merged/data/interaction_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 60)


if __name__ == "__main__":
    main()
