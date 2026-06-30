"""Debug why pair gets 'push' instead of 'fight' on sample_1."""
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

cameras = [{"camera_id": "cam_debug", "name": "Debug", "rtsp_url": "", "location": ""}]
pipeline = SyncPipeline(settings, cameras=cameras)

frame_idx = 0
interaction_label = None

while frame_idx < 50:
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

    interactions = result.get("interactions", [])
    for ip in interactions:
        if ip.label != "none":
            interaction_label = ip.label
            print(f"Frame {frame_idx}: label={ip.label} conf={ip.confidence:.4f} "
                  f"tracks={ip.track_id_a},{ip.track_id_b}")

# Now check skeleton data once we have sustained pairs
pipe = pipeline._pipelines["cam_debug"]
skel_bufs = pipe.track_manager.get_all_skeletons()
print(f"\nSkeleton buffers available: {list(skel_bufs.keys())}")

# Try to trace what center_dist and motion values are seen
for tid, skel in skel_bufs.items():
    print(f"\nTrack {tid}: skeleton shape={skel.shape}")
    if skel.ndim >= 3:
        x = skel[0]
        y = skel[1]
        conf = skel[2]
        T = skel.shape[1]
        V = skel.shape[2]
        M = skel.shape[3]
        print(f"  T={T}, V={V}, M={M}")
        print(f"  x range: [{x.min():.1f}, {x.max():.1f}]")
        print(f"  y range: [{y.min():.1f}, {y.max():.1f}]")
        # motion
        if T > 1:
            dx = np.diff(x, axis=0)
            dy = np.diff(y, axis=0)
            motion = np.sqrt(dx**2 + dy**2) * conf[:-1]
            print(f"  mean motion: {motion.mean():.3f}")
            print(f"  max motion: {motion.max():.3f}")

# If we have 2 tracks, compute center_dist
track_ids = list(skel_bufs.keys())[:2]
if len(track_ids) >= 2:
    a_skel = skel_bufs[track_ids[0]]
    b_skel = skel_bufs[track_ids[1]]
    print(f"\nPair tracks {track_ids}:")
    print(f"  skel_a shape: {a_skel.shape}")
    print(f"  skel_b shape: {b_skel.shape}")
    # center_dist (matching _pair_probs logic)
    a_centers = a_skel[0, :, :, 0].mean(axis=1)
    b_centers = b_skel[0, :, :, 0].mean(axis=1)
    print(f"  a_center (mean over frames): {a_centers.mean():.1f}")
    print(f"  b_center (mean over frames): {b_centers.mean():.1f}")
    center_dist = float(np.abs(a_centers - b_centers).mean())
    print(f"  center_dist: {center_dist:.1f}")
    # motion for each
    for tid, skel in [(track_ids[0], a_skel), (track_ids[1], b_skel)]:
        x = skel[0]
        conf = skel[2]
        T = skel.shape[1]
        if T > 1:
            dx = np.diff(x, axis=0)
            dy = np.diff(skel[1], axis=0)
            motion = np.sqrt(dx**2 + dy**2) * conf[:-1]
            print(f"  track {tid} mean motion: {motion.mean():.3f}")

cap.release()
print(f"\nTotal frames: {frame_idx}")
print(f"Final interaction label: {interaction_label}")
