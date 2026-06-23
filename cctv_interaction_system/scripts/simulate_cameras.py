"""Generate synthetic RTSP streams for testing.

Uses FFmpeg's lavfi testsrc + drawnbox to simulate person-like motion.
Each stream is exposed on a different localhost port.

Usage:
    python -m scripts.simulate_cameras --count 5 --start-port 8554
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logger import get_logger, setup_logger  # noqa

setup_logger()
logger = get_logger()


def start_camera(port: int, camera_id: str) -> subprocess.Popen:
    """Start an RTSP server using FFmpeg's lavfi + rtsp output."""
    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-re",
        "-f", "lavfi",
        "-i", "testsrc=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", "30",
        "-f", "rtsp",
        f"rtsp://localhost:{port}/{camera_id}",
    ]
    logger.info(f"[{camera_id}] starting RTSP server on port {port}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--start-port", type=int, default=8554)
    parser.add_argument("--prefix", default="cam")
    args = parser.parse_args()

    procs = []
    for i in range(args.count):
        camera_id = f"{args.prefix}_{i+1:03d}"
        port = args.start_port + i
        procs.append(start_camera(port, camera_id))

    def _shutdown(*_):
        for p in procs:
            p.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(f"simulating {args.count} cameras. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(5)
            for i, p in enumerate(procs):
                if p.poll() is not None:
                    logger.warning(f"camera {i+1} exited, restarting")
                    procs[i] = start_camera(args.start_port + i, f"{args.prefix}_{i+1:03d}")
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
