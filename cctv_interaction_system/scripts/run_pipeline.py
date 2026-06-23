"""CLI entry point — run the CCTV pipeline.

Usage:
    python -m scripts.run_pipeline --mock --cameras config/cameras.yaml
    python -m scripts.run_pipeline --rtsp --cameras config/cameras.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings, load_cameras  # noqa
from src.common.logger import get_logger, setup_logger  # noqa
from src.pipeline.orchestrator import Pipeline  # noqa

setup_logger()
logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="CCTV Interaction Recognition Pipeline")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use mock ingestion (synthetic frames)")
    parser.add_argument("--rtsp", action="store_true",
                        help="Use real RTSP ingestion (overrides --mock)")
    parser.add_argument("--cameras", default="config/cameras.yaml",
                        help="Path to cameras YAML")
    parser.add_argument("--api", action="store_true",
                        help="Start FastAPI server alongside pipeline")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=0,
                        help="Run for N seconds (0 = forever)")
    args = parser.parse_args()

    settings = get_settings()
    cameras = load_cameras(args.cameras)
    if not cameras:
        logger.error(f"no cameras found in {args.cameras}")
        sys.exit(1)
    logger.info(f"loaded {len(cameras)} cameras")

    use_mock = args.mock and not args.rtsp
    pipeline = Pipeline(settings=settings, cameras=cameras, use_mock_ingestion=use_mock)

    # Optional API server
    api_thread = None
    if args.api:
        import threading
        import uvicorn
        from src.layer7_monitoring.api import create_app

        app = create_app(alert_manager=pipeline.alert_manager, cameras=cameras)
        config = uvicorn.Config(app, host="0.0.0.0", port=args.api_port, log_level="warning")
        server = uvicorn.Server(config)

        def _run():
            server.run()
        api_thread = threading.Thread(target=_run, daemon=True, name="api-server")
        api_thread.start()
        logger.info(f"API server started on :{args.api_port}")

    def _shutdown(signum, frame):
        logger.info("shutting down...")
        pipeline.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pipeline.start()
    logger.info("Pipeline running. Press Ctrl+C to stop.")

    try:
        if args.duration > 0:
            time.sleep(args.duration)
            pipeline.stop()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
