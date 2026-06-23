"""Export models to TensorRT engines.

This script is a placeholder — actual export requires GPU + TensorRT installed.
Reads model weights and writes engine files.

Usage (with GPU):
    python -m scripts.export_models --model yolov8n-pose
    python -m scripts.export_models --model slowfast
    python -m scripts.export_models --model poseconv3d-pair
    python -m scripts.export_models --model poseconv3d-individual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logger import get_logger, setup_logger  # noqa

setup_logger()
logger = get_logger()


def export_yolov8_pose(weights: str = "yolov8n-pose.pt", imgsz: int = 640,
                       batch: int = 32, half: bool = True) -> str:
    """Export YOLOv8-Pose to TensorRT."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed — install with: pip install ultralytics")
        sys.exit(1)
    model = YOLO(weights)
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        dynamic=True,
        batch=batch,
        workspace=4,
        device=0,
    )
    logger.info(f"YOLOv8-Pose engine written to {engine_path}")
    return engine_path


def export_slowfast(weights: str = "slowfast_r50.pt") -> str:
    """Export SlowFast to TensorRT via ONNX."""
    logger.warning("SlowFast export requires mmaction2 + manual ONNX export")
    logger.info("Step 1: Build model with mmaction2")
    logger.info("Step 2: torch.onnx.export(model, dummy_input, 'slowfast.onnx', dynamic_axes={'input': {0: 'batch'}})")
    logger.info("Step 3: trtexec --onnx=slowfast.onnx --saveEngine=slowfast.engine --fp16")
    return "slowfast.engine"


def export_poseconv3d(weights: str = "poseconv3d.pth", mode: str = "pair") -> str:
    """Export PoseConv3D to TensorRT."""
    logger.warning(f"PoseConv3D ({mode}) export requires mmaction2 + manual ONNX export")
    return f"poseconv3d_{mode}.engine"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["yolov8n-pose", "slowfast", "poseconv3d-pair", "poseconv3d-individual"])
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    if args.model == "yolov8n-pose":
        export_yolov8_pose(args.weights or "yolov8n-pose.pt")
    elif args.model == "slowfast":
        export_slowfast(args.weights or "slowfast_r50.pt")
    elif args.model == "poseconv3d-pair":
        export_poseconv3d(args.weights or "poseconv3d_pair.pth", mode="pair")
    elif args.model == "poseconv3d-individual":
        export_poseconv3d(args.weights or "poseconv3d_individual.pth", mode="single")


if __name__ == "__main__":
    main()
