"""Export models to TensorRT engines and ONNX.

Converts PyTorch weights → ONNX → TensorRT engine files.

Usage:
    # YOLOv8-Pose
    python -m scripts.export_models --model yolov8n-pose --weights yolov8n-pose.pt

    # SlowFast (requires mmaction2)
    python -m scripts.export_models --model slowfast --weights slowfast_r50.pth

    # PoseConv3D (requires mmaction2)
    python -m scripts.export_models --model poseconv3d-pair --weights poseconv3d_pair.pth
    python -m scripts.export_models --model poseconv3d-individual --weights poseconv3d_individual.pth

    # Export all
    python -m scripts.export_models --model all

Requirements:
    pip install ultralytics torch onnx onnxruntime-gpu tensorrt pycuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logger import get_logger, setup_logger  # noqa

setup_logger()
logger = get_logger()

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _trtexec(onnx_path: str, engine_path: str, half: bool = True, workspace: int = 4) -> str:
    """Run trtexec to convert ONNX to TensorRT engine."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace * 1024 * 1024 * 1024}",
    ]
    if half:
        cmd.append("--fp16")
    logger.info(f"Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"trtexec failed: {result.stderr[:500]}")
        raise RuntimeError(f"trtexec failed for {onnx_path}")
    logger.info(f"Engine written to {engine_path}")
    return engine_path


def export_yolov8_pose_to_onnx(weights: str, imgsz: int = 640, batch: int = 32) -> str:
    """Export YOLOv8-Pose PyTorch → ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed: pip install ultralytics")
        sys.exit(1)

    model = YOLO(weights)
    onnx_path = str(MODELS_DIR / "yolov8n-pose.onnx")
    model.export(format="onnx", imgsz=imgsz, half=False, dynamic=True, batch=batch)
    # ultralytics exports to cwd by default; move to models dir
    cwd_onnx = Path(weights).with_suffix(".onnx")
    if cwd_onnx.exists():
        cwd_onnx.rename(onnx_path)
    logger.info(f"ONNX written to {onnx_path}")
    return onnx_path


def export_yolov8_pose(weights: str = "yolov8n-pose.pt", imgsz: int = 640,
                       batch: int = 32, half: bool = True) -> str:
    """Export YOLOv8-Pose to TensorRT (via ONNX → trtexec)."""
    onnx_path = export_yolov8_pose_to_onnx(weights, imgsz, batch)
    engine_path = str(MODELS_DIR / "yolov8n-pose.engine")
    return _trtexec(onnx_path, engine_path, half=half)


def export_torch_to_onnx(
    model_class: str,
    weights: str,
    onnx_path: str,
    dummy_input,
    input_name: str = "input",
    dynamic_batch: bool = True,
):
    """Generic PyTorch → ONNX export."""
    import torch

    if model_class == "slowfast":
        from mmaction.apis import init_model
        config = "configs/recognition/slowfast/slowfast_r50_8xb8-8x8x256-256e_kinetics400.py"
        model = init_model(config, weights, device="cpu")
    elif "poseconv3d" in model_class:
        from mmaction.apis import init_model
        mode = "pair" if "pair" in model_class else "single"
        config = f"configs/skeleton/poseconv3d/slowonly_r50_8xb16-u48-240e_ntu120-xsub-keypoint-{mode}.py"
        model = init_model(config, weights, device="cpu")
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=[input_name],
        output_names=["output"],
        dynamic_axes={input_name: {0: "batch"}} if dynamic_batch else None,
        opset_version=17,
    )
    logger.info(f"ONNX written to {onnx_path}")


def export_slowfast(weights: str = "slowfast_r50.pth") -> str:
    """Export SlowFast to TensorRT (requires mmaction2)."""
    onnx_path = str(MODELS_DIR / "slowfast.onnx")
    engine_path = str(MODELS_DIR / "slowfast.engine")

    import torch
    dummy = torch.randn(1, 3, 32, 224, 224)
    export_torch_to_onnx("slowfast", weights, onnx_path, dummy)
    return _trtexec(onnx_path, engine_path)


def export_poseconv3d(weights: str = "poseconv3d.pth", mode: str = "pair") -> str:
    """Export PoseConv3D to TensorRT (requires mmaction2)."""
    suffix = "pair" if mode == "pair" else "individual"
    onnx_path = str(MODELS_DIR / f"poseconv3d_{suffix}.onnx")
    engine_path = str(MODELS_DIR / f"poseconv3d_{suffix}.engine")

    import torch
    T, V, M = 48, 17, 2 if mode == "pair" else 1
    dummy = torch.randn(1, 3, T, V, M)
    export_torch_to_onnx(f"poseconv3d_{mode}", weights, onnx_path, dummy)
    return _trtexec(onnx_path, engine_path)


def export_osnet_to_onnx(weights: str = "osnet_x0_25.pth") -> str:
    """Export OSNet ReID to ONNX (requires torchreid)."""
    onnx_path = str(MODELS_DIR / "osnet_x0_25.onnx")
    import torch
    try:
        import torchreid
        model = torchreid.models.build_model(name="osnet_x0_25", num_classes=1, pretrained=False)
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state)
    except Exception:
        model = torch.hub.load("bubbliiiing/osnet-pytorch", "osnet_x0_25", pretrained=True)
    model.eval()
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}}, opset_version=17)
    logger.info(f"ONNX written to {onnx_path}")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX / TensorRT")
    parser.add_argument("--model", required=True,
                        choices=["yolov8n-pose", "slowfast", "poseconv3d-pair",
                                 "poseconv3d-individual", "osnet", "all"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--half", action="store_true", default=True)
    args = parser.parse_args()

    if args.model == "all":
        export_yolov8_pose(args.weights or f"{MODELS_DIR}/yolov8n-pose.pt")
        # Remaining models require mmaction2 / torchreid — try but don't exit
        try:
            export_slowfast(args.weights or f"{MODELS_DIR}/slowfast_r50.pth")
        except Exception as e:
            logger.warning(f"SlowFast export skipped: {e}")
        try:
            export_poseconv3d(args.weights or f"{MODELS_DIR}/poseconv3d_pair.pth", mode="pair")
        except Exception as e:
            logger.warning(f"PoseConv3D pair export skipped: {e}")
        try:
            export_poseconv3d(args.weights or f"{MODELS_DIR}/poseconv3d_individual.pth", mode="single")
        except Exception as e:
            logger.warning(f"PoseConv3D individual export skipped: {e}")
        try:
            export_osnet_to_onnx(args.weights or f"{MODELS_DIR}/osnet_x0_25.pth")
        except Exception as e:
            logger.warning(f"OSNet export skipped: {e}")
    elif args.model == "yolov8n-pose":
        export_yolov8_pose(args.weights or f"{MODELS_DIR}/yolov8n-pose.pt", half=args.half)
    elif args.model == "slowfast":
        export_slowfast(args.weights or f"{MODELS_DIR}/slowfast_r50.pth")
    elif args.model == "poseconv3d-pair":
        export_poseconv3d(args.weights or f"{MODELS_DIR}/poseconv3d_pair.pth", mode="pair")
    elif args.model == "poseconv3d-individual":
        export_poseconv3d(args.weights or f"{MODELS_DIR}/poseconv3d_individual.pth", mode="single")
    elif args.model == "osnet":
        export_osnet_to_onnx(args.weights or f"{MODELS_DIR}/osnet_x0_25.pth")


if __name__ == "__main__":
    main()
