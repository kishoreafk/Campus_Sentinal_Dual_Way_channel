"""Check available models and TensorRT."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROJECT_ROOT

print("=== Model files ===")
model_dir = PROJECT_ROOT / "models"
if model_dir.exists():
    for f in sorted(model_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"  No models/ directory at {model_dir}")

print()
print("=== TensorRT ===")
try:
    import tensorrt as trt
    print(f"  TensorRT version: {trt.__version__}")
except ImportError:
    print("  TensorRT not installed")

print()
print("=== ONNX Runtime ===")
try:
    import onnxruntime as ort
    print(f"  ONNX Runtime version: {ort.__version__}")
except ImportError:
    print("  ONNX Runtime not installed")

print()
print("=== CUDA ===")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("  PyTorch not installed")
