"""Check if real model engines exist."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
s = get_settings()
s.mock_mode = False

print("layer1 (detector) engine:", s.layer1.detector_engine_path)
print("  exists:", Path(s.layer1.detector_engine_path).exists() if s.layer1.detector_engine_path else "N/A")
print("layer2 tracker:", s.layer2.tracker_type)
print("layer4a slowfast:", s.layer4a.slowfast_engine_path)
print("  exists:", Path(s.layer4a.slowfast_engine_path).exists() if s.layer4a.slowfast_engine_path else "N/A")
print("layer4a poseconv3d:", s.layer4a.poseconv3d_engine_path)
print("  exists:", Path(s.layer4a.poseconv3d_engine_path).exists() if s.layer4a.poseconv3d_engine_path else "N/A")
print("layer4b poseconv3d:", s.layer4b.poseconv3d_engine_path)
print("  exists:", Path(s.layer4b.poseconv3d_engine_path).exists() if s.layer4b.poseconv3d_engine_path else "N/A")
print()

# Try creating real detectors/models
try:
    from src.layer1_detection.detector import Detector
    d = Detector(use_mock=False)
    print("Detector(use_mock=False): OK")
except Exception as e:
    print(f"Detector(use_mock=False): FAILED - {e}")

try:
    from src.layer2_tracking.track_manager import make_track_manager
    tm = make_track_manager("test")
    print("TrackManager: OK")
except Exception as e:
    print(f"TrackManager: FAILED - {e}")

try:
    from src.layer4a_interaction.slowfast import SlowFast
    sf = SlowFast(use_mock=False)
    print("SlowFast(use_mock=False): OK")
except Exception as e:
    print(f"SlowFast(use_mock=False): FAILED - {e}")

try:
    from src.layer4a_interaction.poseconv3d import PoseConv3D
    pc = PoseConv3D(use_mock=False)
    print("PoseConv3D(use_mock=False): OK")
except Exception as e:
    print(f"PoseConv3D(use_mock=False): FAILED - {e}")

try:
    from src.layer4b_individual.individual_recognizer import make_individual_recognizer
    ir = make_individual_recognizer()
    print("IndividualRecognizer: OK")
except Exception as e:
    print(f"IndividualRecognizer: FAILED - {e}")

try:
    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
except ImportError as e:
    print(f"TensorRT import: FAILED - {e}")
