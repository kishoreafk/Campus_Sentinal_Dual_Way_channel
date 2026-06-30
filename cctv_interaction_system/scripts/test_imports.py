import sys
sys.path.insert(0, ".")
from config.settings import get_settings, load_cameras
from src.pipeline.orchestrator import SyncPipeline
import numpy as np
print("All imports work!")
settings = get_settings()
print(f"Settings mock_mode: {settings.mock_mode}")
cameras = load_cameras()
print(f"Cameras loaded: {len(cameras)}")
