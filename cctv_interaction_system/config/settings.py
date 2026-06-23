"""Application configuration using pydantic-settings.

All tunable parameters live here. Environment variables override defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    max_memory: str = "8gb"
    maxmemory_policy: str = "allkeys-lru"

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "cctv"
    password: str = "cctv"
    database: str = "cctv_alerts"

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class Layer0Config(BaseModel):
    """Video ingestion."""
    detection_stride: int = 5  # Detect every Nth frame (30 FPS / 5 = 6 FPS)
    target_fps: int = 30
    frame_width: int = 1280
    frame_height: int = 720
    rtsp_transport: str = "tcp"  # tcp is more reliable than udp
    ffmpeg_bufsize: int = 10 ** 8
    reconnect_delay_s: float = 2.0
    max_reconnect_attempts: int = 5


class Layer1Config(BaseModel):
    """Detection / pose estimation."""
    model_path: str = "models/yolov8n-pose.engine"
    onnx_path: str = "models/yolov8n-pose.onnx"
    weights_path: str = "models/yolov8n-pose.pt"
    imgsz: int = 640
    max_batch: int = 32
    batch_timeout_ms: int = 50
    half_precision: bool = True
    conf_threshold: float = 0.35
    iou_threshold: float = 0.65
    device_id: int = 0
    # If true, fall back to a CPU mock (used in tests / CI without GPU)
    use_mock: bool = True


class Layer2Config(BaseModel):
    """Tracking."""
    track_thresh: float = 0.5
    track_high_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30
    # OSNet Re-ID
    reid_model: str = "osnet_x0_25"
    reid_feature_dim: int = 512
    reid_input_h: int = 256
    reid_input_w: int = 128
    reid_cosine_threshold: float = 0.6
    # Kalman
    kalman_process_noise: float = 0.01
    # Skeleton buffer (per person)
    skeleton_buffer_len: int = 48
    use_mock: bool = True


class Layer3Config(BaseModel):
    """Pair analysis."""
    # All thresholds must pass for a pair to be considered interacting
    distance_ratio_threshold: float = 0.8  # distance < ratio * avg_height
    iou_threshold: float = 0.15
    face_to_face_dot_threshold: float = -0.3  # dot product < threshold
    sustained_proximity_frames: int = 15  # ~0.5s @ 30 FPS
    min_keypoint_confidence: float = 0.5


class Layer4AConfig(BaseModel):
    """Interaction recognition (cascade)."""
    poseconv3d_engine_path: str = "models/poseconv3d_pair.engine"
    slowfast_engine_path: str = "models/slowfast.engine"
    # Cascade filter
    cascade_score_threshold: float = 0.4
    # SlowFast input
    slowfast_clip_len: int = 32
    slowfast_img_size: int = 224
    slowfast_batch: int = 16
    # PoseConv3D input
    pose_clip_len: int = 48
    pose_num_keypoints: int = 17
    pose_batch: int = 16
    # Fusion
    fusion_pose_weight: float = 0.6
    fusion_rgb_weight: float = 0.4
    roi_margin_ratio: float = 0.2
    use_mock: bool = True
    interaction_labels: List[str] = Field(
        default_factory=lambda: [
            "hug", "kiss", "fight", "push", "handshake",
            "high-five", "other", "none",
        ]
    )


class Layer4BConfig(BaseModel):
    """Individual action recognition."""
    poseconv3d_engine_path: str = "models/poseconv3d_individual.engine"
    pose_clip_len: int = 48
    pose_num_keypoints: int = 17
    batch_size: int = 64
    conf_threshold: float = 0.6
    use_mock: bool = True
    individual_labels: List[str] = Field(
        default_factory=lambda: [
            "walking", "standing", "running", "sitting",
            "waiting", "other", "none",
        ]
    )


class Layer5Config(BaseModel):
    """Post-processing."""
    ema_alpha: float = 0.7  # weight on previous score
    # State machine thresholds (frames)
    state_candidate_frames: int = 5
    state_confirmed_frames: int = 10
    state_alert_frames: int = 15
    state_candidate_score: float = 0.5
    state_confirmed_score: float = 0.7
    state_alert_score: float = 0.8
    state_reset_score: float = 0.3
    state_reset_frames: int = 5
    # Alert dedup
    dedup_window_s: float = 3.0


class Layer6Config(BaseModel):
    """Alert management."""
    clip_duration_s: int = 10  # 5s before + 5s after
    clip_before_s: int = 5
    clip_after_s: int = 5
    clip_bitrate: int = 2_000_000  # 2 Mbps
    clip_codec: str = "libx264"
    clip_storage_path: str = "data/clips"
    hot_retention_days: int = 30
    cold_retention_days: int = 365


class Layer7Config(BaseModel):
    """Monitoring / API."""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    prometheus_port: int = 9090
    grafana_port: int = 3000
    metrics_interval_s: float = 5.0


class Settings(BaseSettings):
    """Top-level settings."""

    model_config = SettingsConfigDict(
        env_prefix="CCTV_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # General
    environment: str = "development"
    log_level: str = "INFO"
    # When True, all GPU components use CPU mocks (testing / CI / no-GPU dev)
    mock_mode: bool = True

    # Sub-configs
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    layer0: Layer0Config = Field(default_factory=Layer0Config)
    layer1: Layer1Config = Field(default_factory=Layer1Config)
    layer2: Layer2Config = Field(default_factory=Layer2Config)
    layer3: Layer3Config = Field(default_factory=Layer3Config)
    layer4a: Layer4AConfig = Field(default_factory=Layer4AConfig)
    layer4b: Layer4BConfig = Field(default_factory=Layer4BConfig)
    layer5: Layer5Config = Field(default_factory=Layer5Config)
    layer6: Layer6Config = Field(default_factory=Layer6Config)
    layer7: Layer7Config = Field(default_factory=Layer7Config)

    # Camera config file (YAML)
    cameras_file: str = "config/cameras.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def load_cameras(cameras_file: str | None = None) -> List[dict]:
    """Load camera definitions from YAML.

    Each entry: {camera_id, rtsp_url, name, location}
    """
    if cameras_file is None:
        cameras_file = str(PROJECT_ROOT / "config" / "cameras.yaml")
    path = Path(cameras_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or []
    return data.get("cameras", data) if isinstance(data, dict) else data
