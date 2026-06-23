"""Prometheus metrics for Layer 7 monitoring.

All layers register metrics here. The FastAPI app exposes /metrics.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------
# Layer 0: Ingestion
# ---------------------------------------------------------------------
FRAMES_INGESTED = Counter(
    "cctv_frames_ingested_total",
    "Frames ingested from RTSP",
    ["camera_id"],
    registry=REGISTRY,
)
FRAMES_DROPPED = Counter(
    "cctv_frames_dropped_total",
    "Frames dropped due to backpressure",
    ["camera_id"],
    registry=REGISTRY,
)
INGESTION_LATENCY = Histogram(
    "cctv_ingestion_latency_seconds",
    "Time from frame capture to ingestion queue",
    registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------
# Layer 1: Detection
# ---------------------------------------------------------------------
DETECTION_BATCH_SIZE = Histogram(
    "cctv_detection_batch_size",
    "Detection batch size actually run",
    registry=REGISTRY,
    buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64),
)
DETECTION_LATENCY = Histogram(
    "cctv_detection_latency_seconds",
    "Detection forward+postprocess latency per batch",
    registry=REGISTRY,
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
PERSONS_DETECTED = Counter(
    "cctv_persons_detected_total",
    "Persons detected (post-NMS)",
    ["camera_id"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------
# Layer 2: Tracking
# ---------------------------------------------------------------------
ACTIVE_TRACKLETS = Gauge(
    "cctv_active_tracklets",
    "Active tracklets per camera",
    ["camera_id"],
    registry=REGISTRY,
)
REID_MATCHES = Counter(
    "cctv_reid_matches_total",
    "OSNet Re-ID successful matches",
    ["camera_id"],
    registry=REGISTRY,
)
TRACKING_LATENCY = Histogram(
    "cctv_tracking_latency_seconds",
    "Tracking update latency per frame",
    registry=REGISTRY,
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)

# ---------------------------------------------------------------------
# Layer 3: Pair analysis
# ---------------------------------------------------------------------
PAIRS_DETECTED = Counter(
    "cctv_pairs_detected_total",
    "Interacting pairs detected",
    ["camera_id"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------
# Layer 4: Recognition
# ---------------------------------------------------------------------
CASCADE_FILTER_RATE = Gauge(
    "cctv_cascade_filter_rate",
    "Fraction of pairs that passed cascade filter (skipped SlowFast)",
    registry=REGISTRY,
)
INTERACTION_PREDICTIONS = Counter(
    "cctv_interaction_predictions_total",
    "Interaction predictions",
    ["camera_id", "label"],
    registry=REGISTRY,
)
INDIVIDUAL_PREDICTIONS = Counter(
    "cctv_individual_predictions_total",
    "Individual action predictions",
    ["camera_id", "label"],
    registry=REGISTRY,
)
RECOGNITION_LATENCY = Histogram(
    "cctv_recognition_latency_seconds",
    "Action recognition latency",
    ["branch"],  # interaction | individual
    registry=REGISTRY,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# ---------------------------------------------------------------------
# Layer 5: Post-processing
# ---------------------------------------------------------------------
EMA_SMOOTHED_SCORE = Gauge(
    "cctv_ema_smoothed_score",
    "EMA-smoothed score",
    ["camera_id", "action_type"],
    registry=REGISTRY,
)
STATE_MACHINE_TRANSITIONS = Counter(
    "cctv_state_machine_transitions_total",
    "State machine transitions",
    ["from_state", "to_state"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------
# Layer 6: Alerts
# ---------------------------------------------------------------------
ALERTS_GENERATED = Counter(
    "cctv_alerts_generated_total",
    "Alerts generated (after dedup)",
    ["camera_id", "action_type"],
    registry=REGISTRY,
)
ALERTS_SUPPRESSED = Counter(
    "cctv_alerts_suppressed_total",
    "Alerts suppressed by dedup window",
    ["camera_id", "action_type"],
    registry=REGISTRY,
)
ALERT_LATENCY = Histogram(
    "cctv_alert_latency_seconds",
    "End-to-end alert latency (frame capture -> alert)",
    registry=REGISTRY,
    buckets=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0),
)

# ---------------------------------------------------------------------
# System
# ---------------------------------------------------------------------
GPU_UTILIZATION = Gauge(
    "cctv_gpu_utilization_percent",
    "GPU utilization",
    ["gpu_id"],
    registry=REGISTRY,
)
GPU_MEMORY = Gauge(
    "cctv_gpu_memory_used_bytes",
    "GPU memory used",
    ["gpu_id"],
    registry=REGISTRY,
)


def metrics_text() -> str:
    """Render metrics in Prometheus text format."""
    return generate_latest(REGISTRY).decode("utf-8")
