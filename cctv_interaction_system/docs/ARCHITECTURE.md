# Architecture

This document summarises the design decisions behind the CCTV Interaction
Recognition System. For full specifications see the original architecture
document that drove this implementation.

## Layer responsibilities

```
+----------------------------------------------------------+
|  Layer 0  | RTSP ingest -> NVDEC decode -> frame router |
+----------------------------------------------------------+
|  Layer 1  | YOLOv8n-Pose TensorRT FP16 (batched)        |
+----------------------------------------------------------+
|  Layer 2  | ByteTrack + OSNet Re-ID + Kalman + SkelBuf  |
+----------------------------------------------------------+
|  Layer 3  | Pair analysis (distance/IoU/face-to-face)   |
+----------------------------------------------------------+
|  Layer 4A | PoseConv3D (M=2) -> SlowFast cascade + fusion|
+----------------------------------------------------------+
|  Layer 4B | PoseConv3D (M=1) for individual actions      |
+----------------------------------------------------------+
|  Layer 5  | EMA smoothing + state machine + alert dedup |
+----------------------------------------------------------+
|  Layer 6  | Alert manager + PostgreSQL + clip writer    |
+----------------------------------------------------------+
|  Layer 7  | FastAPI + Prometheus + Grafana dashboard   |
+----------------------------------------------------------+
```

## Mock vs production mode

Every GPU-dependent component has a CPU mock fallback so the entire
pipeline can be tested without TensorRT, CUDA, or real cameras:

| Component   | Production            | Mock                          |
|-------------|-----------------------|-------------------------------|
| Ingestion   | FFmpeg + NVDEC        | `MockSource` synthetic frames |
| Detection   | YOLOv8n-Pose TRT      | `MockDetector` (deterministic)|
| Re-ID       | OSNet (torchreid)     | `MockReID` (histogram hash)   |
| PoseConv3D  | TensorRT engine       | `MockPoseConv3D` (motion heuristic) |
| SlowFast    | TensorRT engine       | `MockSlowFast` (motion heuristic) |
| Postgres    | PostgreSQL 16         | In-memory list                |

The global `CCTV_MOCK_MODE=true` flag (or per-layer `*_USE_MOCK=true`)
switches between them. Tests run in mock mode by default (see
`tests/conftest.py`).

## Threading model

The async pipeline (`Pipeline`) uses one thread per camera for ingestion,
plus two shared worker threads for detection and tracking. Inter-thread
communication is via Python `queue.Queue` (production would use Redis
Streams for cross-process / cross-node).

The sync pipeline (`SyncPipeline`) processes one frame at a time in the
caller's thread — useful for tests and offline evaluation.

## Performance budget

Per the original spec, the pipeline targets:
- 1.2 - 1.8 s end-to-end alert latency
- < 500 ms compute latency (frame capture → prediction)
- 6,400 FPS detection throughput per RTX 4090
- 16-64 clip batches for action recognition
- 3 s alert deduplication window

The CPU mock mode does NOT meet these targets — it's for functional
testing only. Production requires the GPU TensorRT engines.

## Detection stride trade-off

Detection runs at 6 FPS (every 5th frame) while tracking runs at 30 FPS.
The 4/5 frames between detections are filled by Kalman-predicted poses
from `PoseInterpolator`. This trades a small accuracy loss for an 5x
reduction in detection cost.

## Cascade filter

Stage 1 (PoseConv3D on skeletons) is cheap (~5 ms / batch of 16). Stage 2
(SlowFast on RGB clips) is expensive (~15-20 ms / batch of 16). The
cascade skips Stage 2 for any pair whose Stage 1 score is below threshold
(default 0.4) — saving 70-80% of the expensive inference in typical
scenes where most pairs are NOT interacting.

## State machine

Per-(camera, track, action) state machine transitions:
```
none -> candidate -> confirmed -> alert
                                   |
                                   v
        <---- reset (low score for N frames) ----+
```

Default thresholds (configurable in `config/settings.py`):
- candidate: score >= 0.5 for 5 frames
- confirmed: score >= 0.7 for 10 frames
- alert:     score >= 0.8 for 15 frames
- reset:     score < 0.3 for 5 frames

## Alert deduplication

Once an alert fires, subsequent alerts for the same `(camera_id, sorted
track_ids, action_label)` are suppressed for 3 seconds. This prevents
alert storms when the state machine oscillates around the alert threshold.

## Database schema

See `src/layer6_alerts/db_schema.sql`. Three tables:
- `alerts` — confirmed alerts with clip paths
- `cameras` — registered camera metadata
- `system_events` — generic event log (startup, errors, etc.)

## Scaling

For >120 cameras, deploy multiple inference nodes behind a load balancer.
Each node handles its own subset of cameras. The Redis Streams + Postgres
backends are shared across nodes.
