# CCTV Interaction Recognition System

A production-grade, multi-layer pipeline for real-time detection and
recognition of human interactions and individual actions from CCTV
camera streams. Designed for 100–120 RTSP cameras at 30 FPS with
sub-1.8-second end-to-end alert latency.

## Architecture (8 layers)

| Layer | Purpose | Key Tech |
|------|---------|----------|
| 0 | Video Ingestion & Decode | FFmpeg + NVIDIA NVDEC |
| 1 | Detection & Pose Estimation | YOLOv8n-Pose TensorRT FP16 |
| 2 | Tracking & Re-ID | ByteTrack + OSNet + Kalman |
| 3 | Pair Analysis & Routing | Distance matrix + face-to-face check |
| 4A | Interaction Recognition | PoseConv3D (cascade) + SlowFast (fusion) |
| 4B | Individual Action Recognition | PoseConv3D (M=1) |
| 5 | Post-Processing | EMA + State Machine + Dedup |
| 6 | Alert Management | PostgreSQL + FFmpeg clip writer |
| 7 | Monitoring & Dashboard | FastAPI + Prometheus + Grafana |

## Project Structure

```
cctv_interaction_system/
├── config/                  # Settings, camera list, Prometheus config
│   ├── settings.py          # All tunable parameters
│   ├── cameras.yaml         # Camera list (RTSP URLs)
│   └── prometheus.yml
├── src/
│   ├── common/              # Shared: schemas, logger, metrics, redis, db
│   ├── layer0_ingestion/    # FFmpeg pipeline, frame router, ingestion worker
│   ├── layer1_detection/    # TensorRT detector, batch collector, NMS
│   ├── layer2_tracking/     # ByteTrack, OSNet, Kalman, skeleton buffer
│   ├── layer3_pair_analysis/# Distance matrix, pair analyzer, router
│   ├── layer4a_interaction/ # PoseConv3D, SlowFast, cascade, ROI, fusion
│   ├── layer4b_individual/  # Skeleton preprocess, individual recognizer
│   ├── layer5_postprocess/  # EMA, state machine, alert dedup
│   ├── layer6_alerts/       # DB schema, clip writer, alert manager
│   ├── layer7_monitoring/   # FastAPI app, Prometheus endpoint
│   └── pipeline/            # End-to-end orchestrator (Pipeline + SyncPipeline)
├── tests/                   # Pytest suite (all modules + e2e)
├── scripts/                 # CLI entry points
│   ├── run_pipeline.py      # Start the full pipeline
│   ├── init_db.py           # Initialize PostgreSQL schema
│   ├── export_models.py     # Export TensorRT engines (GPU required)
│   └── simulate_cameras.py  # Generate synthetic RTSP streams
├── deploy/docker/           # Dockerfiles (CPU base + GPU base + API)
├── grafana/                 # Dashboard provisioning + sample dashboard
├── docker-compose.yml       # Full multi-service deployment
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Install dependencies (CPU / mock mode)

```bash
cd cctv_interaction_system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run end-to-end in mock mode (no GPU / no cameras needed)

```bash
python -m scripts.run_pipeline \
    --mock \
    --cameras config/cameras.yaml \
    --api --api-port 8000
```

This starts:
- 5 mock ingestion workers (synthetic BGR frames)
- Detection / tracking / recognition loops
- Alert manager with in-memory store
- FastAPI server on http://localhost:8000

Endpoints:
- `GET /health` — health check
- `GET /metrics` — Prometheus metrics
- `GET /alerts` — recent alerts
- `GET /cameras` — configured cameras
- `WS  /ws/alerts` — real-time alert push

### 3. Run with real RTSP cameras

Edit `config/cameras.yaml` with your RTSP URLs, then:

```bash
python -m scripts.run_pipeline --rtsp --cameras config/cameras.yaml --api
```

### 4. Run with Docker Compose (mock mode)

```bash
cp .env.example .env
docker compose --profile mock up --build
```

### 5. Run with Docker Compose (GPU production)

```bash
docker compose --profile gpu up --build
```

### 6. Run tests

```bash
pytest -v
```

## Production GPU Deployment

### TensorRT engine export

```bash
# Requires GPU + ultralytics + tensorrt
python -m scripts.export_models --model yolov8n-pose
python -m scripts.export_models --model slowfast
python -m scripts.export_models --model poseconv3d-pair
python -m scripts.export_models --model poseconv3d-individual
```

Place the resulting `.engine` files in `models/` and set the paths
in `.env`:

```env
CCTV_LAYER1__MODEL_PATH=models/yolov8n-pose.engine
CCTV_LAYER1__USE_MOCK=false
CCTV_LAYER4A__SLOWFAST_ENGINE_PATH=models/slowfast.engine
CCTV_LAYER4A__POSECONV3D_ENGINE_PATH=models/poseconv3d_pair.engine
CCTV_LAYER4A__USE_MOCK=false
CCTV_LAYER4B__POSECONV3D_ENGINE_PATH=models/poseconv3d_individual.engine
CCTV_LAYER4B__USE_MOCK=false
CCTV_MOCK_MODE=false
```

### Database

```bash
# Start PostgreSQL (docker compose handles this)
docker compose up -d postgres

# Initialize schema
python -m scripts.init_db
```

### Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- FastAPI: http://localhost:8000

## Configuration

All tunable parameters live in `config/settings.py` and can be overridden
via environment variables (prefix `CCTV_`, double underscore for nested
fields — e.g. `CCTV_LAYER1__MAX_BATCH=64`). See `.env.example` for the
full list.

## Performance Targets

| Metric | Target |
|--------|--------|
| End-to-end alert latency | 1.2 – 1.8 s |
| Compute latency (frame → prediction) | < 500 ms |
| Detection throughput | 6,400 FPS per RTX 4090 |
| Action recognition batch | 16–64 clips per forward pass |
| Alert deduplication window | 3 s per (camera, pair, action) |
| System uptime | 99.9% |

## Detected Actions

**Interactions** (2-person): `hug`, `kiss`, `fight`, `push`, `handshake`,
`high-five`, `other`, `none`

**Individual** (1-person): `walking`, `standing`, `running`, `sitting`,
`waiting`, `other`, `none`

## Testing

The test suite exercises every layer with deterministic mocks so it runs
without a GPU or real cameras:

```bash
pytest -v --tb=short
```

For coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## License

MIT
