# PPG Vital Signs Training Service

A complete pipeline for training, exporting, and deploying PPG-based vital signs estimation models.

## Overview

This service enables:
1. **Data Collection** - Export raw PPG signals (RGB channels) from mobile app
2. **Training** - Train BPNet or PhysFormer models on collected data
3. **Export** - Convert PyTorch models to TFLite for mobile deployment
4. **Deployment** - Automatically copy models to Flutter app assets

## Architecture

```
Mobile App (Flutter)          Training Service (Python)          Mobile App
┌─────────────────┐           ┌─────────────────────────┐        ┌─────────────────┐
│ Record PPG      │           │ Load JSON exports       │        │ Load TFLite     │
│ Signals         │──────────▶│ Preprocess & Augment    │        │ Model           │
│ Export JSON     │           │ Train BPNet/PhysFormer  │───────▶│ Run Inference   │
└─────────────────┘           │ Export to TFLite        │        └─────────────────┘
                              │ Deploy to assets        │
                              └─────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd training_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Training Data

1. Run the Flutter app on device
2. Take PPG measurements (rear camera for finger PPG, front for rPPG)
3. Enter ground truth values in Validation mode
4. Press **"Export Signals"** button (purple icon)
5. Share/save the JSON file to your computer
6. Place JSON files in `training_service/data/raw_signals/`

### 3. Train Model

```bash
# Train BPNet (recommended for mobile)
./scripts/deploy.sh bpnet ../../vital_monitoring_app/assets

# Or train PhysFormer (higher accuracy, larger model)
./scripts/deploy.sh physformer ../../vital_monitoring_app/assets
```

### 4. Rebuild Flutter App

```bash
cd ../../vital_monitoring_app
flutter clean
flutter pub get
flutter build apk --release
```

### 5. Test on Device

```bash
adb install build/app/outputs/flutter-apk/app-release.apk
```

## API Service (Optional)

Run the FastAPI service for remote training/inference:

```bash
cd training_service
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/data/upload` | POST | Upload signal JSON |
| `/data/list` | GET | List training files |
| `/train/start` | POST | Start background training |
| `/train/status` | GET | Check training progress |
| `/train/export` | POST | Export to TFLite |
| `/train/deploy` | POST | Deploy to mobile |
| `/inference` | POST | Run inference |
| `/models/list` | GET | List available models |
| `/models/{name}/download` | GET | Download model |

### Example: Train via API

```bash
# Upload data
curl -X POST "http://localhost:8000/data/upload" -F "file=@ppg_signals.json"

# Start training
curl -X POST "http://localhost:8000/train/start" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "bpnet", "epochs": 100, "batch_size": 32}'

# Check status
curl "http://localhost:8000/train/status"

# Export to TFLite
curl -X POST "http://localhost:8000/train/export" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "bpnet", "optimization": "DEFAULT"}'

# Deploy to mobile
curl -X POST "http://localhost:8000/train/deploy" \
  -F "model_type=bpnet" \
  -F "mobile_assets=../../vital_monitoring_app/assets"
```

## Model Architectures

### BPNet (Default)
- **Type**: Dilated residual CNN
- **Parameters**: ~200K
- **Input**: 3 channels × 300 samples (10s @ 30Hz)
- **Tasks**: HR, SpO2, Hb, RR, BP_sys, BP_dia
- **Mobile-friendly**: Yes (~500KB TFLite)

### PhysFormer
- **Type**: Vision Transformer for 1D signals
- **Parameters**: ~1.2M
- **Input**: 3 channels × 300 samples, patch size 16
- **Tasks**: Same as BPNet
- **Mobile-friendly**: Larger (~3MB TFLite)

## Data Format

JSON export from mobile app:

```json
{
  "metadata": {
    "timestamp": "2026-01-15T10:30:00.000Z",
    "sessionId": "session_1234567890",
    "patientId": "patient_2026-01-15",
    "cameraMode": "rear",
    "samplingRateHz": 30,
    "durationSeconds": 10.0
  },
  "signals": {
    "red": [100.5, 101.2, ...],
    "green": [98.3, 99.1, ...],
    "blue": [95.7, 96.2, ...]
  },
  "vitals": {
    "heartRate": 72.5,
    "spo2": 98.0,
    "hemoglobin": 14.2,
    "respiratoryRate": 16.0
  },
  "groundTruth": {
    "bpm": 72,
    "spo2": 98,
    "hb": 14.0,
    "bpSys": 120,
    "bpDia": 80,
    "glucose": 90,
    "respRate": 16
  },
  "quality": {
    "perfusionIndex": 0.52,
    "redPI": 0.48,
    "bluePI": 0.55,
    "bufferLength": 300,
    "state": "complete"
  }
}
```

## Configuration

Edit `config/training_config.yaml` to customize:

- **Training hyperparameters** (lr, batch size, epochs)
- **Model architecture** (hidden dim, layers, heads)
- **Signal processing** (filter bands, window size)
- **Export settings** (TFLite optimization, quantization)
- **Logging** (TensorBoard, Weights & Biases)

## Output Files

After training, `models/` directory contains:

```
models/
├── bpnet_best.p          # Best PyTorch checkpoint
├── bpnet_latest.pth      # Latest checkpoint
├── bpnet.tflite          # TFLite model for mobile
├── bpnet.onnx            # ONNX intermediate
├── bpnet_test_metrics.json  # Test set metrics
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- TensorFlow 2.15+ (for TFLite conversion)
- ONNX 1.15+ (for ONNX export)
- CUDA (optional, for GPU training)

## Troubleshooting

### TFLite Export Fails
```bash
# Install conversion dependencies
pip install onnx-tf tensorflow
```

### Out of Memory
Reduce batch size in config:
```yaml
training:
  batch_size: 16
```

### No Training Data
Ensure JSON files are in `data/raw_signals/` with correct structure.

### Mobile Model Not Loading
1. Check `pubspec.yaml` includes `assets/vitals_estimator.tflite`
2. Run `flutter clean && flutter pub get`
3. Verify TFLite model inputs/outputs match inference code

## License

Research use only. Not for clinical diagnosis.