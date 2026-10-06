#!/bin/bash
# Automated training and deployment script for PPG vital signs models
# Usage: ./deploy.sh [model_type] [mobile_assets_path]

set -e

MODEL_TYPE=${1:-bpnet}
MOBILE_ASSETS_PATH=${2:-../../vital_monitoring_app/assets}
CONFIG_PATH="config/training_config.yaml"
TRAINING_SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PPG Model Training & Deployment ==="
echo "Model type: $MODEL_TYPE"
echo "Mobile assets: $MOBILE_ASSETS_PATH"
echo "Config: $CONFIG_PATH"
echo ""

cd "$TRAINING_SERVICE_DIR"

# Check if Python environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for training data
DATA_DIR="data/raw_signals"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR)" ]; then
    echo "WARNING: No training data found in $DATA_DIR"
    echo "Please export signals from mobile app first."
    echo "Files should be JSON exports from the app's 'Export Signals' button."
    exit 1
fi

echo "Found $(ls -1 $DATA_DIR/*.json | wc -l) signal files"

# Train model
echo ""
echo "=== Training $MODEL_TYPE ==="
python scripts/train.py --config $CONFIG_PATH --model $MODEL_TYPE --export-tflite --deploy-mobile --mobile-assets $MOBILE_ASSETS_PATH

# Verify deployment
TFLITE_PATH="models/${MODEL_TYPE}.tflite"
MOBILE_MODEL="$MOBILE_ASSETS_PATH/vitals_estimator.tflite"

if [ -f "$TFLITE_PATH" ]; then
    echo ""
    echo "=== TFLite Model ==="
    echo "Path: $TFLITE_PATH"
    echo "Size: $(du -h $TFLITE_PATH | cut -f1)"
    
    # Check model info
    python3 -c "
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='$TFLITE_PATH')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('Input shape:', input_details[0]['shape'])
print('Input dtype:', input_details[0]['dtype'])
for i, out in enumerate(output_details):
    print(f'Output {i}: {out[\"name\"]} shape={out[\"shape\"]} dtype={out[\"dtype\"]}')
"
else
    echo "ERROR: TFLite model not generated!"
    exit 1
fi

if [ -f "$MOBILE_MODEL" ]; then
    echo ""
    echo "=== Mobile Deployment ==="
    echo "Model deployed to: $MOBILE_MODEL"
    echo "Size: $(du -h $MOBILE_MODEL | cut -f1)"
else
    echo "WARNING: Model not found in mobile assets!"
fi

echo ""
echo "=== Training Metrics ==="
METRICS_FILE="models/${MODEL_TYPE}_test_metrics.json"
if [ -f "$METRICS_FILE" ]; then
    python3 -c "
import json
with open('$METRICS_FILE') as f:
    m = json.load(f)
for k, v in m.items():
    print(f'{k}: {v:.4f}')
"
else
    echo "No metrics file found"
fi

echo ""
echo "=== Deployment Complete ==="
echo "Next steps:"
echo "1. Rebuild Flutter app: cd ../../vital_monitoring_app && flutter build apk"
echo "2. Install on device: adb install build/app/outputs/flutter-apk/app-release.apk"
echo "3. Test with live PPG measurements"