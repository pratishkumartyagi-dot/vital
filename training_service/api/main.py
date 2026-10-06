"""FastAPI service for PPG model training and inference."""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yaml
import json
import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.bpnet import BPNet, create_bpnet
from models.physformer import PhysFormer, create_physformer
from data.processor import PPGSignalProcessor, PPGSignalData, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PPG Vital Signs Training Service",
    description="API for training and deploying PPG-based vital signs models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config: Dict[str, Any] = {}
models: Dict[str, torch.nn.Module] = {}
processors: Dict[str, PPGSignalProcessor] = {}
training_status: Dict[str, Any] = {"status": "idle", "progress": 0, "message": ""}


class TrainingConfig(BaseModel):
    model_type: str = "bpnet"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


class ExportRequest(BaseModel):
    model_type: str = "bpnet"
    optimization: str = "DEFAULT"  # DEFAULT, FLOAT16, INT8


class InferenceRequest(BaseModel):
    model_type: str = "bpnet"
    signals: Dict[str, List[float]]  # red, green, blue
    sampling_rate: int = 30


class InferenceResponse(BaseModel):
    heart_rate: float
    spo2: float
    hemoglobin: float
    resp_rate: float
    bp_sys: float
    bp_dia: float


@app.on_event("startup")
async def startup_event():
    global config, processors
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    config = load_config(str(config_path))
    
    # Initialize processors
    processors['default'] = PPGSignalProcessor(config.get('training', {}))
    
    logger.info("Training service started")


@app.get("/")
async def root():
    return {
        "service": "PPG Vital Signs Training Service",
        "version": "1.0.0",
        "status": "running",
        "models_available": list(models.keys()),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")}


@app.get("/config")
async def get_config():
    return config


@app.post("/config")
async def update_config(new_config: Dict[str, Any]):
    global config
    config.update(new_config)
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return {"status": "updated", "config": config}


@app.post("/data/upload")
async def upload_signals(file: UploadFile = File(...)):
    """Upload raw PPG signals JSON file."""
    data_dir = Path(config.get('data', {}).get('raw_signals_dir', './data/raw_signals'))
    data_dir.mkdir(parents=True, exist_ok=True)
    
    content = await file.read()
    try:
        data = json.loads(content)
        # Validate structure
        required_keys = ['metadata', 'signals', 'vitals']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ppg_{timestamp}.json"
        filepath = data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {"status": "success", "filename": filename, "path": str(filepath)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/data/list")
async def list_data():
    """List available signal files."""
    data_dir = Path(config.get('data', {}).get('raw_signals_dir', './data/raw_signals'))
    if not data_dir.exists():
        return {"files": []}
    
    files = []
    for f in data_dir.glob("*.json"):
        stat = f.stat()
        files.append({
            "name": f.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    return {"files": sorted(files, key=lambda x: x["modified"], reverse=True)}


@app.delete("/data/{filename}")
async def delete_data(filename: str):
    """Delete a signal file."""
    data_dir = Path(config.get('data', {}).get('raw_signals_dir', './data/raw_signals'))
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    filepath.unlink()
    return {"status": "deleted", "filename": filename}


@app.post("/train/start")
async def start_training(background_tasks: BackgroundTasks, train_config: TrainingConfig):
    """Start model training in background."""
    global training_status
    
    if training_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    training_status = {"status": "running", "progress": 0, "message": "Starting training..."}
    
    # Update config with request params
    config['training']['epochs'] = train_config.epochs
    config['training']['batch_size'] = train_config.batch_size
    config['training']['learning_rate'] = train_config.learning_rate
    
    background_tasks.add_task(run_training, train_config.model_type)
    
    return {"status": "started", "model_type": train_config.model_type}


async def run_training(model_type: str):
    """Background training task."""
    global training_status, models
    
    try:
        from scripts.train import PPGTrainer
        
        trainer = PPGTrainer(config, model_type)
        
        # Monkey-patch to update progress
        original_train_epoch = trainer.train_epoch
        original_validate = trainer.validate
        
        def train_epoch_with_progress(loader):
            training_status["message"] = "Training..."
            return original_train_epoch(loader)
        
        def validate_with_progress(loader):
            training_status["message"] = "Validating..."
            return original_validate(loader)
        
        trainer.train_epoch = train_epoch_with_progress
        trainer.validate = validate_with_progress
        
        test_metrics = trainer.train()
        
        training_status = {
            "status": "completed",
            "progress": 100,
            "message": "Training completed",
            "metrics": test_metrics,
        }
        
        # Load best model into memory
        models_dir = Path(config.get('data', {}).get('models_dir', './models'))
        best_path = models_dir / f'{model_type}_best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location='cpu')
            model = trainer._create_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models[model_type] = model
            logger.info(f"Loaded best {model_type} model for inference")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_status = {
            "status": "failed",
            "progress": 0,
            "message": f"Training failed: {str(e)}",
        }


@app.get("/train/status")
async def get_training_status():
    return training_status


@app.post("/train/export")
async def export_model(request: ExportRequest):
    """Export trained model to TFLite."""
    global models
    
    model_type = request.model_type
    if model_type not in models:
        # Try to load from disk
        models_dir = Path(config.get('data', {}).get('models_dir', './models'))
        best_path = models_dir / f'{model_type}_best.pth'
        if not best_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found. Train first.")
        
        checkpoint = torch.load(best_path, map_location='cpu')
        if model_type == 'bpnet':
            model = create_bpnet(config)
        elif model_type == 'physformer':
            model = create_physformer(config)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models[model_type] = model
    
    model = models[model_type]
    
    # Export to TFLite
    from scripts.train import export_to_tflite, deploy_to_mobile
    
    models_dir = Path(config.get('data', {}).get('models_dir', './models'))
    tflite_path = models_dir / f'{model_type}.tflite'
    
    export_config = config.get('export', {})
    export_config['tflite_optimization'] = request.optimization
    config['export'] = export_config
    
    result = export_to_tflite(model, config, tflite_path)
    
    if result is None:
        raise HTTPException(status_code=500, detail="TFLite export failed")
    
    return {
        "status": "success",
        "model_type": model_type,
        "tflite_path": str(tflite_path),
        "size_kb": tflite_path.stat().st_size / 1024,
    }


@app.post("/train/deploy")
async def deploy_model(model_type: str = Form("bpnet"), mobile_assets: str = Form("../../vital_monitoring_app/assets")):
    """Deploy TFLite model to mobile app assets."""
    models_dir = Path(config.get('data', {}).get('models_dir', './models'))
    tflite_path = models_dir / f'{model_type}.tflite'
    
    if not tflite_path.exists():
        raise HTTPException(status_code=404, detail="TFLite model not found. Export first.")
    
    from scripts.train import deploy_to_mobile
    
    mobile_dir = Path(mobile_assets)
    success = deploy_to_mobile(tflite_path, mobile_dir)
    
    if not success:
        raise HTTPException(status_code=500, detail="Deployment failed")
    
    return {"status": "success", "deployed_to": str(mobile_dir / 'vitals_estimator.tflite')}


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Run inference on PPG signals."""
    global models
    
    model_type = request.model_type
    if model_type not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not loaded. Train or export first.")
    
    model = models[model_type]
    processor = processors['default']
    
    # Process signals
    red = np.array(request.signals.get('red', []), dtype=np.float32)
    green = np.array(request.signals.get('green', []), dtype=np.float32)
    blue = np.array(request.signals.get('blue', []), dtype=np.float32)
    
    if len(red) == 0 or len(green) == 0 or len(blue) == 0:
        raise HTTPException(status_code=400, detail="Missing signal channels")
    
    # Create PPGSignalData
    ppg_data = PPGSignalData(
        red=red,
        green=green,
        blue=blue,
        sampling_rate=request.sampling_rate,
        metadata={},
        vitals={},
        ground_truth={},
        quality={},
    )
    
    # Extract features
    features = processor.extract_features(ppg_data)
    segmented = features['segmented']
    
    # Use last window for inference
    if len(segmented) == 0:
        raise HTTPException(status_code=400, detail="Signal too short for inference")
    
    input_tensor = torch.from_numpy(segmented[-1:]).float()  # [1, 3, T]
    
    # Inference
    with torch.no_grad():
        predictions = model(input_tensor)
    
    return InferenceResponse(
        heart_rate=float(predictions['heart_rate'].item()),
        spo2=float(predictions['spo2'].item()),
        hemoglobin=float(predictions['hemoglobin'].item()),
        resp_rate=float(predictions['resp_rate'].item()),
        bp_sys=float(predictions['bp_sys'].item()),
        bp_dia=float(predictions['bp_dia'].item()),
    )


@app.get("/models/list")
async def list_models():
    """List available trained models."""
    models_dir = Path(config.get('data', {}).get('models_dir', './models'))
    if not models_dir.exists():
        return {"models": []}
    
    models_list = []
    for f in models_dir.glob("*_best.pth"):
        stat = f.stat()
        model_name = f.stem.replace('_best', '')
        models_list.append({
            "name": model_name,
            "type": "pytorch",
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    for f in models_dir.glob("*.tflite"):
        stat = f.stat()
        model_name = f.stem
        models_list.append({
            "name": model_name,
            "type": "tflite",
            "size_kb": stat.st_size / 1024,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    return {"models": models_list}


@app.get("/models/{model_name}/download")
async def download_model(model_name: str):
    """Download model file."""
    models_dir = Path(config.get('data', {}).get('models_dir', './models'))
    
    # Try both formats
    for ext in ['.tflite', '_best.pth']:
        filepath = models_dir / f"{model_name}{ext}"
        if filepath.exists():
            return FileResponse(
                filepath,
                media_type='application/octet-stream',
                filename=filepath.name
            )
    
    raise HTTPException(status_code=404, detail="Model not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)