"""Training script for PPG vital signs models."""
import os
import sys
import argparse
import logging
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.processor import PPGDataset, load_config
from models.bpnet import BPNet, MultiTaskLoss, create_bpnet
from models.physformer import PhysFormer, create_physformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PPGTrainer:
    """Trainer for PPG vital signs models."""
    
    def __init__(self, config: Dict[str, Any], model_type: str = 'bpnet'):
        self.config = config
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Training config
        train_config = config.get('training', {})
        self.batch_size = train_config.get('batch_size', 32)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.epochs = train_config.get('epochs', 100)
        self.patience = train_config.get('early_stopping_patience', 10)
        self.weight_decay = train_config.get('weight_decay', 1e-4)
        
        # Model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Loss and optimizer
        self.tasks = ['heart_rate', 'spo2', 'hemoglobin', 'resp_rate', 'bp_sys', 'bp_dia']
        self.criterion = MultiTaskLoss(self.tasks).to(self.device)
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Logging
        self.log_dir = Path(config.get('logging', {}).get('tensorboard_dir', './logs/tensorboard'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            logger.warning("TensorBoard not available")
        
        # Model saving
        self.models_dir = Path(config.get('data', {}).get('models_dir', './models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
    
    def _create_model(self) -> nn.Module:
        """Create model based on type."""
        if self.model_type == 'bpnet':
            return create_bpnet(self.config)
        elif self.model_type == 'physformer':
            return create_physformer(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data loaders."""
        dataset = PPGDataset(self.config)
        X, y = dataset.load_all()
        
        if len(X) == 0:
            raise ValueError("No valid training data found!")
        
        splits = dataset.split_data(X, y)
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.from_numpy(splits['train']['X']),
            *[torch.from_numpy(splits['train']['y'][task]) for task in self.tasks]
        )
        val_dataset = TensorDataset(
            torch.from_numpy(splits['val']['X']),
            *[torch.from_numpy(splits['val']['y'][task]) for task in self.tasks]
        )
        test_dataset = TensorDataset(
            torch.from_numpy(splits['test']['X']),
            *[torch.from_numpy(splits['test']['y'][task]) for task in self.tasks]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.criterion.train()
        
        total_loss = 0
        task_losses = {task: 0.0 for task in self.tasks}
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            X = batch[0].to(self.device)
            targets = {task: batch[i+1].to(self.device) for i, task in enumerate(self.tasks)}
            
            self.optimizer.zero_grad()
            predictions = self.model(X)
            loss, losses = self.criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            for task, l in losses.items():
                task_losses[task] += l.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: l / num_batches for task, l in task_losses.items()}
        
        return {'total': avg_loss, **avg_task_losses}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        self.criterion.eval()
        
        total_loss = 0
        task_losses = {task: 0.0 for task in self.tasks}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                X = batch[0].to(self.device)
                targets = {task: batch[i+1].to(self.device) for i, task in enumerate(self.tasks)}
                
                predictions = self.model(X)
                loss, losses = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                for task, l in losses.items():
                    task_losses[task] += l.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: l / num_batches for task, l in task_losses.items()}
        
        return {'total': avg_loss, **avg_task_losses}
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test model and compute metrics."""
        self.model.eval()
        self.criterion.eval()
        
        all_preds = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                X = batch[0].to(self.device)
                targets = {task: batch[i+1].to(self.device) for i, task in enumerate(self.tasks)}
                
                predictions = self.model(X)
                
                for task in self.tasks:
                    all_preds[task].append(predictions[task].cpu().numpy())
                    all_targets[task].append(targets[task].cpu().numpy())
        
        # Concatenate
        for task in self.tasks:
            all_preds[task] = np.concatenate(all_preds[task])
            all_targets[task] = np.concatenate(all_targets[task])
        
        # Compute metrics
        metrics = {}
        for task in self.tasks:
            pred = all_preds[task]
            target = all_targets[task]
            
            mse = np.mean((pred - target) ** 2)
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(mse)
            
            # R-squared
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            metrics[f'{task}_mse'] = mse
            metrics[f'{task}_mae'] = mae
            metrics[f'{task}_rmse'] = rmse
            metrics[f'{task}_r2'] = r2
            
            logger.info(f"{task}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_type': self.model_type,
        }
        
        # Save latest
        latest_path = self.models_dir / f'{self.model_type}_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.models_dir / f'{self.model_type}_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.model_type}")
        logger.info(f"Model size: {self.model.get_model_size()}")
        
        train_loader, val_loader, test_loader = self.load_data()
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_metrics['total']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val Loss: {val_metrics['total']:.4f}")
            
            # Log to TensorBoard
            if self.writer:
                for key, val in train_metrics.items():
                    self.writer.add_scalar(f'train/{key}', val, epoch)
                for key, val in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', val, epoch)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Check for improvement
            val_loss = val_metrics['total']
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final test
        logger.info("\nRunning final test evaluation...")
        test_metrics = self.test(test_loader)
        
        if self.writer:
            for key, val in test_metrics.items():
                self.writer.add_scalar(f'test/{key}', val, epoch)
            self.writer.close()
        
        logger.info("Training completed!")
        return test_metrics


def export_to_tflite(model: nn.Module, config: Dict[str, Any], output_path: Path, input_shape: Tuple[int, ...] = (1, 3, 300)):
    """Export PyTorch model to TFLite."""
    logger.info(f"Exporting model to TFLite: {output_path}")
    
    model.eval()
    
    # Export to ONNX first
    onnx_path = output_path.with_suffix('.onnx')
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heart_rate', 'spo2', 'hemoglobin', 'resp_rate', 'bp_sys', 'bp_dia'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'heart_rate': {0: 'batch_size'},
            'spo2': {0: 'batch_size'},
            'hemoglobin': {0: 'batch_size'},
            'resp_rate': {0: 'batch_size'},
            'bp_sys': {0: 'batch_size'},
            'bp_dia': {0: 'batch_size'},
        }
    )
    logger.info(f"Exported ONNX to {onnx_path}")
    
    # Convert ONNX to TFLite using onnx-tf
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        saved_model_dir = output_path.parent / f"{output_path.stem}_savedmodel"
        tf_rep.export_graph(str(saved_model_dir))
        
        # Convert to TFLite
        import tensorflow as tf
        
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        
        # Optimization
        export_config = config.get('export', {})
        optimization = export_config.get('tflite_optimization', 'DEFAULT')
        
        if optimization == 'FLOAT16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif optimization == 'INT8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Need representative dataset for INT8
            logger.warning("INT8 quantization requires representative dataset")
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"Exported TFLite to {output_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.1f} KB")
        
        return output_path
        
    except ImportError as e:
        logger.error(f"TFLite conversion failed: {e}")
        logger.info("Install onnx-tf and tensorflow for TFLite export")
        return None


def deploy_to_mobile(tflite_path: Path, mobile_assets_dir: Path):
    """Copy TFLite model to mobile app assets."""
    if not tflite_path.exists():
        logger.error(f"TFLite model not found: {tflite_path}")
        return False
    
    mobile_assets_dir.mkdir(parents=True, exist_ok=True)
    dest_path = mobile_assets_dir / 'vitals_estimator.tflite'
    
    import shutil
    shutil.copy2(tflite_path, dest_path)
    logger.info(f"Deployed model to {dest_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Train PPG vital signs model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='bpnet', choices=['bpnet', 'physformer'], help='Model type')
    parser.add_argument('--export-tflite', action='store_true', help='Export to TFLite after training')
    parser.add_argument('--deploy-mobile', action='store_true', help='Deploy to mobile assets')
    parser.add_argument('--mobile-assets', type=str, default='../../vital_monitoring_app/assets', help='Mobile assets directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    trainer = PPGTrainer(config, args.model)
    test_metrics = trainer.train()
    
    # Export to TFLite
    if args.export_tflite:
        best_model_path = Path(config['data']['models_dir']) / f'{args.model}_best.pth'
        checkpoint = torch.load(best_model_path, map_location='cpu')
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        tflite_path = Path(config['data']['models_dir']) / f'{args.model}.tflite'
        export_to_tflite(trainer.model, config, tflite_path)
        
        # Deploy to mobile
        if args.deploy_mobile:
            mobile_dir = Path(args.mobile_assets)
            deploy_to_mobile(tflite_path, mobile_dir)
    
    # Save test metrics
    metrics_path = Path(config['data']['models_dir']) / f'{args.model}_test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()