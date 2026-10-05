#!/usr/bin/env python3
"""
Correncoder Training Pipeline for PPG Respiratory Waveform Extraction
Based on: Davies & Mandic, "Rapid extraction of respiratory waveforms from PPG:
A deep corr-encoder approach", BSPC 2023.

Architecture: Convolutional autoencoder (correncoder) for PPG → Respiration waveform
- 3 Conv1D encoder layers (8 kernels each, kernels 150/75/50)
- 3 ConvTranspose1D decoder layers (8→8→1 kernels)
- Trained on Capnobase/BIDMC: simultaneous PPG + capnography/impedance pneumography
- LOSO (Leave-One-Subject-Out) cross-validation over 42 subjects

Usage:
    python train_correncoder.py --data capnobase --data_path /path/to/capnobase --folds 42
    python train_correncoder.py --data bidmc --data_path /path/to/bidmc --folds 53
    python train_correncoder.py --data custom --data_path my_data.npz --folds 5
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CorrencoderModel(nn.Module):
    def __init__(
        self,
        n_in=1,
        n_out=8,
        n_out2=8,
        n_out3=8,
        n_outputs=1,
        kernel_size=150,
        kernel_size2=75,
        kernel_size3=50,
        padding=20,
        padding2=20,
        padding3=10,
        dilation=1,
        dilation2=1,
        dilation3=1,
        stride=1,
        stride2=1,
        stride3=1,
        dropout=0.5
    ):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out, n_out2, kernel_size=kernel_size2, padding=padding2, dilation=dilation2, stride=stride2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out2, n_out3, kernel_size=kernel_size3, padding=padding3, dilation=dilation3, stride=stride3),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out3, n_out2, kernel_size=kernel_size3, padding=padding3, stride=stride3),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out2, n_out, kernel_size=kernel_size2, padding=padding2, stride=stride2),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out, n_outputs, kernel_size=kernel_size, padding=padding, stride=stride)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class CorrencoderTrainer:
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.001,
        weight_decay=1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        
    def train_step(self, ppg, resp_target):
        self.model.train()
        ppg = ppg.to(self.device)
        resp_target = resp_target.to(self.device)
        
        self.optimizer.zero_grad()
        resp_pred = self.model(ppg)
        
        loss = self.criterion(resp_pred, resp_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'mae': nn.functional.l1_loss(resp_pred, resp_target).item()
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        count = 0
        
        for ppg, resp_target in val_loader:
            ppg = ppg.to(self.device)
            resp_target = resp_target.to(self.device)
            
            resp_pred = self.model(ppg)
            
            loss = self.criterion(resp_pred, resp_target)
            total_loss += loss.item()
            total_mae += nn.functional.l1_loss(resp_pred, resp_target).item()
            count += 1
        
        return {
            'val_loss': total_loss / count,
            'val_mae': total_mae / count
        }
    
    def save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']


def breaths_per_min_zc(output_array, input_array):
    """Calculate breaths per minute error using zero crossings at 0.5 threshold."""
    peak_count_output = []
    peak_count_cap = []
    for ind in range(output_array.shape[0]):
        out_temp = output_array[ind, 0, :] - 0.5
        in_temp = input_array[ind, :] - 0.5
        zc_out = ((out_temp[:-1] * out_temp[1:]) < 0).sum()
        zc_in = ((in_temp[:-1] * in_temp[1:]) < 0).sum()
        peak_count_output.append(zc_out)
        peak_count_cap.append(zc_in)
    
    peak_count_output = np.array(peak_count_output)
    peak_count_cap = np.array(peak_count_cap)
    # 6.5 scaling factor for 60/6.5 second segments
    mean_error = ((np.mean(peak_count_output - peak_count_cap)) / 2) * 6.5
    mean_abs_error = ((np.mean(np.abs(peak_count_output - peak_count_cap))) / 2) * 6.5
    return mean_abs_error, mean_error


def load_capnobase_data(data_path: str, segment_length=128):
    """Load Capnobase dataset from .mat file (segments_all_standardised.mat)."""
    try:
        import mat73
    except ImportError:
        print("Installing mat73...")
        os.system("pip install mat73")
        import mat73
    
    data_path = Path(data_path)
    if data_path.is_dir():
        mat_file = data_path / 'segments_all_standardised.mat'
    else:
        mat_file = data_path
    
    print(f"Loading Capnobase data from {mat_file}")
    data = mat73.loadmat(str(mat_file))
    
    ppg_data = data['segs_ppg']  # Shape: (42, N_segments, segment_length)
    co2_data = data['segs_co2']  # Shape: (42, N_segments, segment_length)
    
    print(f"PPG shape: {ppg_data.shape}, CO2 shape: {co2_data.shape}")
    
    # Flatten subjects: (n_subjects * n_segments, segment_length)
    n_subjects = ppg_data.shape[0]
    ppg_flat = ppg_data.reshape(-1, segment_length)
    co2_flat = co2_data.reshape(-1, segment_length)
    
    # Subject indices for LOSO
    subject_ids = np.repeat(np.arange(n_subjects), ppg_data.shape[1])
    
    return ppg_flat.astype(np.float32), co2_flat.astype(np.float32), subject_ids


def load_bidmc_data(data_path: str, segment_length=128):
    """Load BIDMC dataset from .mat or preprocessed files."""
    # BIDMC has impedance pneumography as respiration reference
    # Implementation depends on data format
    raise NotImplementedError("BIDMC loading not yet implemented")


def load_custom_data(data_path: str, segment_length=128):
    """Load custom PPG-Respiration paired data from .npz file.
    Expected format: {'ppg': (N, L), 'resp': (N, L)}
    """
    data = np.load(data_path)
    ppg = data['ppg'].astype(np.float32)
    resp = data['resp'].astype(np.float32)
    
    if ppg.ndim == 2 and ppg.shape[1] > segment_length:
        ppg = ppg[:, :segment_length]
        resp = resp[:, :segment_length]
    elif ppg.ndim == 2 and ppg.shape[1] < segment_length:
        pad_width = segment_length - ppg.shape[1]
        ppg = np.pad(ppg, ((0, 0), (0, pad_width)), mode='constant')
        resp = np.pad(resp, ((0, 0), (0, pad_width)), mode='constant')
    
    return ppg, resp, None


def train_correncoder(
    train_loader,
    val_loader,
    config,
    epochs=80,
    patience=15,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    results_folder='results/correncoder'
):
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    model = CorrencoderModel(**config).to(device)
    trainer = CorrencoderTrainer(model, device=device, lr=config.get('lr', 0.001))

    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

    print(f"\nStarting Correncoder training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        train_losses = []
        train_maes = []

        for ppg, resp in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            metrics = trainer.train_step(ppg, resp)
            train_losses.append(metrics['loss'])
            train_maes.append(metrics['mae'])

        val_metrics = trainer.validate(val_loader)

        history['train_loss'].append(np.mean(train_losses))
        history['train_mae'].append(np.mean(train_maes))
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_mae'].append(val_metrics['val_mae'])

        if val_metrics['val_mae'] < best_val_mae:
            best_val_mae = val_metrics['val_mae']
            patience_counter = 0
            checkpoint_path = Path(results_folder) / 'correncoder_best.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
            print(f"  Epoch {epoch+1}: NEW BEST - Val MAE: {val_metrics['val_mae']:.6f}")
        else:
            patience_counter += 1
            print(f"  Epoch {epoch+1}: Val MAE: {val_metrics['val_mae']:.6f} (patience: {patience_counter}/{patience})")

        trainer.scheduler.step(val_metrics['val_loss'])

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    final_path = Path(results_folder) / 'correncoder_final.pth'
    trainer.save_checkpoint(final_path, epoch, val_metrics)

    with open(Path(results_folder) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val MAE: {best_val_mae:.6f}")
    print(f"Models saved to {results_folder}")

    return model, history


def evaluate_resp_rate(model, test_loader, device, fs=125):
    """Evaluate respiratory rate estimation accuracy on test set."""
    model.eval()
    all_rates = []
    all_targets = []
    
    with torch.no_grad():
        for ppg, resp in test_loader:
            ppg = ppg.to(device)
            resp_pred = model(ppg).cpu().numpy()
            resp_true = resp.cpu().numpy()
            
            # Estimate rate from zero crossings
            for i in range(resp_pred.shape[0]):
                pred_wave = resp_pred[i, 0] - 0.5
                true_wave = resp_true[i, 0] - 0.5
                zc_pred = ((pred_wave[:-1] * pred_wave[1:]) < 0).sum()
                zc_true = ((true_wave[:-1] * true_wave[1:]) < 0).sum()
                
                duration = resp_pred.shape[2] / fs
                rate_pred = (zc_pred / 2) / duration * 60
                rate_true = (zc_true / 2) / duration * 60
                
                if 5 <= rate_pred <= 50 and 5 <= rate_true <= 50:
                    all_rates.append(rate_pred)
                    all_targets.append(rate_true)
    
    if len(all_rates) > 0:
        all_rates = np.array(all_rates)
        all_targets = np.array(all_targets)
        mae = np.mean(np.abs(all_rates - all_targets))
        print(f"\nRespiratory Rate Test MAE: {mae:.3f} breaths/min")
        print(f"  Mean Pred: {np.mean(all_rates):.1f}, Mean True: {np.mean(all_targets):.1f}")
        return mae
    return None


def main():
    parser = argparse.ArgumentParser(description='Correncoder Training Pipeline for PPG Respiration')
    parser.add_argument('--data', type=str, required=True,
                       choices=['capnobase', 'bidmc', 'custom'],
                       help='Data source')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory or file')
    parser.add_argument('--segment_length', type=int, default=128,
                       help='Segment length (samples) - Capnobase uses 128 @ 125Hz = ~1s')
    parser.add_argument('--folds', type=int, default=42,
                       help='Number of CV folds (42 for Capnobase LOSO)')
    parser.add_argument('--test_fold', type=int, default=0,
                       help='Test fold index (0 to folds-1)')
    parser.add_argument('--valid_fold', type=int, default=1,
                       help='Validation fold index')
    parser.add_argument('--batch_size', type=int, default=30,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--results_folder', type=str, default='results/correncoder',
                       help='Results output folder')
    parser.add_argument('--seed', type=int, default=55,
                       help='Random seed')
    
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = {
        'n_in': 1,
        'n_out': 8,
        'n_out2': 8,
        'n_out3': 8,
        'n_outputs': 1,
        'kernel_size': 150,
        'kernel_size2': 75,
        'kernel_size3': 50,
        'padding': 20,
        'padding2': 20,
        'padding3': 10,
        'dilation': 1,
        'dilation2': 1,
        'dilation3': 1,
        'stride': 1,
        'stride2': 1,
        'stride3': 1,
        'dropout': args.dropout,
        'lr': args.lr
    }

    print("=== Correncoder Training Pipeline ===")
    print(f"Config: {json.dumps(config, indent=2)}")

    if args.data == 'capnobase':
        ppg, resp, subject_ids = load_capnobase_data(args.data_path, args.segment_length)
    elif args.data == 'bidmc':
        ppg, resp, subject_ids = load_bidmc_data(args.data_path, args.segment_length)
    elif args.data == 'custom':
        ppg, resp, subject_ids = load_custom_data(args.data_path, args.segment_length)
    else:
        print(f"Unknown data source: {args.data}")
        sys.exit(1)

    print(f"Loaded {len(ppg)} segments")
    print(f"PPG range: {ppg.min():.3f}-{ppg.max():.3f}, Resp range: {resp.min():.3f}-{resp.max():.3f}")

    # Reshape for Conv1D: (N, 1, L)
    ppg = ppg[:, np.newaxis, :]
    resp = resp[:, np.newaxis, :]

    # K-fold / LOSO split
    n = len(ppg)
    if subject_ids is not None:
        # LOSO: group by subject
        unique_subjects = np.unique(subject_ids)
        if args.folds > len(unique_subjects):
            args.folds = len(unique_subjects)
        
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_indices = list(kf.split(unique_subjects))
        
        test_subjects = unique_subjects[fold_indices[args.test_fold][1]]
        test_idx = np.where(np.isin(subject_ids, test_subjects))[0]
        
        remaining_subjects = unique_subjects[fold_indices[args.test_fold][0]]
        val_subjects = remaining_subjects[fold_indices[args.valid_fold % len(remaining_subjects)][1]]
        val_idx = np.where(np.isin(subject_ids, val_subjects))[0]
        
        train_subjects = np.setdiff1d(remaining_subjects, val_subjects)
        train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
    else:
        # Random split
        indices = np.random.permutation(n)
        fold_size = n // args.folds
        test_start = args.test_fold * fold_size
        test_end = test_start + fold_size
        test_idx = indices[test_start:test_end]
        
        remaining_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        val_start = args.valid_fold * fold_size
        val_end = min(val_start + fold_size, len(remaining_idx))
        val_idx = remaining_idx[val_start:val_end]
        train_idx = np.setdiff1d(remaining_idx, val_idx)

    train_dataset = TensorDataset(
        torch.FloatTensor(ppg[train_idx]),
        torch.FloatTensor(resp[train_idx])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(ppg[val_idx]),
        torch.FloatTensor(resp[val_idx])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(ppg[test_idx]),
        torch.FloatTensor(resp[test_idx])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model, history = train_correncoder(
        train_loader, val_loader, config,
        epochs=args.epochs, patience=args.patience,
        device=args.device, results_folder=args.results_folder
    )

    # Test evaluation
    print("\n=== Test Evaluation ===")
    trainer = CorrencoderTrainer(model, device=args.device)
    test_metrics = trainer.validate(test_loader)
    print(f"Test MSE: {test_metrics['val_loss']:.6f}, MAE: {test_metrics['val_mae']:.6f}")
    
    evaluate_resp_rate(model, test_loader, args.device, fs=125)

    with open(Path(args.results_folder) / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == '__main__':
    main()