#!/usr/bin/env python3
"""
BPNet Training Pipeline for Cuff-less BP Estimation from PPG
Based on: Mohammadi et al., Sci Rep 2025 - CNN-BiLSTM-Attention architecture

Usage:
    python train_bpnet.py --data mimic_ppg --folds 5 --epochs 500
    python train_bpnet.py --data custom_ppg_csv --valid_fold 1 --test_fold 0
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

TRAINING_DIR = Path(__file__).parent
sys.path.insert(0, str(TRAINING_DIR))

# Import BPNet directly to avoid circular import via rppglib.models.__init__
import importlib.util
bpnet_path = TRAINING_DIR / 'rppglib' / 'models' / 'BPNet.py'
spec = importlib.util.spec_from_file_location("BPNet", bpnet_path)
bpnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpnet_module)
BPNet = bpnet_module.BPNet
BPNetTrainer = bpnet_module.BPNetTrainer
create_bpnet = bpnet_module.create_bpnet


class PPGBPDataset(torch.utils.data.Dataset):
    def __init__(self, ppg_signals, sbp_targets, dbp_targets, segment_length=1024):
        self.ppg_signals = ppg_signals
        self.sbp_targets = sbp_targets
        self.dbp_targets = dbp_targets
        self.segment_length = segment_length

    def __len__(self):
        return len(self.ppg_signals)

    def __getitem__(self, idx):
        ppg = self.ppg_signals[idx]
        sbp = self.sbp_targets[idx]
        dbp = self.dbp_targets[idx]

        if len(ppg) > self.segment_length:
            start = np.random.randint(0, len(ppg) - self.segment_length)
            ppg = ppg[start:start + self.segment_length]
        elif len(ppg) < self.segment_length:
            ppg = np.pad(ppg, (0, self.segment_length - len(ppg)), mode='constant')

        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)

        return (
            torch.FloatTensor(ppg).unsqueeze(0),
            torch.FloatTensor([sbp]),
            torch.FloatTensor([dbp])
        )


def load_mimic_ppg_data(data_path: str, segment_length=1024, fs=125):
    """
    Load MIMIC-II PPG-ABP dataset (preprocessed by Kachuee et al.)
    Expected format: .npz files with 'ppg' and 'abp' signals
    """
    data_path = Path(data_path)
    ppg_segments = []
    sbp_targets = []
    dbp_targets = []

    for npz_file in tqdm(sorted(data_path.glob('*.npz')), desc="Loading MIMIC data"):
        try:
            data = np.load(npz_file)
            ppg = data['ppg'].astype(np.float32)
            abp = data['abp'].astype(np.float32)

            if len(ppg) < segment_length or len(abp) < segment_length:
                continue

            num_segments = len(ppg) // segment_length
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                ppg_seg = ppg[start:end]
                abp_seg = abp[start:end]

                sbp = np.max(abp_seg)
                dbp = np.min(abp_seg)

                if 70 <= sbp <= 200 and 40 <= dbp <= 130 and sbp > dbp:
                    ppg_segments.append(ppg_seg)
                    sbp_targets.append(sbp)
                    dbp_targets.append(dbp)

        except Exception as e:
            print(f"Error loading {npz_file}: {e}")

    return np.array(ppg_segments), np.array(sbp_targets), np.array(dbp_targets)


def load_custom_ppg_csv(csv_path: str, ppg_col='ppg', sbp_col='sbp', dbp_col='dbp', segment_length=1024):
    """
    Load custom PPG-BP data from CSV.
    CSV should have columns: ppg (array or path), sbp, dbp
    """
    df = pd.read_csv(csv_path)
    ppg_segments = []
    sbp_targets = []
    dbp_targets = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading custom data"):
        try:
            if isinstance(row[ppg_col], str) and row[ppg_col].endswith('.npy'):
                ppg = np.load(row[ppg_col])
            else:
                ppg = np.array(eval(row[ppg_col]) if isinstance(row[ppg_col], str) else row[ppg_col])

            sbp = float(row[sbp_col])
            dbp = float(row[dbp_col])

            if len(ppg) < segment_length:
                ppg = np.pad(ppg, (0, segment_length - len(ppg)), mode='constant')
            elif len(ppg) > segment_length:
                ppg = ppg[:segment_length]

            ppg_segments.append(ppg.astype(np.float32))
            sbp_targets.append(sbp)
            dbp_targets.append(dbp)

        except Exception as e:
            print(f"Error processing row: {e}")

    return np.array(ppg_segments), np.array(sbp_targets), np.array(dbp_targets)


def train_bpnet(
    train_loader,
    val_loader,
    config,
    epochs=500,
    patience=30,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    results_folder='results/bpnet'
):
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    # Filter config for BPNet (remove training params)
    bpnet_config = {k: v for k, v in config.items() if k != 'lr'}
    model = create_bpnet(bpnet_config).to(device)
    trainer = BPNetTrainer(model, device=device, lr=config.get('lr', 0.001))

    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_mae_sbp': [], 'train_mae_dbp': [],
               'val_loss': [], 'val_mae_sbp': [], 'val_mae_dbp': []}

    print(f"\nStarting BPNet training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        train_losses = []
        train_mae_sbp = []
        train_mae_dbp = []

        for ppg, sbp, dbp in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            metrics = trainer.train_step(ppg, sbp, dbp)
            train_losses.append(metrics['loss'])
            train_mae_sbp.append(metrics['mae_sbp'])
            train_mae_dbp.append(metrics['mae_dbp'])

        val_metrics = trainer.validate(val_loader)

        history['train_loss'].append(np.mean(train_losses))
        history['train_mae_sbp'].append(np.mean(train_mae_sbp))
        history['train_mae_dbp'].append(np.mean(train_mae_dbp))
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_mae_sbp'].append(val_metrics['val_mae_sbp'])
        history['val_mae_dbp'].append(val_metrics['val_mae_dbp'])

        current_val_mae = (val_metrics['val_mae_sbp'] + val_metrics['val_mae_dbp']) / 2

        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            patience_counter = 0
            checkpoint_path = Path(results_folder) / 'bpnet_best.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
            print(f"  Epoch {epoch+1}: NEW BEST - Val MAE SBP: {val_metrics['val_mae_sbp']:.3f}, DBP: {val_metrics['val_mae_dbp']:.3f}")
        else:
            patience_counter += 1
            print(f"  Epoch {epoch+1}: Val MAE SBP: {val_metrics['val_mae_sbp']:.3f}, DBP: {val_metrics['val_mae_dbp']:.3f} (patience: {patience_counter}/{patience})")

        trainer.scheduler.step(val_metrics['val_loss'])

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model and history
    final_path = Path(results_folder) / 'bpnet_final.pth'
    trainer.save_checkpoint(final_path, epoch, val_metrics)

    with open(Path(results_folder) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val MAE: {best_val_mae:.3f}")
    print(f"Models saved to {results_folder}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='BPNet Training Pipeline for PPG-based BP Estimation')
    parser.add_argument('--data', type=str, required=True,
                       help='Data source: mimic_ppg, custom_csv, or path to data directory')
    parser.add_argument('--data_path', type=str, default='',
                       help='Path to data directory (for mimic_ppg) or CSV file (for custom_csv)')
    parser.add_argument('--csv_ppg_col', type=str, default='ppg',
                       help='Column name for PPG signal in custom CSV')
    parser.add_argument('--csv_sbp_col', type=str, default='sbp',
                       help='Column name for SBP target in custom CSV')
    parser.add_argument('--csv_dbp_col', type=str, default='dbp',
                       help='Column name for DBP target in custom CSV')
    parser.add_argument('--segment_length', type=int, default=1024,
                       help='Input segment length (samples)')
    parser.add_argument('--fs', type=int, default=125,
                       help='Sampling frequency (Hz)')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--valid_fold', type=int, default=1,
                       help='Validation fold index')
    parser.add_argument('--test_fold', type=int, default=0,
                       help='Test fold index')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--results_folder', type=str, default='results/bpnet',
                       help='Results output folder')
    parser.add_argument('--convert_tfjs', action='store_true',
                       help='Convert best model to TF.js format after training')

    args = parser.parse_args()

    config = {
        'input_length': args.segment_length,
        'cnn_filters': [32, 64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.2,
        'lr': args.lr
    }

    print("=== BPNet Training Pipeline ===")
    print(f"Config: {json.dumps(config, indent=2)}")

    if args.data == 'mimic_ppg':
        if not args.data_path:
            print("Error: --data_path required for mimic_ppg")
            sys.exit(1)
        ppg, sbp, dbp = load_mimic_ppg_data(args.data_path, args.segment_length, args.fs)
    elif args.data == 'custom_csv':
        if not args.data_path:
            print("Error: --data_path required for custom_csv")
            sys.exit(1)
        ppg, sbp, dbp = load_custom_ppg_csv(
            args.data_path, args.csv_ppg_col, args.csv_sbp_col, args.csv_dbp_col, args.segment_length
        )
    else:
        print(f"Unknown data source: {args.data}")
        sys.exit(1)

    print(f"Loaded {len(ppg)} samples")
    print(f"SBP range: {sbp.min():.1f}-{sbp.max():.1f}, DBP range: {dbp.min():.1f}-{dbp.max():.1f}")

    # K-fold split
    n = len(ppg)
    indices = np.random.permutation(n)
    fold_size = n // args.folds

    test_start = args.test_fold * fold_size
    test_end = test_start + fold_size
    test_idx = indices[test_start:test_end]

    remaining_idx = np.concatenate([indices[:test_start], indices[test_end:]])
    val_start = args.valid_fold * fold_size
    val_end = val_start + fold_size
    if val_end > len(remaining_idx):
        val_end = len(remaining_idx)
    val_idx = remaining_idx[val_start:val_end]
    train_idx = np.setdiff1d(remaining_idx, val_idx)

    train_dataset = PPGBPDataset(ppg[train_idx], sbp[train_idx], dbp[train_idx], args.segment_length)
    val_dataset = PPGBPDataset(ppg[val_idx], sbp[val_idx], dbp[val_idx], args.segment_length)
    test_dataset = PPGBPDataset(ppg[test_idx], sbp[test_idx], dbp[test_idx], args.segment_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model, history = train_bpnet(
        train_loader, val_loader, config,
        epochs=args.epochs, patience=args.patience,
        device=args.device, results_folder=args.results_folder
    )

    # Test evaluation
    print("\n=== Test Evaluation ===")
    trainer = BPNetTrainer(model, device=args.device)
    test_metrics = trainer.validate(test_loader)
    print(f"Test MAE - SBP: {test_metrics['val_mae_sbp']:.3f}, DBP: {test_metrics['val_mae_dbp']:.3f}")

    with open(Path(args.results_folder) / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    if args.convert_tfjs:
        print("\nConverting to TF.js format...")
        from convert_to_tfjs import convert_pytorch_to_tfjs
        convert_pytorch_to_tfjs(
            Path(args.results_folder) / 'bpnet_best.pth',
            Path(args.results_folder) / 'tfjs_model',
            config
        )


if __name__ == '__main__':
    main()