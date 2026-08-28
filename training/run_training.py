#!/usr/bin/env python3
"""
MCD-rPPG Training Pipeline
Trains SCNN_8rois model for remote photoplethysmography (rPPG) using the MCD-rPPG dataset.

Usage:
    python run_training.py --model SCNN_8rois --dataset mcd_rppg --folds 5
    python run_training.py --model SCNN_8rois --dataset mcd_rppg --valid_fold 1 --test_fold 0
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add training directory to path
TRAINING_DIR = Path(__file__).parent
sys.path.insert(0, str(TRAINING_DIR))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import rppglib
from rppglib.train import train_fold, calc_metrics
from rppglib.params import get_config


def prepare_dataset_metadata(dataset_path, dataset_name, num_folds=5):
    """Generate metadata CSV for preprocessed dataset files."""
    dataset_path = Path(dataset_path)
    cache_folder = dataset_path / 'cache'
    cache_folder.mkdir(exist_ok=True)
    
    files = sorted(list(cache_folder.glob('*_vid.npy')))
    if not files:
        print(f"No preprocessed files found in {cache_folder}")
        print("Run preprocessing first using preprocessing scripts.")
        return None
    
    records = []
    for f in files:
        stem = f.stem.replace('_vid', '')
        ppg_file = cache_folder / f"{stem}_ppg.npy"
        if ppg_file.exists():
            fold = hash(stem) % num_folds
            records.append({
                'file': str(f),
                'ppg_file': str(ppg_file),
                'fold': fold,
                'subject': stem.split('_')[1] if '_' in stem else stem
            })
    
    df = pd.DataFrame(records)
    csv_path = dataset_path / f'{dataset_name}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path} ({len(df)} samples)")
    return csv_path


def run_training(args):
    """Execute training pipeline."""
    config = get_config()
    
    # Override config with args
    if args.model:
        config.model = args.model
    if args.dataset:
        config.train_dataset = args.dataset
    if args.valid_fold is not None:
        config.valid_fold = args.valid_fold
    if args.test_fold is not None:
        config.test_fold = args.test_fold
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.results_folder:
        config.results_folder = args.results_folder
    if args.test_datasets:
        config.test_datasets = args.test_datasets
    
    # Create results directory
    Path(config.results_folder).mkdir(exist_ok=True)
    
    print(f"=== MCD-rPPG Training Pipeline ===")
    print(f"Model: {config.model}")
    print(f"Dataset: {config.train_dataset}")
    print(f"Valid fold: {config.valid_fold}")
    print(f"Test fold: {config.test_fold}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print()
    
    # Check if metadata exists
    metadata_path = f'{config.train_dataset}.csv'
    if not os.path.exists(metadata_path):
        print(f"Metadata file {metadata_path} not found.")
        print("Creating sample metadata for demo purposes...")
        create_sample_metadata(config)
    
    # Run training
    try:
        config = train_fold(config)
        
        # Save results
        results_path = Path(config.results_folder) / f'{config.model}_results.json'
        with open(results_path, 'w') as f:
            json.dump(config.test_results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        return config.test_results
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_sample_metadata(config):
    """Create sample metadata CSV for demonstration."""
    os.makedirs('results', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    
    num_samples = 10
    files = []
    for i in range(num_samples):
        video = np.random.randint(0, 255, (600, 64, 64, 3), dtype=np.uint8)
        ppg = np.random.randn(600).astype(np.float32)
        landmarks = np.random.randint(0, 64, (600, 68, 2), dtype=np.int16)
        np.savez(f'cache/sample_{i}.npz', video=video, landmarks=landmarks, ppg=ppg)
        files.append(f'cache/sample_{i}.npz')
    
    df = pd.DataFrame({
        'file': files,
        'fold': [i % 5 for i in range(num_samples)]
    })
    df.to_csv(f'{config.train_dataset}.csv', index=False)
    print(f"Created sample metadata with {num_samples} samples")


def main():
    parser = argparse.ArgumentParser(description='MCD-rPPG Training Pipeline')
    parser.add_argument('--model', type=str, default='SCNN_8rois',
                       help='Model to train (SCNN_8rois, POS, PBV, OMIT, etc.)')
    parser.add_argument('--dataset', type=str, default='mcd_rppg',
                       help='Dataset name (mcd_rppg, MMPD, SCAMPS, UBFC_rPPG)')
    parser.add_argument('--valid_fold', type=int, default=1,
                       help='Validation fold index')
    parser.add_argument('--test_fold', type=int, default=0,
                       help='Test fold index')
    parser.add_argument('--batch_size', type=int, default=36,
                       help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--results_folder', type=str, default='results',
                       help='Folder to save results')
    parser.add_argument('--test_datasets', type=str, nargs='+', default=['mcd_rppg'],
                       help='Datasets to test on')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    results = run_training(args)
    
    if results:
        print("\n=== Training Complete ===")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
    else:
        print("\nTraining did not complete successfully.")
        sys.exit(1)


if __name__ == '__main__':
    main()
