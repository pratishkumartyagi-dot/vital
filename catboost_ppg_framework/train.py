"""
Training Script for CatBoost PPG Framework
==========================================

Trains CatBoost models for:
1. Blood Pressure (SBP, DBP) estimation
2. Blood Glucose (HbA1c) estimation  
3. Mental Stress/Workload classification

Supports multiple data formats including synthetic data, MIMIC-II, or custom CSV files.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from catboost_ppg import (
    PPGPreprocessor,
    PPGFeatureExtractor,
    CatBoostBPModel,
    CatBoostGlucoseModel,
    CatBoostStressModel,
    PreprocessingConfig
)


def generate_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic PPG training data for demonstration.
    This mimics the structure of real PPG features with associated targets.
    """
    np.random.seed(42)
    
    # Generate synthetic features with realistic ranges
    data = {
        # Morphology features
        'morph_num_peaks': np.random.normal(8.5, 1.5, n_samples),
        'morph_mean_peak_amplitude': np.random.normal(0.7, 0.15, n_samples),
        'morph_pulse_width': np.random.normal(0.7, 0.12, n_samples),
        'morph_rise_time': np.random.normal(0.3, 0.05, n_samples),
        'morph_fall_time': np.random.normal(0.4, 0.07, n_samples),
        'morph_systolic_area': np.random.normal(0.45, 0.1, n_samples),
        'morph_diastolic_area': np.random.normal(0.25, 0.08, n_samples),
        'morph_area_ratio': np.random.normal(1.8, 0.3, n_samples),
        'morph_p2p_amplitude': np.random.normal(0.85, 0.1, n_samples),
        'morph_max_dppg': np.random.normal(2.5, 0.5, n_samples),
        'morph_min_dppg': np.random.normal(-1.5, 0.3, n_samples),
        'morph_skewness': np.random.normal(0.1, 0.3, n_samples),
        'morph_kurtosis_val': np.random.normal(-0.8, 0.5, n_samples),
        
        # Frequency features
        'freq_dominant_freq': np.random.normal(1.2, 0.15, n_samples),
        'freq_total_power': np.random.normal(0.5, 0.12, n_samples),
        'freq_lf_power': np.random.normal(0.25, 0.08, n_samples),
        'freq_hf_power': np.random.normal(0.15, 0.05, n_samples),
        'freq_lf_hf_ratio': np.random.normal(1.7, 0.4, n_samples),
        'freq_spectral_entropy': np.random.normal(0.7, 0.08, n_samples),
        
        # HRV features
        'hrv_mean_ibi': np.random.normal(850, 100, n_samples),
        'hrv_sdnn': np.random.normal(45, 15, n_samples),
        'hrv_rmssd': np.random.normal(35, 12, n_samples),
        'hrv_pnn50': np.random.normal(15, 8, n_samples),
        'hrv_hr_mean': np.random.normal(72, 8, n_samples),
        'hrv_sd1': np.random.normal(25, 8, n_samples),
        'hrv_sd2': np.random.normal(55, 18, n_samples),
        'hrv_sd_ratio': np.random.normal(2.2, 0.4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate targets based on features (with realistic correlations)
    # BP: correlated with HR, stiffness (morph features), etc.
    df['sbp'] = 115 + 0.3 * (df['hrv_hr_mean'] - 72) + 8 * df['morph_pulse_width'] + np.random.normal(0, 5, n_samples)
    df['dbp'] = 75 + 0.15 * (df['hrv_hr_mean'] - 72) + 3 * df['morph_area_ratio'] + np.random.normal(0, 3, n_samples)
    
    # Glucose: weak correlations
    df['glucose'] = 95 + 5 * df['freq_lf_hf_ratio'] + 2 * df['hrv_sdnn'] / 10 + np.random.normal(0, 10, n_samples)
    
    # Stress: 3 levels based on HRV
    stress_scores = -2 * df['hrv_rmssd'] + 0.1 * df['hrv_hr_mean'] + np.random.normal(0, 15, n_samples)
    stress_labels = pd.cut(stress_scores, bins=3, labels=['low', 'medium', 'high'])
    df['stress'] = stress_labels
    
    return df


def train_models(data_path: str = None, output_dir: str = "models") -> None:
    """Train all CatBoost models."""
    print("=== CatBoost PPG Framework Training ===")
    
    # Load or generate training data
    if data_path and Path(data_path).exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Generating synthetic training data...")
        df = generate_synthetic_training_data(1000)
        # Save synthetic data
        syn_path = Path(output_dir) / "synthetic_training_data.csv"
        syn_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(syn_path, index=False)
        print(f"Saved synthetic data to {syn_path}")
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if not col in ['sbp', 'dbp', 'glucose', 'stress', 'hba1c']]
    features_df = df[feature_cols]
    
    # Initialize models
    bp_model = CatBoostBPModel(output_dir)
    glucose_model = CatBoostGlucoseModel(output_dir)
    stress_model = CatBoostStressModel(output_dir)
    
    # Train BP model
    print("\n1. Training Blood Pressure Model (SBP, DBP)...")
    bp_metrics = bp_model.train(
        features_df,
        df['sbp'].values,
        df['dbp'].values
    )
    bp_model.save_model()
    print(f"BP Model trained. SBP best score: {bp_metrics['sbp_best_score']}, DBP best score: {bp_metrics['dbp_best_score']}")
    
    # Train Glucose model
    print("\n2. Training Blood Glucose Model...")
    glucose_metrics = glucose_model.train(features_df, df['glucose'].values)
    glucose_model.save_model("catboost_glucose.cbm")
    print(f"Glucose Model trained. Best score: {glucose_metrics['best_score']}")
    
    # Train Stress model
    print("\n3. Training Stress Classification Model...")
    stress_metrics = stress_model.train(features_df, df['stress'].values)
    stress_model.save_model("catboost_stress.cbm")
    print(f"Stress Model trained. Best score: {stress_metrics['best_score']}")
    
    # Save training report
    report = {
        'bp_metrics': bp_metrics,
        'glucose_metrics': glucose_metrics,
        'stress_metrics': stress_metrics,
        'n_samples': len(df),
        'feature_count': len(feature_cols)
    }
    report_path = Path(output_dir) / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Models saved to: {output_dir}")
    print(f"Training report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost PPG Models")
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models', help='Output directory for models')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples if no data provided')
    
    args = parser.parse_args()
    train_models(args.data, args.output)


if __name__ == '__main__':
    main()