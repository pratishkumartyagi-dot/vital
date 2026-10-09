"""
Inference Script for CatBoost PPG Framework
==========================================

Example usage of the trained CatBoost PPG models for inference on PPG signals.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from catboost_ppg import CatBoostPPGPipeline, PipelineConfig


def generate_synthetic_ppg(duration_seconds: int = 60, fs: float = 125.0) -> np.ndarray:
    """
    Generate synthetic PPG signal for demonstration.
    Mimics a realistic fingertip PPG waveform.
    """
    t = np.arange(0, duration_seconds, 1/fs)
    # Base heart rate of ~75 BPM
    hr = 75
    freq = hr / 60.0
    
    # Generate PPG-like waveform (combination of sine waves + modulation)
    ppg = (
        np.sin(2 * np.pi * freq * t) +  # Cardiac component
        0.3 * np.sin(2 * np.pi * 2 * freq * t) +  # Second harmonic
        0.1 * np.sin(2 * np.pi * 0.1 * t) +  # Respiratory modulation
        0.05 * np.random.normal(0, 1, len(t))  # Noise
    )
    
    # Normalize to realistic amplitude
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min()) * 0.8 + 0.1
    return ppg.astype(np.float32)


def main():
    print("=== CatBoost PPG Framework Inference Demo ===")
    
    # Generate synthetic PPG signal
    print("Generating synthetic fingertip PPG signal (60 seconds, 125 Hz)...")
    ppg_signal = generate_synthetic_ppg(duration_seconds=60, fs=125.0)
    
    # Initialize pipeline (will try to load models)
    config = PipelineConfig(fs=125.0)
    pipeline = CatBoostPPGPipeline(config, model_dir="models")
    
    # Try to load pre-trained models
    try:
        pipeline.load_models()
        print("Pre-trained models loaded successfully")
    except Exception as e:
        print(f"Note: {e}")
        print("Using pipeline without pre-trained models (predictions may not be accurate)")
    
    # Run inference
    print("\nRunning inference...")
    result = pipeline.predict(ppg_signal)
    
    # Display results
    print(result.summary)
    
    if result.features:
        print(f"\nExtracted Features Summary:")
        print(f"  Morphology: {len(result.features.morphology.__dict__)} features")
        print(f"  Frequency: {len(result.features.frequency.__dict__)} features")
        print(f"  HRV: {len(result.features.hrv.__dict__)} features")
        print(f"  Total: {len(result.features_df.columns)} flattened features")
    
    print(f"\nProcessing time: {result.processing_time_ms:.2f} ms")


if __name__ == '__main__':
    main()