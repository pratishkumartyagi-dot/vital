"""
Example Usage - CatBoost Fingertip PPG Framework
===============================================

This example demonstrates how to use the framework to analyze fingertip PPG signals
for blood pressure, glucose, and mental stress prediction.
"""

import numpy as np
from catboost_ppg import CatBoostPPGPipeline, PipelineConfig


def example_basic_usage():
    """Basic usage example."""
    # Load or generate PPG signal
    # In practice, this would come from your sensor/camera
    ppg = np.random.randn(125 * 30)  # 30 seconds at 125 Hz
    
    # Create pipeline
    pipeline = CatBoostPPGPipeline()
    
    # Load pre-trained models (if available)
    try:
        pipeline.load_models()
    except:
        print("No pre-trained models found. Train them first with train.py")
    
    # Predict
    result = pipeline.predict(ppg)
    
    # Print results
    print(result.summary)


def example_with_custom_config():
    """Example with custom configuration."""
    config = PipelineConfig(
        fs=125.0,
        use_segmentation=True,
        segment_length=10.0
    )
    pipeline = CatBoostPPGPipeline(config)
    
    # Generate synthetic signal
    t = np.linspace(0, 60, 60 * 125)
    ppg = np.sin(2 * np.pi * 1.25 * t) + 0.2 * np.random.randn(len(t))
    
    result = pipeline.predict(ppg)
    
    if result.bp:
        print(f"SBP={result.bp.sbp:.1f}, DBP={result.bp.dbp:.1f}")
    if result.stress:
        print(f"Stress level: {result.stress.stress_level}")


if __name__ == "__main__":
    example_basic_usage()