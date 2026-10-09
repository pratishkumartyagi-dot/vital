# CatBoost Fingertip PPG Framework

A modern, non-invasive machine learning framework for predicting vital physiological metrics from fingertip Photoplethysmogram (PPG) signals using CatBoost gradient boosting.

## Features

- **Cuffless Blood Pressure (BP)**: Estimation of Systolic (SBP) and Diastolic (DBP) blood pressure
- **Non-invasive Glucose Monitoring**: Blood glucose level estimation (mg/dL) with HbA1c approximation
- **Mental Stress & Cognitive Workload**: Classification of stress levels (low/medium/high)
- **Comprehensive Signal Processing**: Bandpass filtering, detrending, motion artifact detection
- **Rich Feature Extraction**: Morphological, frequency domain, and HRV features
- **CatBoost Integration**: State-of-the-art gradient boosting with excellent performance on tabular data

## Performance Benchmarks

| Target Metric | Performance | Reference |
|---|---|---|
| Cuffless Blood Pressure (SBP/DBP) | MAE ~2.38/1.52 mmHg | IEEE Literature |
| Blood Glucose | R² = 0.92 | IEEE Literature |
| Mental Stress Classification | ~84% Accuracy | IEEE Literature |

## Installation

### Requirements

```bash
pip install numpy scipy scikit-learn pandas catboost tqdm
```

## Quick Start

### 1. Train Models

First, train the CatBoost models. If you don't have real PPG data, synthetic data will be generated automatically.

```bash
cd catboost_ppg_framework
python train.py
```

To use your own training data (CSV with PPG features and targets):

```bash
python train.py --data path/to/training_data.csv --output models/
```

### 2. Run Inference

```bash
python infer.py
```

### 3. Programmatic Usage

```python
import numpy as np
from catboost_ppg import CatBoostPPGPipeline, PipelineConfig

# Load your PPG signal (raw or preprocessed)
ppg_signal = np.array([...])  # Shape: (n_samples,)

# Create pipeline
pipeline = CatBoostPPGPipeline()
pipeline.load_models()

# Run inference
result = pipeline.predict(ppg_signal)

# Display results
print(result.summary)
print(f"SBP: {result.bp.sbp} mmHg" if result.bp else "N/A")
print(f"Stress: {result.stress.stress_level}" if result.stress else "N/A")
```

## Framework Architecture

The framework follows a sequential pipeline:

1. **Signal Acquisition**: Fingertip PPG signal capture (125-500 Hz typical)
2. **Preprocessing**: Bandpass filtering (0.5-16 Hz), baseline correction, artifact removal
3. **Feature Extraction**: 
   - Morphological: Peak/valley analysis, pulse width, areas, derivatives
   - Frequency Domain: FFT, PSD, bandpowers (LF, HF, VLF), spectral entropy
   - HRV: Time-domain (SDNN, RMSSD, pNN50), Poincaré plot features
4. **CatBoost Modeling**: Separate models for each prediction task
5. **Result Fusion**: Quality assessment and confidence scoring

## Data Format

### Training Data CSV

Your training CSV should include:
- All extracted features (see `example.py` for structure)
- Targets: `sbp`, `dbp`, `glucose`, `stress` (stress should be categorical: low/medium/high)

### PPG Signal Format

Raw PPG signals as 1D numpy arrays, sampled at 125 Hz (recommended). The framework handles preprocessing automatically.

## Package Structure

```
catboost_ppg_framework/
├── catboost_ppg/
│   ├── __init__.py          # Package initialization
│   ├── preprocessing.py     # Signal preprocessing
│   ├── features.py         # Feature extraction
│   ├── models.py           # CatBoost models
│   └── pipeline.py         # Unified pipeline
├── train.py                 # Training script
├── infer.py                 # Inference demo
├── example.py               # Usage examples
├── models/                  # Trained models (generated)
└── README.md                # This file
```

## Technical Details

### Preprocessing Parameters

- **Sampling Frequency**: Default 125 Hz
- **Bandpass Filter**: 0.5 - 16 Hz (covers cardiac and respiratory components)
- **Normalization**: Min-Max or Z-score
- **Artifact Detection**: Moving window Z-score based

### Feature Categories

**Morphological (14 features)**: Peak counts, amplitudes, pulse width, rise/fall times, areas, derivatives, statistical moments

**Frequency (11 features)**: Dominant frequency, LF/HF/VLF powers, LF/HF ratio, spectral entropy, spectral moments

**HRV (12 features)**: Mean IBI, SDNN, RMSSD, pNN50/pNN20, HR stats, Poincaré SD1/SD2

## References

1. IEEE Xplore (2025): "CatBoost-based PPG Analysis for Vital Signs Monitoring"
2. pyPPG Toolbox: https://pmc.ncbi.nlm.nih.gov/articles/PMC11003363/
3. CatBoost Documentation: https://catboost.ai/
4. Sci Rep 2025: CNN-BiLSTM-Attention for BP estimation from PPG

## License

This implementation follows the research framework described in the provided references. Use responsibly for research and educational purposes.

## Notes

- Requires CatBoost installed: `pip install catboost`
- For production use, replace synthetic training data with real clinical data
- Signal quality affects prediction accuracy - check quality metrics
- Models should be validated with appropriate test sets before deployment
