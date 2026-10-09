"""
CatBoost Fingertip PPG Framework
=================================

A modern, non-invasive machine learning framework for predicting vital physiological
metrics from fingertip Photoplethysmogram (PPG) signals using CatBoost gradient boosting.

Supports:
- Cuffless Blood Pressure (SBP, DBP) estimation
- Non-invasive Blood Glucose (HbA1c) monitoring
- Mental Stress & Cognitive Workload classification

References:
- IEEE Xplore: "CatBoost-based PPG Analysis for Vital Signs Monitoring" (2025)
- pyPPG Toolbox: Long-term continuous PPG time-series analysis
- CatBoost: Categorical Boosting for tabular data
"""

from .preprocessing import PPGPreprocessor, PreprocessingConfig, SignalQualityMetrics
from .features import PPGFeatureExtractor, PPGFeatures, MorphologyFeatures, FrequencyFeatures, HRVFeatures
from .models import CatBoostBPModel, CatBoostGlucoseModel, CatBoostStressModel
from .pipeline import (
    CatBoostPPGPipeline,
    PipelineConfig,
    PipelineResult,
    BPPrediction,
    GlucosePrediction,
    StressPrediction
)

__version__ = "1.0.0"
__author__ = "CatBoost PPG Framework Team"

__all__ = [
    "PPGPreprocessor",
    "PreprocessingConfig",
    "SignalQualityMetrics",
    "PPGFeatureExtractor",
    "PPGFeatures",
    "MorphologyFeatures",
    "FrequencyFeatures",
    "HRVFeatures",
    "CatBoostBPModel",
    "CatBoostGlucoseModel",
    "CatBoostStressModel",
    "CatBoostPPGPipeline",
    "PipelineConfig",
    "PipelineResult",
    "BPPrediction",
    "GlucosePrediction",
    "StressPrediction",
]