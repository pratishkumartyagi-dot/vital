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

from .preprocessing import PPGPreprocessor
from .features import PPGFeatureExtractor
from .models import CatBoostBPModel, CatBoostGlucoseModel, CatBoostStressModel
from .pipeline import CatBoostPPGPipeline

__version__ = "1.0.0"
__author__ = "CatBoost PPG Framework Team"

__all__ = [
    "PPGPreprocessor",
    "PPGFeatureExtractor", 
    "CatBoostBPModel",
    "CatBoostGlucoseModel",
    "CatBoostStressModel",
    "CatBoostPPGPipeline",
]