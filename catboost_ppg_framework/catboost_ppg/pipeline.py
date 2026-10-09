"""
Unified Pipeline Module
======================

Implements the complete CatBoost Fingertip PPG Framework pipeline for end-to-end
processing: signal acquisition → preprocessing → feature extraction → ML prediction.

Supports simultaneous prediction of:
- Blood Pressure (SBP, DBP)
- Blood Glucose (HbA1c)  
- Mental Stress/Workload level
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path

from .preprocessing import PPGPreprocessor, PreprocessingConfig, SignalQualityMetrics
from .features import PPGFeatureExtractor, PPGFeatures
from .models import CatBoostBPModel, CatBoostGlucoseModel, CatBoostStressModel


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    fs: float = 125.0
    preprocessing: PreprocessingConfig = None
    segment_length: float = 30.0  # Use 30-second segments for better stability
    use_segmentation: bool = False
    auto_save_features: bool = False
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig(fs=self.fs)


@dataclass
class BPPrediction:
    """Blood pressure prediction results."""
    sbp: float  # Systolic BP (mmHg)
    dbp: float  # Diastolic BP (mmHg)
    map: float  # Mean Arterial Pressure
    pulse_pressure: float
    confidence: float = 0.85
    quality_passed: bool = True
    
    @property
    def summary(self) -> str:
        return f"BP: {self.sbp:.1f}/{self.dbp:.1f} mmHg, MAP: {self.map:.1f} mmHg"


@dataclass
class GlucosePrediction:
    """Blood glucose prediction results."""
    glucose: float  # Glucose level (mg/dL)
    hba1c_estimate: float  # Estimated HbA1c (%)
    confidence: float = 0.75
    quality_passed: bool = True
    
    @property
    def summary(self) -> str:
        return f"Glucose: {self.glucose:.1f} mg/dL (HbA1c est: {self.hba1c_estimate:.1f}%)"


@dataclass
class StressPrediction:
    """Mental stress prediction results."""
    stress_level: str  # 'low', 'medium', 'high'
    stress_score: float  # Numeric score (0-100)
    confidence: float = 0.70
    probabilities: Dict[str, float] = None
    quality_passed: bool = True
    
    @property
    def summary(self) -> str:
        return f"Stress: {self.stress_level} (score: {self.stress_score:.1f})"


@dataclass
class PipelineResult:
    """Complete pipeline prediction results."""
    bp: Optional[BPPrediction]
    glucose: Optional[GlucosePrediction]
    stress: Optional[StressPrediction]
    signal_quality: SignalQualityMetrics
    features: PPGFeatures
    features_df: pd.DataFrame
    processing_time_ms: float = 0.0
    
    @property
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CatBoost PPG Framework Results",
            "=" * 60,
            f"Signal Quality: {self.signal_quality.overall_quality:.1f}% (Acceptable: {self.signal_quality.is_acceptable})",
        ]
        if self.bp:
            lines.append(f"Blood Pressure: {self.bp.summary}")
        if self.glucose:
            lines.append(f"Glucose: {self.glucose.summary}")
        if self.stress:
            lines.append(f"Stress: {self.stress.summary}")
        lines.append("=" * 60)
        return "\n".join(lines)


class CatBoostPPGPipeline:
    """
    Unified pipeline for CatBoost-based PPG vital signs prediction.
    """
    
    def __init__(self,
                 config: Optional[PipelineConfig] = None,
                 bp_model: Optional[CatBoostBPModel] = None,
                 glucose_model: Optional[CatBoostGlucoseModel] = None,
                 stress_model: Optional[CatBoostStressModel] = None,
                 model_dir: str = "models"):
        self.config = config or PipelineConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = PPGPreprocessor(self.config.preprocessing)
        self.feature_extractor = PPGFeatureExtractor(self.config.fs)
        
        # Initialize or load models
        self.bp_model = bp_model or CatBoostBPModel(str(self.model_dir))
        self.glucose_model = glucose_model or CatBoostGlucoseModel(str(self.model_dir))
        self.stress_model = stress_model or CatBoostStressModel(str(self.model_dir))
        
        self.models_loaded = False
    
    def load_models(self,
                   bp_model_file: str = "catboost_bp.cbm",
                   glucose_model_file: str = "catboost_glucose.cbm",
                   stress_model_file: str = "catboost_stress.cbm") -> None:
        """Load pre-trained models."""
        try:
            self.bp_model.load_model(bp_model_file)
        except Exception as e:
            print(f"Warning: Could not load BP model: {e}")
        
        try:
            self.glucose_model.load_model(glucose_model_file)
        except Exception as e:
            print(f"Warning: Could not load Glucose model: {e}")
        
        try:
            self.stress_model.load_model(stress_model_file)
        except Exception as e:
            print(f"Warning: Could not load Stress model: {e}")
        
        self.models_loaded = (self.bp_model.is_trained or 
                             self.glucose_model.is_trained or 
                             self.stress_model.is_trained)
    
    def predict(self, ppg_signal: np.ndarray) -> PipelineResult:
        """
        Run complete pipeline prediction.
        
        Args:
            ppg_signal: Raw PPG signal as numpy array
            
        Returns:
            PipelineResult with all predictions
        """
        import time
        start_time = time.time()
        
        # 1. Preprocessing
        processed_signal, signal_quality = self.preprocessor.preprocess(
            ppg_signal, 
            normalize='minmax'
        )
        
        # 2. Feature extraction
        if self.config.use_segmentation and len(processed_signal) > self.config.segment_length * self.config.fs:
            segments = self.preprocessor.segment_signal(processed_signal, self.config.segment_length)
            # Use mean of segments for stability
            all_features = []
            for seg in segments:
                feat = self.feature_extractor.extract_features(seg)
                all_features.append(feat.all_features)
            features_df = pd.DataFrame(all_features).mean().to_frame().T
            # Use first segment's full feature object for structure
            features = self.feature_extractor.extract_features(segments[0])
        else:
            features = self.feature_extractor.extract_features(processed_signal)
            features_df = pd.DataFrame([features.all_features])
        
        # 3. Make predictions
        bp_pred = None
        glucose_pred = None
        stress_pred = None
        
        # BP prediction (requires acceptable quality)
        if self.bp_model.is_trained and signal_quality.overall_quality >= 50:
            try:
                sbp, dbp = self.bp_model.predict(features_df)
                sbp_val = float(sbp[0])
                dbp_val = float(dbp[0])
                map_val = dbp_val + (sbp_val - dbp_val) / 3.0
                pulse_press = sbp_val - dbp_val
                bp_pred = BPPrediction(
                    sbp=sbp_val,
                    dbp=dbp_val,
                    map=map_val,
                    pulse_pressure=pulse_press,
                    confidence=0.85,
                    quality_passed=signal_quality.is_acceptable
                )
            except Exception as e:
                print(f"BP prediction error: {e}")
        
        # Glucose prediction
        if self.glucose_model.is_trained and signal_quality.overall_quality >= 50:
            try:
                glucose = self.glucose_model.predict(features_df)
                glucose_val = float(glucose[0])
                hba1c = (glucose_val + 46.7) / 28.7  # Rough conversion
                glucose_pred = GlucosePrediction(
                    glucose=glucose_val,
                    hba1c_estimate=hba1c,
                    confidence=0.75,
                    quality_passed=signal_quality.is_acceptable
                )
            except Exception as e:
                print(f"Glucose prediction error: {e}")
        
        # Stress prediction
        if self.stress_model.is_trained and signal_quality.overall_quality >= 40:
            try:
                stress_preds, stress_probs = self.stress_model.predict(features_df)
                stress_level = str(stress_preds[0])
                # Map to labels if needed
                if hasattr(self.stress_model, 'class_labels') and stress_level.isdigit():
                    idx = int(stress_level)
                    stress_level = self.stress_model.class_labels[idx] if idx < len(self.stress_model.class_labels) else stress_level
                
                # Compute numeric score
                prob_dict = {}
                for i, label in enumerate(self.stress_model.class_labels):
                    prob_dict[label] = float(stress_probs[0][i]) if i < stress_probs.shape[1] else 0.0
                stress_score = prob_dict.get('high', 0)*100 + prob_dict.get('medium', 0)*50
                
                stress_pred = StressPrediction(
                    stress_level=stress_level,
                    stress_score=stress_score,
                    confidence=float(np.max(stress_probs[0])),
                    probabilities=prob_dict,
                    quality_passed=signal_quality.overall_quality >= 50
                )
            except Exception as e:
                print(f"Stress prediction error: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            bp=bp_pred,
            glucose=glucose_pred,
            stress=stress_pred,
            signal_quality=signal_quality,
            features=features,
            features_df=features_df,
            processing_time_ms=processing_time
        )
    
    def batch_predict(self, ppg_signals: List[np.ndarray]) -> List[PipelineResult]:
        """Predict for multiple signals."""
        return [self.predict(sig) for sig in ppg_signals]